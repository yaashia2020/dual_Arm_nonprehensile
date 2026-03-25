
# control_blocks/integrate_z_two_in.py
from __future__ import annotations
from typing import Optional, Literal
from pydrake.all import (
    DiagramBuilder, Adder, Gain, Integrator, Saturation,
    Demultiplexer, Multiplexer, PassThrough, LeafSystem
)

class ContactGate(LeafSystem):
    """
    Gates the error signal based on contact flag.
    
    Modes:
    - "binary": contact > 0.5 → pass, else → 0
    - "continuous": contact directly scales the error (0.0 to 1.0)
    - "threshold": contact > threshold → pass, else → 0
    """
    def __init__(self, mode="continuous", threshold=0.5):
        super().__init__()
        self.mode = mode
        self.threshold = threshold
        self.DeclareVectorInputPort("error", size=1)
        self.DeclareVectorInputPort("contact", size=1)
        self.DeclareVectorOutputPort("gated_error", size=1, calc=self.CalcOutput)
    
    def CalcOutput(self, context, output):
        error = self.GetInputPort("error").Eval(context)[0]
        contact = self.GetInputPort("contact").Eval(context)[0]
        
        if self.mode == "binary":
            # Binary: contact > 0.5 means contact exists
            gated = error if contact > 0.5 else 0.0
        elif self.mode == "continuous":
            # Continuous: contact value scales the error (0.0 to 1.0)
            gated = error * contact
        elif self.mode == "threshold":
            # Custom threshold
            gated = error if contact > self.threshold else 0.0
        else:
            # Default to continuous
            gated = error * contact
        
        output.set_value([gated])

def _make_integral_limiter_1d(
    Ki: float,
    u_min: Optional[float] = None,
    u_max: Optional[float] = None,
    Kaw: Optional[float] = None,
    name: str = "IntegralLimiter1D",
):
    """
    1-D block:
      xdot = Ki*e  (+ Kaw * (u - x) if Kaw is not None)
      u_unsat = x
      u = sat(u_unsat, u_min, u_max) if limits are provided else u_unsat

    Ports:
      in 0: "e"        (scalar)
      out 0: "u"       (scalar)         -- saturated integral
      out 1: "u_unsat" (scalar)         -- pre-saturation (integrator state)
    """
    b = DiagramBuilder()
    g_ki = b.AddSystem(Gain(float(Ki), 1))
    integ = b.AddSystem(Integrator(1))

    if u_min is not None and u_max is not None:
        sat = b.AddSystem(Saturation([u_min], [u_max]))
    else:
        sat = b.AddSystem(PassThrough(1))  # no clamp → pass-through

    # e -> Ki -> (+) -> Integrator -> (sat)
    adder = b.AddSystem(Adder(1 if Kaw is None else 2, 1))
    b.ExportInput(g_ki.get_input_port(), "e")
    b.Connect(g_ki.get_output_port(), adder.get_input_port(0))
    b.Connect(adder.get_output_port(), integ.get_input_port())
    b.Connect(integ.get_output_port(), sat.get_input_port())

    b.ExportOutput(sat.get_output_port(), "u")
    b.ExportOutput(integ.get_output_port(), "u_unsat")

    if Kaw is not None:
        kaw = b.AddSystem(Gain(float(Kaw), 1))
        diff = b.AddSystem(Adder(2, 1))   # (u - u_unsat)
        neg1 = b.AddSystem(Gain(-1.0, 1))
        b.Connect(sat.get_output_port(), diff.get_input_port(0))
        b.Connect(integ.get_output_port(), neg1.get_input_port())
        b.Connect(neg1.get_output_port(), diff.get_input_port(1))
        b.Connect(diff.get_output_port(), kaw.get_input_port())
        b.Connect(kaw.get_output_port(), adder.get_input_port(1))

    diagram = b.Build(); diagram.set_name(name)
    return diagram


def make_integrate_z_two_in_block(
    Ki_z: float,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    Kaw_z: Optional[float] = None,
    *,
    error_mode: Literal["a_minus_b", "b_minus_a"] = "a_minus_b",
    passthrough_xy_from: Literal["a", "b"] = "a",
    name: str = "IntegrateZ_TwoIn_3vec",
):
    """
    Block: two 3-vectors in → one 3-vector out.

    Inputs:
      in 0: 'a'  (R^3)  e.g., reference or first pose
      in 1: 'b'  (R^3)  e.g., measured or second pose
      in 2: 'contact'  (scalar) 1 if contact exists, 0 otherwise

    Computation:
      ez = a.z - b.z          if error_mode == "a_minus_b"
         = b.z - a.z          if error_mode == "b_minus_a"
      gated_ez = ez if contact > 0.5 else 0
      z_out = ∫ (Ki_z * gated_ez) dt   (optionally clamped to [z_min, z_max])
      xy passthrough = from 'a' or 'b' per `passthrough_xy_from`

    Outputs:
      out 0: 'u'         (R^3) = [x_passthrough, y_passthrough, z_out]
      out 1: 'uz_unsat'  (scalar) raw integrated z before clamp (for logging)
      out 2: 'ez'        (scalar) the z-axis error used for integration

    Notes:
      - Set z_min/z_max to enable clamp (anti-windup via Kaw_z optional).
      - Keep Ki_z, Kaw_z moderate; start with Kaw_z ≈ Ki_z.
      - Integration only occurs when contact > 0.5
    """
    b = DiagramBuilder()

    # Demux both inputs a, b (each 3 → scalars)
    demux_a = b.AddSystem(Demultiplexer(3))
    demux_b = b.AddSystem(Demultiplexer(3))
    b.ExportInput(demux_a.get_input_port(), "a")
    b.ExportInput(demux_b.get_input_port(), "b")

    # Choose which xy to pass through
    px = b.AddSystem(PassThrough(1))
    py = b.AddSystem(PassThrough(1))
    if passthrough_xy_from == "a":
        b.Connect(demux_a.get_output_port(0), px.get_input_port())
        b.Connect(demux_a.get_output_port(1), py.get_input_port())
    else:
        b.Connect(demux_b.get_output_port(0), px.get_input_port())
        b.Connect(demux_b.get_output_port(1), py.get_input_port())

    # z-error = a.z - b.z  (or inverted)
    diff = b.AddSystem(Adder(2, 1))   # (+az) + (-bz)
    neg1 = b.AddSystem(Gain(-1.0, 1))
    if error_mode == "a_minus_b":
        b.Connect(demux_a.get_output_port(2), diff.get_input_port(0))
        b.Connect(demux_b.get_output_port(2), neg1.get_input_port())
    else:  # b_minus_a
        b.Connect(demux_b.get_output_port(2), diff.get_input_port(0))
        b.Connect(demux_a.get_output_port(2), neg1.get_input_port())
    b.Connect(neg1.get_output_port(), diff.get_input_port(1))

    # Gate the error based on contact - binary mode
    # Binary: integration only when contact > 0.5 (i.e., contact = 1.0)
    gate = b.AddSystem(ContactGate(mode="binary", threshold=0.5))
    b.ExportInput(gate.GetInputPort("contact"), "contact")
    b.Connect(diff.get_output_port(), gate.GetInputPort("error"))
    
    # 1-D z integrator with clamp/anti-windup
    iz = b.AddSystem(_make_integral_limiter_1d(
        Ki=Ki_z, u_min=z_min, u_max=z_max, Kaw=Kaw_z, name="I_z"
    ))
    b.Connect(gate.GetOutputPort("gated_error"), iz.get_input_port(0))  # gated ez → integrator

    # Recombine to 3-vector: [x_passthrough, y_passthrough, z_integrated]
    mux = b.AddSystem(Multiplexer(3))
    b.Connect(px.get_output_port(), mux.get_input_port(0))
    b.Connect(py.get_output_port(), mux.get_input_port(1))
    b.Connect(iz.get_output_port(0), mux.get_input_port(2))

    # Export outputs
    b.ExportOutput(mux.get_output_port(), "u")
    b.ExportOutput(iz.get_output_port(1), "uz_unsat")
    b.ExportOutput(diff.get_output_port(), "ez")

    diagram = b.Build(); diagram.set_name(name)
    return diagram
