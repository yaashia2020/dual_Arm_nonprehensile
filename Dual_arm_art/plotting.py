import numpy as np
from pydrake.all import LeafSystem
from CERG_Setup import RobotSpec
from dataclasses import dataclass
from typing import List, Dict
from pydrake.systems.framework import OutputPort, DiagramBuilder
from pydrake.systems.primitives import Multiplexer, LogVectorOutput


class BodyPoseExtractor(LeafSystem):
    """
    Generic body pose extractor for logging.

    - Input: robot state (positions + velocities) sized to the given model instance.
    - Output: world-frame translation (x, y, z) of the chosen body.
    """

    def __init__(self, plant, model_instance, body_name: str, robot_spec: RobotSpec | None = None):
        super().__init__()
        self.plant = plant
        self.model_instance = model_instance
        self.body_name = body_name

        # Use sizes from robot spec when provided; fall back to plant introspection.
        if robot_spec is not None and robot_spec.num_positions is not None:
            self.nq = robot_spec.num_positions
            self.nv = robot_spec.num_velocities
        else:
            self.nq = plant.num_positions(model_instance)
            self.nv = plant.num_velocities(model_instance)

        # Input: full state (q, v)
        self.DeclareVectorInputPort("joint_positions", size=self.nq + self.nv)
        # Output: body world position
        self.DeclareVectorOutputPort("body_world_positions", size=3, calc=self.CalcOutput)

        self.temp_context = plant.CreateDefaultContext()
        self.body = plant.GetBodyByName(body_name, model_instance)

    def CalcOutput(self, context, output):
        full_state = self.GetInputPort("joint_positions").Eval(context)
        q = np.array(full_state[: self.nq], dtype=float)

        self.plant.SetPositions(self.temp_context, self.model_instance, q)
        pose = self.plant.EvalBodyPoseInWorld(self.temp_context, self.body)
        translation = pose.translation()
        output.SetFromVector([translation[0], translation[1], translation[2]])


@dataclass(frozen=True)
class LogSignal:
    name: str
    port: OutputPort


class MultiLog:
    def __init__(self, log_sys, slices: Dict[str, slice]):
        self._log_sys = log_sys
        self._slices = slices

    def data(self, diagram_context):
        log = self._log_sys.FindLog(diagram_context)
        Y = log.data()
        t = log.sample_times()
        return Y, t

    def get(self, diagram_context, name: str):
        Y, t = self.data(diagram_context)
        return Y[self._slices[name], :], t


def add_mux_logger(
    builder: DiagramBuilder,
    signals: List[LogSignal],
    sample_period: float,
    name: str = "mux_logger",
) -> MultiLog:
    sizes = [s.port.size() for s in signals]
    mux = builder.AddNamedSystem(
        f"{name}_mux",
        Multiplexer(sizes),
    )

    slices: Dict[str, slice] = {}
    offset = 0
    for i, sig in enumerate(signals):
        builder.Connect(sig.port, mux.get_input_port(i))
        slices[sig.name] = slice(offset, offset + sig.port.size())
        offset += sig.port.size()

    log_sys = LogVectorOutput(
        mux.get_output_port(),
        builder,
        sample_period,
    )

    try:
        log_sys.set_name(name)
    except Exception:
        pass

    return MultiLog(log_sys, slices)

