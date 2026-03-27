"""
Standalone test for make_integrate_z_two_in_block.

No robot, no URDF, no RelaxedIK — purely synthetic signals.
Runs in ~2 seconds and prints PASS/FAIL per phase.

Phases:
  A  0–2s  : contact=0, box at target         → u[2] stays at 0
  B  2–6s  : contact=1, box 0.4m below        → u[2] ramps up
  C  6–8s  : contact=1, box slips further      → u[2] ramps faster
  D  8–10s : contact=0                         → u[2] freezes
  E  10–12s: contact=1, large ez               → u[2] saturates at z_max, uz_unsat stays close (anti-windup)
  F  12–15s: contact=1, ez reversed (box high) → u[2] unwinds from z_max back toward 0
"""

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import DiagramBuilder, LeafSystem, LogVectorOutput, Simulator

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from z_axis_integrator import make_integrate_z_two_in_block

# ---------------------------------------------------------------------------
# Synthetic signal sources
# ---------------------------------------------------------------------------

class SyntheticTarget(LeafSystem):
    """Constant target: [0, 0, z_target]."""
    def __init__(self, z_target: float):
        super().__init__()
        self.z_target = z_target
        self.DeclareVectorOutputPort("target", size=3, calc=self.Calc)

    def Calc(self, context, output):
        output.SetFromVector([0.0, 0.0, self.z_target])


class SyntheticBox(LeafSystem):
    """
    Time-varying box Z:
      Phase A  0–2s  : z_b = 0.95   (at target,     ez =  0.0)
      Phase B  2–6s  : z_b = 0.55   (steady slip,   ez =  0.4)
      Phase C  6–8s  : z_b 0.55→0.35 (worsening,   ez = 0.4→0.6)
      Phase D  8–10s : z_b = 0.35   (contact off,   ez =  0.6)
      Phase E  10–12s: z_b = -4.05  (large slip,    ez =  5.0) → saturates z_max
      Phase F  12–15s: z_b = 5.95   (box above ref, ez = -5.0) → unwinds
    """
    def __init__(self):
        super().__init__()
        self.DeclareVectorOutputPort("box_pos", size=3, calc=self.Calc)

    def Calc(self, context, output):
        t = context.get_time()
        if t < 2.0:
            z_b = 0.95
        elif t < 6.0:
            z_b = 0.55
        elif t < 8.0:
            z_b = 0.55 - 0.20 * (t - 6.0) / 2.0   # 0.55 → 0.35
        elif t < 10.0:
            z_b = 0.35
        elif t < 12.0:
            z_b = -4.05   # ez = 0.95 - (-4.05) = 5.0 → fast saturation
        else:
            z_b = 5.95    # ez = 0.95 - 5.95 = -5.0 → fast unwind
        output.SetFromVector([0.0, 0.0, z_b])


class SyntheticContact(LeafSystem):
    """Contact = 1 during [2–8s] and [10–15s], else 0."""
    def __init__(self):
        super().__init__()
        self.DeclareVectorOutputPort("contact", size=1, calc=self.Calc)

    def Calc(self, context, output):
        t = context.get_time()
        on = (2.0 <= t < 8.0) or (t >= 10.0)
        output.SetFromVector([1.0 if on else 0.0])


# ---------------------------------------------------------------------------
# Build diagram
# ---------------------------------------------------------------------------
builder = DiagramBuilder()

z_integrator = builder.AddNamedSystem(
    "ZAxisIntegrator",
    make_integrate_z_two_in_block(
        Ki_z=0.7,
        z_min=0.0,
        z_max=5.0,
        Kaw_z=0.7,
        error_mode="a_minus_b",
        passthrough_xy_from="a",
        name="ZAxisIntegrator",
    ),
)

target_src  = builder.AddSystem(SyntheticTarget(z_target=0.95))
box_src     = builder.AddSystem(SyntheticBox())
contact_src = builder.AddSystem(SyntheticContact())

builder.Connect(target_src.GetOutputPort("target"),    z_integrator.GetInputPort("a"))
builder.Connect(box_src.GetOutputPort("box_pos"),      z_integrator.GetInputPort("b"))
builder.Connect(contact_src.GetOutputPort("contact"),  z_integrator.GetInputPort("contact"))

log_u       = LogVectorOutput(z_integrator.GetOutputPort("u"),       builder)
log_uz_unsat= LogVectorOutput(z_integrator.GetOutputPort("uz_unsat"),builder)
log_ez      = LogVectorOutput(z_integrator.GetOutputPort("ez"),      builder)
log_contact = LogVectorOutput(contact_src.GetOutputPort("contact"),  builder)

diagram = builder.Build()

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
simulator = Simulator(diagram)
simulator.Initialize()
simulator.AdvanceTo(15.0)

ctx          = simulator.get_context()
t            = log_u.FindLog(ctx).sample_times()
data_u       = log_u.FindLog(ctx).data().T        # (N, 3)
data_uz_unsat= log_uz_unsat.FindLog(ctx).data().T # (N, 1)
data_ez      = log_ez.FindLog(ctx).data().T       # (N, 1)
data_c       = log_contact.FindLog(ctx).data().T  # (N, 1)

u_z      = data_u[:, 2]
uz_unsat = data_uz_unsat[:, 0]
ez       = data_ez[:, 0]
c        = data_c[:, 0]

# ---------------------------------------------------------------------------
# Phase masks
# ---------------------------------------------------------------------------
mask_A = t <  2.0
mask_B = (t >= 2.0)  & (t < 6.0)
mask_C = (t >= 6.0)  & (t < 8.0)
mask_D = (t >= 8.0)  & (t < 10.0)
mask_E = (t >= 10.0) & (t < 12.0)
mask_F = t >= 12.0

# ---------------------------------------------------------------------------
# PASS / FAIL checks
# ---------------------------------------------------------------------------
tol    = 0.05
z_max  = 5.0

results = {}

# Phase A: no contact → u[2] stays at 0
if mask_A.any():
    results["A: no-contact → u[2] stays 0"] = np.max(np.abs(u_z[mask_A])) < tol

# Phase B: contact + slip → u[2] ramps up
if mask_B.any():
    results["B: slip → u[2] ramps up"] = u_z[mask_B][-1] > u_z[mask_B][0] + tol

# Phase C: worsening slip → ramps faster than B
if mask_B.any() and mask_C.any():
    rate_B = (u_z[mask_B][-1] - u_z[mask_B][0]) / (t[mask_B][-1] - t[mask_B][0] + 1e-9)
    rate_C = (u_z[mask_C][-1] - u_z[mask_C][0]) / (t[mask_C][-1] - t[mask_C][0] + 1e-9)
    print(f"  [INFO] Phase B ramp rate: {rate_B:.4f} m/s  (expected ~0.28,  Ki*ez = 0.7*0.4)")
    print(f"  [INFO] Phase C ramp rate: {rate_C:.4f} m/s  (expected ~0.35,  Ki*avg_ez = 0.7*0.5)")
    results["C: more slip → u[2] ramps faster"] = rate_C > rate_B - tol

# Phase D: contact lost → u[2] freezes
if mask_D.any():
    drift_D = np.max(u_z[mask_D]) - np.min(u_z[mask_D])
    results["D: contact lost → u[2] freezes"] = drift_D < tol

# Phase E: large ez → u[2] clamps at z_max (doesn't exceed)
if mask_E.any():
    max_uz_E = np.max(u_z[mask_E])
    results["E: saturation → u[2] ≤ z_max"] = max_uz_E <= z_max + tol
    # Anti-windup: uz_unsat should stay close to u (not diverge above z_max)
    max_windup = np.max(uz_unsat[mask_E] - u_z[mask_E])
    print(f"  [INFO] Phase E max windup (uz_unsat - u): {max_windup:.4f}  (small = anti-windup working)")
    results["E: anti-windup → uz_unsat stays near u"] = max_windup < 1.0

# Phase F: ez reversed → u[2] unwinds from z_max back toward 0
if mask_F.any():
    results["F: reversed error → u[2] unwinds"] = u_z[mask_F][-1] < u_z[mask_F][0] - tol

print("\n=== Z Integrator Test Results ===")
all_pass = True
for name, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed:
        all_pass = False
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}\n")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

axs[0].plot(t, ez, color="steelblue", linewidth=2)
axs[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
axs[0].set_ylabel("ez  (a.z − b.z)")
axs[0].set_title("Z Error (before contact gate)")
axs[0].grid(True, alpha=0.3)

axs[1].plot(t, u_z,      color="darkorange", linewidth=2, label="u[2] (saturated)")
axs[1].plot(t, uz_unsat, color="purple",     linewidth=1.5, linestyle="--", label="uz_unsat (raw integrator)")
axs[1].axhline(z_max, color="red", linestyle="--", alpha=0.6, label=f"z_max={z_max}")
axs[1].axhline(0,     color="gray", linestyle="--", alpha=0.4)
axs[1].set_ylabel("Z value (m)")
axs[1].set_title("Integrator Output  (gap between lines = windup)")
axs[1].legend(fontsize=8, loc="upper left")
axs[1].grid(True, alpha=0.3)

axs[2].plot(t, uz_unsat - u_z, color="crimson", linewidth=2)
axs[2].axhline(0, color="gray", linestyle="--", alpha=0.5)
axs[2].set_ylabel("uz_unsat − u[2]")
axs[2].set_title("Windup gap  (should stay small with Kaw_z=0.7)")
axs[2].grid(True, alpha=0.3)

axs[3].plot(t, c, color="green", linewidth=2, drawstyle="steps-post")
axs[3].set_ylim(-0.1, 1.2)
axs[3].set_ylabel("contact")
axs[3].set_xlabel("Time (s)")
axs[3].set_title("Contact Flag")
axs[3].grid(True, alpha=0.3)

for ax in axs:
    for boundary, label in [(2.0,"A→B"), (6.0,"B→C"), (8.0,"C→D"), (10.0,"D→E"), (12.0,"E→F")]:
        ax.axvline(boundary, color="red", linestyle=":", alpha=0.4)

fig.suptitle("Z-Axis Integrator: Slip Detection + Anti-Windup Test", fontsize=13)
fig.tight_layout()
plt.show()
