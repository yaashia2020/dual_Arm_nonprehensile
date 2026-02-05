from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Any

import numpy as np
from pydrake.all import ContactModel, DiscreteContactApproximation, HydroelasticContactRepresentation


# ---------------------------
# Params (data only)
# ---------------------------

@dataclass(frozen=True)
class PredictionParams:
    dt: float = 0.01
    horizon: float = 0.2

    def num_steps(self) -> int:
        return int(self.horizon / self.dt)


@dataclass(frozen=True)
class ErgParams:
    # core ERG/DSM scalars
    robust_delta_tau: float = 0.1
    kappa_tau: float = 1.0
    robust_delta_q: float = 0.1
    kappa_q: float = 15.0
    robust_delta_dq: float = 0.1
    kappa_dq: float = 7.0
    robust_delta_dp_EE: float = 0.01
    kappa_dp_EE: float = 7.0
    kappa_terminal_energy: float = 7.5
    # regularization / update
    eta: float = 0.001
    zeta_q: float = 0.15
    delta_q: float = 0.1
    dt: float = 0.001          # ERG update dt
    FD: float = 1.0
    E_max: float = 0.20
    # soft navigation field
    soft_delta_s: float = 0.1
    soft_eta: float = 0.005


@dataclass(frozen=True)
class ContactParams:
    contact_model: Any = ContactModel.kHydroelasticWithFallback
    discrete_solver: Any = DiscreteContactApproximation.kSap
    mesh_type: Any = HydroelasticContactRepresentation.kTriangle


# ---------------------------
# Robot profile/spec (data only)
# ---------------------------

@dataclass(frozen=True)
class JointProfile:
    name: str
    U: int
    kp: np.ndarray
    kd: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray
    tau_max: np.ndarray


@dataclass(frozen=True)
class RobotSpec:
    name: str
    urdf_path: str
    q0: Optional[np.ndarray]
    num_positions: int
    num_velocities: int
    controlled_dofs: int               # U
    controlled_indices: np.ndarray     # length U, indices into plant q
    tracked_bodies: List[str]
    kp: np.ndarray
    kd: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray
    tau_max: np.ndarray
    limit_dp_trans: float = 1.7
    limit_dp_rot: float = 2.5

    def __post_init__(self):
        U = self.controlled_dofs
        if self.controlled_indices.shape != (U,):
            raise ValueError(f"controlled_indices must be shape ({U},), got {self.controlled_indices.shape}")
        for arr, name in [
            (self.kp, "kp"), (self.kd, "kd"),
            (self.q_min, "q_min"), (self.q_max, "q_max"),
            (self.dq_max, "dq_max"), (self.tau_max, "tau_max")
        ]:
            if arr.shape != (U,):
                raise ValueError(f"{name} must be shape ({U},), got {arr.shape}")


# ---------------------------
# Introspection helpers
# ---------------------------

def detect_panda_tracked_bodies_from_urdf(plant) -> List[str]:
    """
    Return a list of body names to track that actually exist in this URDF.
    Keeps ERG logic independent of URDF variant (fingers vs pad etc).
    """
    base = [
        "panda_link1", "panda_link2", "panda_link3", "panda_link4",
        "panda_link5", "panda_link6", "panda_link7", "panda_hand"
    ]
    extras_try = ["panda_leftfinger", "panda_rightfinger", "rubber_pad", "panda_rubber_pad", "gripper_pad"]
    tracked: List[str] = []
    for b in base:
        try:
            plant.GetBodyByName(b)
            tracked.append(b)
        except Exception:
            pass
    found_extra: List[str] = []
    for b in extras_try:
        try:
            plant.GetBodyByName(b)
            found_extra.append(b)
        except Exception:
            pass
    # prefer finger pair if available
    if "panda_leftfinger" in found_extra and "panda_rightfinger" in found_extra:
        tracked += ["panda_leftfinger", "panda_rightfinger"]
    elif len(found_extra) > 0:
        tracked.append(found_extra[0])
    return tracked


# ---------------------------
# Profiles + builders
# ---------------------------

def panda_arm7_profile() -> JointProfile:
    return JointProfile(
        name="panda_arm7",
        U=7,
        kp=np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0], dtype=float),
        kd=np.array([8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0], dtype=float),
        q_min=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=float),
        q_max=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=float),
        dq_max=np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=float),
        tau_max=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=float),
    )


def build_robot_spec_from_urdf(
    *,
    name: str,
    urdf_path: str,
    profile: JointProfile,
    q0: Optional[np.ndarray] = None,
    controlled_indices: Optional[np.ndarray] = None,
    tracked_bodies: Optional[List[str]] = None,
    limit_dp_trans: float = 1.7,
    limit_dp_rot: float = 2.5,
):
    """
    Builds a RobotSpec by introspecting the URDF via a tiny Drake plant.
    """
    from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, Parser

    urdf_path_abs = os.path.abspath(urdf_path)
    builder = DiagramBuilder()
    plant, _sg = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    model_instances = Parser(plant).AddModelsFromUrl("file://" + urdf_path_abs)
    # Drake returns a list; use the first model instance (single-robot URDF).
    model_instance = model_instances[0] if isinstance(model_instances, list) else model_instances
    plant.Finalize()

    num_positions = plant.num_positions(model_instance)
    num_velocities = plant.num_velocities(model_instance)
    U = profile.U
    # default: first U joints are controlled (true for most Panda URDFs)
    if controlled_indices is None:
        controlled_indices = np.arange(U, dtype=int)
    if tracked_bodies is None:
        tracked_bodies = detect_panda_tracked_bodies_from_urdf(plant)

    return RobotSpec(
        name=name,
        urdf_path=urdf_path_abs,
        q0=q0,
        num_positions=num_positions,
        num_velocities=num_velocities,
        controlled_dofs=U,
        controlled_indices=controlled_indices,
        tracked_bodies=tracked_bodies,
        kp=profile.kp,
        kd=profile.kd,
        q_min=profile.q_min,
        q_max=profile.q_max,
        dq_max=profile.dq_max,
        tau_max=profile.tau_max,
        limit_dp_trans=limit_dp_trans,
        limit_dp_rot=limit_dp_rot,
    )


def panda_spec(
    urdf_path: Optional[str] = None,
    *,
    name: str = "panda",
    controlled_indices: Optional[np.ndarray] = None,
    tracked_bodies: Optional[List[str]] = None,
) -> RobotSpec:
    if urdf_path is None:
        urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_drake.urdf")
        )
    default_q0 = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
    return build_robot_spec_from_urdf(
        name=name,
        urdf_path=urdf_path,
        profile=panda_arm7_profile(),
        q0=default_q0,
        controlled_indices=controlled_indices,
        tracked_bodies=tracked_bodies,
    )

