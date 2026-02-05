import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser


@dataclass(frozen=True)
class FKResult:
    target_xyz: np.ndarray
    fk_xyz: np.ndarray
    error_xyz: np.ndarray
    error_norm: float


class DrakeFK:
    """
    Drake forward-kinematics helper to compare RelaxedIK joint solutions against a target xyz.
    """

    def __init__(self, *, urdf_path: str, body_name: str = "rubber_pad"):
        self.urdf_path = os.path.abspath(urdf_path)
        self.body_name = body_name

        builder = DiagramBuilder()
        # Continuous plant is fine for FK.
        self.plant, _sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        model_instances = Parser(self.plant).AddModelsFromUrl("file://" + self.urdf_path)
        self.model_instance = model_instances[0] if isinstance(model_instances, list) else model_instances
        self.plant.Finalize()

        self.context = self.plant.CreateDefaultContext()
        self.body = self.plant.GetBodyByName(self.body_name, self.model_instance)
        self.nq = self.plant.num_positions(self.model_instance)

    def fk_xyz(self, q: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size < self.nq:
            q_use = np.concatenate([q, np.zeros(self.nq - q.size)])
        else:
            q_use = q[: self.nq]
        self.plant.SetPositions(self.context, self.model_instance, q_use)
        pose = self.plant.EvalBodyPoseInWorld(self.context, self.body)
        return np.asarray(pose.translation(), dtype=float).reshape(3)

    def compare(self, *, q: Sequence[float], target_xyz: Sequence[float]) -> FKResult:
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        fk_xyz = self.fk_xyz(q)
        err = fk_xyz - target_xyz
        return FKResult(
            target_xyz=target_xyz,
            fk_xyz=fk_xyz,
            error_xyz=err,
            error_norm=float(np.linalg.norm(err)),
        )


