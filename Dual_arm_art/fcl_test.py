#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panda FR3 + movable_box (free body): FCL min-distance (robot ↔ movable_box only) + Meshcat visualization.

- ONE MultibodyPlant: Panda URDF + fixed_box (optional) + movable_box SDF
- movable_box is FREE (no weld). We place it via SetFreeBodyPose at (0.6, 0.0, 0.2)
- Panda joints set to a sample configuration
- FCL min signed distance using placeholder 10 cm cubes per link (sanity pipeline)
- Uses BodyIndex (not names) to avoid model-instance name collisions
- Meshcat visualization via AddDefaultVisualization(builder, meshcat)
"""

import os
import numpy as np
import fcl
from typing import List, Tuple

from pydrake.all import (
    AddMultibodyPlantSceneGraph, Parser, RigidTransform,
    StartMeshcat, Simulator, DiagramBuilder
)
from pydrake.multibody.tree import BodyIndex, ModelInstanceIndex
from pydrake.visualization import AddDefaultVisualization


# ------------- FCL helpers (index-based, name-safe) -------------
def build_collision_objects_for_model_by_index(
    plant, model_instance: ModelInstanceIndex
) -> List[Tuple[BodyIndex, str, np.ndarray, fcl.CollisionObject]]:
    """
    Make one 10 cm cube per link of a given model_instance.
    Returns list of (BodyIndex, body_name, T_LC, CollisionObject).
    Skips the world body.
    """
    objs: List[Tuple[BodyIndex, str, np.ndarray, fcl.CollisionObject]] = []
    world_idx = plant.world_body().index()
    for bidx in plant.GetBodyIndices(model_instance):
        if bidx == world_idx:
            continue
        body = plant.get_body(bidx)
        name = body.name()
        geom = fcl.Box(0.1, 0.1, 0.1)  # placeholder cube
        T_LC = np.eye(4)
        objs.append((bidx, name, T_LC, fcl.CollisionObject(geom)))
    if not objs:
        # fallback dummy
        objs.append((plant.world_body().index(), "default", np.eye(4),
                     fcl.CollisionObject(fcl.Box(0.1, 0.1, 0.1))))
    return objs

def update_fcl_objects_from_context_by_index(plant, context, link_objs):
    """Push Drake world poses into FCL objects using BodyIndex (unambiguous)."""
    for bidx, _name, T_LC, obj in link_objs:
        if bidx == plant.world_body().index():
            continue
        body = plant.get_body(bidx)
        T_WL = plant.EvalBodyPoseInWorld(context, body).GetAsMatrix4()
        T_WC = T_WL @ T_LC
        obj.setTransform(fcl.Transform(T_WC[:3, :3], T_WC[:3, 3]))

def min_signed_distance_between_sets(
    objs_a: List[Tuple[BodyIndex, str, np.ndarray, fcl.CollisionObject]],
    objs_b: List[Tuple[BodyIndex, str, np.ndarray, fcl.CollisionObject]],
):
    """Brute-force min signed distance across link pairs (negative => penetration)."""
    req = fcl.DistanceRequest(enable_signed_distance=True, enable_nearest_points=True)
    best = {"d": np.inf, "a_link": None, "b_link": None,
            "nearest_on_a": None, "nearest_on_b": None, "normal_A_to_B": None}
    for (_ia, name_a, _Ta, a_obj) in objs_a:
        for (_ib, name_b, _Tb, b_obj) in objs_b:
            res = fcl.DistanceResult()
            d = fcl.distance(a_obj, b_obj, req, res)
            if d < best["d"]:
                # try to read nearest points robustly across python-fcl versions
                pA = pB = None
                try:
                    pA = np.array(res.nearest_points[0], dtype=float)
                    pB = np.array(res.nearest_points[1], dtype=float)
                except Exception:
                    pass
                normal = None
                if pA is not None and pB is not None:
                    v = pB - pA
                    n = np.linalg.norm(v)
                    if n > 1e-12:
                        normal = v / n
                best.update(d=d, a_link=name_a, b_link=name_b,
                            nearest_on_a=pA, nearest_on_b=pB,
                            normal_A_to_B=normal)
    return best


def get_model_instance_by_name_or_first(plant, preferred_names: List[str]) -> ModelInstanceIndex:
    for nm in preferred_names:
        try:
            return plant.GetModelInstanceByName(nm)
        except Exception:
            pass
    # fallback: first non-world body’s model instance
    for i in range(plant.num_bodies()):
        b = plant.get_body(BodyIndex(i))
        if b.index() != plant.world_body().index():
            return b.model_instance()
    raise RuntimeError("No valid model instance found.")


# ---------------- Main ----------------
if __name__ == "__main__":
    # Paths fixed to your repo structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_panda = os.path.abspath(os.path.join(script_dir, "../models/robots/panda_fr3/urdf/panda_fr3.urdf"))
    sdf_fixed  = os.path.abspath(os.path.join(script_dir, "../models/boxes/fixed_box.sdf"))
    sdf_movable= os.path.abspath(os.path.join(script_dir, "../models/boxes/movable_box.sdf"))

    print("Panda URDF :", urdf_panda)
    print("Fixed  SDF :", sdf_fixed, "(exists:" , os.path.exists(sdf_fixed), ")")
    print("MovableSDF :", sdf_movable)
    if not os.path.exists(urdf_panda):
        raise FileNotFoundError(f"Missing: {urdf_panda}")
    if not os.path.exists(sdf_movable):
        raise FileNotFoundError(f"Missing: {sdf_movable}")

    # One plant + scene graph
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)

    # Load Panda first
    parser.AddModelsFromUrl("file://" + urdf_panda)

    # Load fixed box only if file exists (static ground). Keep as you had.
    if os.path.exists(sdf_fixed):
        parser.AddModelsFromUrl("file://" + sdf_fixed)
        print("Added fixed_box (optional)")

    # Load movable box (FREE body)
    parser.AddModelsFromUrl("file://" + sdf_movable)
    print("Added movable_box")

    plant.Finalize()

    # Model instances
    mi_robot = get_model_instance_by_name_or_first(plant, ["panda", "panda_fr3", "franka", "fr3"])
    # Optional fixed box instance (we won't log distances to it)
    mi_fixed = None
    try:
        mi_fixed = plant.GetModelInstanceByName("fixed_box")
    except Exception:
        mi_fixed = None
    mi_movable = get_model_instance_by_name_or_first(plant, ["movable_box", "box"])

    # Meshcat viz
    meshcat = StartMeshcat()
    AddDefaultVisualization(builder, meshcat)

    diagram = builder.Build()
    sim = Simulator(diagram)
    diagram_ctx = sim.get_mutable_context()
    plant_ctx = diagram.GetMutableSubsystemContext(plant, diagram_ctx)

    # Sample Panda joint config
    npos_robot = plant.num_positions(mi_robot)
    qa = np.zeros(npos_robot, dtype=float)
    if qa.size >= 2:
        qa[0] = 0.30
        qa[1] = -0.50
    plant.SetPositions(plant_ctx, mi_robot, qa)

    # Place movable box (must be non-static in SDF)
    try:
        box_body = plant.GetBodyByName("box_link", mi_movable)
        plant.SetFreeBodyPose(plant_ctx, box_body, RigidTransform([0.6, 0.0, 0.2]))
        print("Movable box pose set to [0.6, 0.0, 0.2]")
    except Exception as e:
        print(f"Warning: movable_box SetFreeBodyPose failed (is SDF static?): {e}")

    # -------- FCL: robot ↔ movable_box only --------
    objs_robot   = build_collision_objects_for_model_by_index(plant, mi_robot)
    objs_movable = build_collision_objects_for_model_by_index(plant, mi_movable)
    # (We intentionally do NOT build pairs with fixed box to keep logs clean)

    update_fcl_objects_from_context_by_index(plant, plant_ctx, objs_robot)
    update_fcl_objects_from_context_by_index(plant, plant_ctx, objs_movable)

    result = min_signed_distance_between_sets(objs_robot, objs_movable)

    print("\n=== Panda FR3 ↔ Movable Box: min signed distance (placeholder cubes) ===")
    print(f"d (signed): {result['d']:.6f} m   (negative ⇒ penetration)")
    print(f"A link    : {result['a_link']}")
    print(f"B link    : {result['b_link']}")
    print(f"nearest A : {result['nearest_on_a']}")
    print(f"nearest B : {result['nearest_on_b']}")
    print(f"normal A→B: {result['normal_A_to_B']}")

    # -------- brief viz --------
    sim.set_target_realtime_rate(1.0)
    sim.set_publish_every_time_step(True)
    sim.Initialize()
    print("\nMeshcat is publishing frames now...")
    sim.AdvanceTo(5.0)
    print("Done.")
