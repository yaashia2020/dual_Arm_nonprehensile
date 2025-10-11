
"""
Panda FR3 + movable_box (free body): FCL min-distance (robot ↔ movable_box only) + Meshcat visualization.

- ONE MultibodyPlant: Panda URDF + fixed_box (optional) + movable_box SDF
- movable_box is FREE (no weld). We place it via SetFreeBodyPose at (0.6, 0.0, 0.2)
- Panda joints set to a sample configuration
- FCL min signed distance using placeholder 10 cm cubes per link (sanity pipeline)
- Uses BodyIndex (not names) to avoid model-instance name collisions
- Meshcat visualization via AddDefaultVisualization(builder, meshcat)
- NEW: Meshcat visualization of the same per-link FCL proxy boxes (no manual wiring needed)
"""

import os
import numpy as np
import fcl
import matplotlib.pyplot as plt
from typing import List, Tuple

from pydrake.all import *
from pydrake.multibody.tree import BodyIndex, ModelInstanceIndex
from pydrake.visualization import AddDefaultVisualization
from pydrake.geometry import Box as DrakeBox, Rgba  # for viz boxes

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


# ------------- LeafSystem: FCL link distances to movable box -------------
from pydrake.systems.framework import LeafSystem, BasicVector

class FCLLinkDistanceSystem(LeafSystem):
    """
    Outputs signed distances (m) from selected robot links to the movable box using FCL.
    Links: panda_link7, panda_hand, panda_leftfinger, panda_rightfinger
    Output: [d_link7, d_hand, d_leftfinger, d_rightfinger]
    """
    def __init__(self, plant, robot_instance: ModelInstanceIndex, box_instance: ModelInstanceIndex):
        super().__init__()
        self._plant = plant
        self._robot_instance = robot_instance
        self._box_instance = box_instance

        # Input: full plant state (q; v)
        state_size = plant.num_positions() + plant.num_velocities()
        self._x_port = self.DeclareVectorInputPort("x", BasicVector(state_size))

        # Output: 4 distances
        self.DeclareVectorOutputPort("distances", BasicVector(4), self._calc_output)

        # Temp context for pose queries
        self._tmp_ctx = plant.CreateDefaultContext()

        # Resolve bodies and build FCL objects for the four links
        self._link_names = [
            "panda_link7",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ]
        self._robot_links: List[Tuple[BodyIndex, str, np.ndarray, fcl.CollisionObject]] = []
        for nm in self._link_names:
            try:
                body = plant.GetBodyByName(nm, robot_instance)
                geom = fcl.Box(0.10, 0.10, 0.10)  # 10 cm cube placeholder per link
                self._robot_links.append((body.index(), nm, np.eye(4), fcl.CollisionObject(geom)))
            except Exception:
                # If a body is missing, append a dummy placeholder tied to world
                self._robot_links.append((plant.world_body().index(), nm, np.eye(4), fcl.CollisionObject(fcl.Box(0.10, 0.10, 0.10))))

        # Build FCL objects for the box (single body: box_link)
        self._box_links: List[Tuple[BodyIndex, str, np.ndarray, fcl.CollisionObject]] = []
        try:
            box_body = plant.GetBodyByName("box_link", box_instance)
            self._box_links.append((box_body.index(), "box_link", np.eye(4), fcl.CollisionObject(fcl.Box(0.22, 0.30, 0.20))))
        except Exception:
            # Fallback dummy
            self._box_links.append((plant.world_body().index(), "box_link", np.eye(4), fcl.CollisionObject(fcl.Box(0.22, 0.30, 0.20))))

    def _calc_output(self, context, output):
        # Pull full plant state and sync temp context
        x = self._x_port.Eval(context)
        try:
            self._plant.SetPositionsAndVelocities(self._tmp_ctx, x)
        except Exception:
            # Fallback: split q and v
            nq = self._plant.num_positions()
            self._plant.SetPositions(self._tmp_ctx, x[:nq])
            self._plant.SetVelocities(self._tmp_ctx, x[nq:])

        # Update FCL object transforms for robot links
        update_fcl_objects_from_context_by_index(self._plant, self._tmp_ctx, self._robot_links)
        # Update for box
        update_fcl_objects_from_context_by_index(self._plant, self._tmp_ctx, self._box_links)

        # Compute distances per link (robot link vs any box geometry)
        dvals = []
        for (bidx, nm, T_LC, obj) in self._robot_links:
            res = min_signed_distance_between_sets([(bidx, nm, T_LC, obj)], self._box_links)
            dvals.append(float(res["d"]))

        # Print distances
        # print(f"FCL distances: link7={dvals[0]:.4f}, hand={dvals[1]:.4f}, leftfinger={dvals[2]:.4f}, rightfinger={dvals[3]:.4f}")
        
        output.SetFromVector(dvals[:4])


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

# ------------- NEW: Meshcat viz of the same FCL proxy boxes -------------
class MeshcatBoxesPublisher(LeafSystem):
    """
    Publishes/upgrades Meshcat box poses every step for selected links (and the movable box).
    Non-intrusive: doesn't touch plant geometry; purely a viz overlay.
    """
    def __init__(self, plant, meshcat, robot_instance, box_instance):
        super().__init__()
        self._plant = plant
        self._meshcat = meshcat
        self._robot_instance = robot_instance
        self._box_instance = box_instance
        # Temp plant context for kinematics evals
        self._tmp_ctx = plant.CreateDefaultContext()
        # Input: full plant state (q; v)
        state_size = plant.num_positions() + plant.num_velocities()
        self._x_port = self.DeclareVectorInputPort("x", BasicVector(state_size))

        self._robot_link_cfg = {
            "panda_link7":      [{"size": (0.10, 0.10, 0.10), "rpy": (0,0,0), "p": (0,0,0)}],
            "panda_hand":       [{"size": (0.10, 0.10, 0.10), "rpy": (0,0,0), "p": (0,0,0)}],
            "panda_leftfinger": [{"size": (0.10, 0.10, 0.10), "rpy": (0,0,0), "p": (0,0,0)}],
            "panda_rightfinger":[{"size": (0.10, 0.10, 0.10), "rpy": (0,0,0), "p": (0,0,0)}],
        }
        self._box_link_cfg = {
            "box_link": [{"size": (0.22, 0.30, 0.20), "rpy": (0,0,0), "p": (0,0,0)}]
        }

        self._entries = []
        self._entries += self._register_for_instance(self._robot_instance, self._robot_link_cfg, ns="/fcl_boxes_robot", color=Rgba(0.1, 0.6, 1.0, 0.35))
        self._entries += self._register_for_instance(self._box_instance,   self._box_link_cfg,   ns="/fcl_boxes_box",   color=Rgba(1.0, 0.4, 0.2, 0.35))

        # Callbacks in some Drake versions pass only (context); others (context, event).
        self.DeclarePerStepPublishEvent(self._do_publish)

    def _register_for_instance(self, model_instance, cfg_dict, ns, color):
        entries = []
        world_idx = self._plant.world_body().index()
        for bidx in self._plant.GetBodyIndices(model_instance):
            if bidx == world_idx:
                continue
            body = self._plant.get_body(bidx)
            name = body.name()
            if name not in cfg_dict:
                continue
            cfgs = cfg_dict[name]
            for i, cfg in enumerate(cfgs):
                sx, sy, sz = cfg["size"]
                rx, ry, rz = cfg.get("rpy", (0,0,0))
                px, py, pz = cfg.get("p",   (0,0,0))
                X_LC = RigidTransform(RollPitchYaw(rx, ry, rz).ToRotationMatrix(), [px, py, pz])
                path = f"{ns}/{name}/{i}"
                self._meshcat.SetObject(path, DrakeBox(sx, sy, sz), color)
                entries.append({"path": path, "bidx": bidx, "X_LC": X_LC})
        return entries

    # 👇 Make `event` optional so both signatures work.
    def _do_publish(self, context, event=None):
        # Pull full plant state from input and update temp context
        x = self._x_port.Eval(context)
        try:
            self._plant.SetPositionsAndVelocities(self._tmp_ctx, x)
        except Exception:
            nq = self._plant.num_positions()
            self._plant.SetPositions(self._tmp_ctx, x[:nq])
            self._plant.SetVelocities(self._tmp_ctx, x[nq:])
        for e in self._entries:
            body = self._plant.get_body(e["bidx"])
            X_WL = self._plant.EvalBodyPoseInWorld(self._tmp_ctx, body)
            X_WC = X_WL @ e["X_LC"]
            self._meshcat.SetTransform(e["path"], X_WC)



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

    # --- NEW: add publisher that visualizes the FCL proxy boxes (robot + movable box)
    boxes_pub = builder.AddSystem(MeshcatBoxesPublisher(plant, meshcat, mi_robot, mi_movable))
    # Feed plant state to publisher for kinematics
    builder.Connect(plant.get_state_output_port(), boxes_pub.GetInputPort("x"))
    boxes_pub.set_name("meshcat_boxes_publisher")

    # Add FCL link distance system (before building diagram) and connect full plant state
    fcl_sys = builder.AddSystem(FCLLinkDistanceSystem(plant, mi_robot, mi_movable))
    builder.Connect(plant.get_state_output_port(), fcl_sys.GetInputPort("x"))
    # Log the 4 distances over time
    fcl_logger = LogVectorOutput(fcl_sys.get_output_port(0), builder)
    fcl_logger.set_name("fcl_distances")

    diagram = builder.Build()
    sim = Simulator(diagram)
    diagram_ctx = sim.get_mutable_context()
    plant_ctx = diagram.GetMutableSubsystemContext(plant, diagram_ctx)

    # Evaluate once before running the sim
    fcl_ctx = diagram.GetMutableSubsystemContext(fcl_sys, diagram_ctx)
    y = fcl_sys.get_output_port(0).Eval(fcl_ctx)
    print("\nInitial signed distances (robot links → movable box) [m]:")
    print(f"link7={y[0]:.6f}, hand={y[1]:.6f}, leftfinger={y[2]:.6f}, rightfinger={y[3]:.6f}")

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

    # -------- FCL: robot ↔ movable_box only (one-shot query before sim) --------
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

    # -------- viz + sim --------
    sim.set_target_realtime_rate(1.0)
    sim.set_publish_every_time_step(True)  # ensures MeshcatBoxesPublisher publishes every step
    sim.Initialize()
    print("\nMeshcat is publishing frames now...")
    sim.AdvanceTo(5.0)

    # Query again after sim
    fcl_ctx = diagram.GetMutableSubsystemContext(fcl_sys, diagram_ctx)
    y_final = fcl_sys.get_output_port(0).Eval(fcl_ctx)
    print("\nFinal signed distances (robot links → movable box) [m]:")
    print(f"link7={y_final[0]:.6f}, hand={y_final[1]:.6f}, leftfinger={y_final[2]:.6f}, rightfinger={y_final[3]:.6f}")
    # Retrieve logged FCL distances
    log_fcl = fcl_logger.FindLog(diagram_ctx)
    t_fcl = log_fcl.sample_times()
    data_fcl = log_fcl.data().transpose()  # shape: N x 4

    # Plot
    plt.figure(figsize=(10, 6))
    labels = ["link7", "hand", "leftfinger", "rightfinger"]
    for i in range(4):
        plt.plot(t_fcl, data_fcl[:, i], label=labels[i])
    plt.axhline(0.0, color='r', linestyle='--', alpha=0.6)
    plt.title('FCL signed distances: robot links → movable box')
    plt.xlabel('Time [s]')
    plt.ylabel('Signed distance [m] (negative ⇒ penetration)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Done.")
