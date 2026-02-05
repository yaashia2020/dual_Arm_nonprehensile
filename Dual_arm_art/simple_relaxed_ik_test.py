#!/usr/bin/env python3
"""
Simple Relaxed IK Test Script
This script demonstrates basic Relaxed IK functionality without Drake integration.
"""

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

wrapper_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../submodules/relaxed_ik_core/wrappers"))
sys.path.insert(0, wrapper_dir)

from python_wrapper import RelaxedIKRust
# Your ctypes wrapper class

from ik_fk_compare import DrakeFK
from plotting import BodyPoseExtractor, add_mux_logger, LogSignal
from CERG_Setup import panda_spec

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ConstantVectorSource,
    DiagramBuilder,
    LeafSystem,
    Parser,
    Simulator,
)

# Use the same PD+G controller implementation as the main ERG test codebase uses (shared module)
from controllers.pd_gravity import PD_gravity


class ToyAttractionQv(LeafSystem):
    """
    Minimal "toy CERG" module:
      - Keeps a discrete internal state q_v (size = nu)
      - Input: q_r (IK joint solution)
      - Attraction field (like trajectoryERG.rho_att):
            att_field = (q_r - q_v) / max(||q_r - q_v||, eta)
      - Update (DSM=1, attraction only):
            q_v <- q_v + att_field * dt
      - Output: q_v (to be used as PD controller target)
    """

    def __init__(self, nu: int, dt: float, eta: float = 1e-3):
        super().__init__()
        self.nu = int(nu)
        self.dt = float(dt)
        self.eta = float(eta)

        self._q_r_port = self.DeclareVectorInputPort(name="q_r", size=self.nu)
        state_index = self.DeclareDiscreteState(self.nu)
        self.DeclareStateOutputPort("q_v", state_index)

        self.DeclarePeriodicDiscreteUpdateEvent(period_sec=self.dt, offset_sec=0.0, update=self._update)

    def _update(self, context, discrete_state):
        q_r = np.asarray(self._q_r_port.Eval(context), dtype=float).reshape(-1)
        q_v = np.asarray(context.get_discrete_state_vector().CopyToVector(), dtype=float).reshape(-1)

        diff = q_r - q_v
        denom = max(float(np.linalg.norm(diff)), self.eta)
        att_field = diff / denom
        q_v_new = q_v + att_field * self.dt

        discrete_state.get_mutable_vector().SetFromVector(q_v_new.tolist())

def _load_relaxedik_config(config_path: str) -> tuple[str, str]:
    """
    Returns (urdf_abs_path, ee_link_name) as used by RelaxedIK for this settings YAML.
    RelaxedIK resolves URDFs from <repo_root>/configs/urdfs/<urdf>.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    urdf_name = cfg.get("urdf", "panda.urdf")
    ee_links = cfg.get("ee_links") or ["panda_hand"]
    ee_link = ee_links[0]
    starting_config = cfg.get("starting_config", None)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    urdf_abs = os.path.join(repo_root, "configs", "urdfs", urdf_name)
    return urdf_abs, ee_link, starting_config

def _make_fk(urdf_path: str, body_name: str) -> DrakeFK:
    """
    Create FK helper; if body isn't present in the URDF, fall back to panda_hand.
    """
    try:
        return DrakeFK(urdf_path=urdf_path, body_name=body_name)
    except Exception as e:
        print(f"[WARN] FK body '{body_name}' not found in URDF ({urdf_path}): {e}")
        print("[WARN] Falling back to FK on 'panda_hand'")
        return DrakeFK(urdf_path=urdf_path, body_name="panda_hand")


def solve_position_iterative(rik, target_xyz, target_quat_xyzw, tolerances, iters: int = 5):
    """
    RelaxedIK sometimes returns the previous solution if the optimizer produced NaNs.
    Running a few iterations (warm-starting) and using non-zero tolerances makes this visible.
    """
    q = None
    for k in range(iters):
        q = rik.solve_position(target_xyz, target_quat_xyzw, tolerances)
    return q


def test_relaxed_ik_basic():
    """Test basic Relaxed IK functionality"""
    
    print("Initializing Relaxed IK...")
    
    # Use our custom settings file with custom initial pose
    # (Choose one of these existing files: panda_settings.yaml / panda_rubber_settings.yaml)
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_rubber_settings.yaml"))
    print(f"Using custom config path: {config_path}")
    
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"RelaxedIK settings YAML not found: {config_path}")
        urdf_for_fk, ee_link, starting_config = _load_relaxedik_config(config_path)
        print(f"RelaxedIK-config EE link: {ee_link}")
        print(f"RelaxedIK-config URDF for FK: {urdf_for_fk}")
        rik = RelaxedIKRust(setting_file_path=config_path)
        print("Relaxed IK initialized successfully with custom settings!")

        # Single target pose components
        position = [0.5, 0.0, 0.5]
        orientation = [0.0, 0.0, 0.0, 1.0]  # quaternion xyzw
        # IMPORTANT: don't use all-zeros tolerances; it can cause numerical issues inside RelaxedIK.
        tolerance = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]
        
        joint_angles = solve_position_iterative(rik, position, orientation, tolerance, iters=5)
        print(f"Solved joint angles (after 5 iters): {joint_angles}")

        # FK debug: compare target position vs FK(ee_link, q_from_RIK)
        fk = _make_fk(urdf_for_fk, ee_link)
        res = fk.compare(q=joint_angles, target_xyz=position)
        print(f"FK({ee_link}) xyz:  {res.fk_xyz.tolist()}")
        print(f"Target xyz:        {res.target_xyz.tolist()}")
        print(f"FK error xyz:      {res.error_xyz.tolist()} (norm={res.error_norm:.6f})")
        if starting_config is not None:
            q0 = np.asarray(starting_config, dtype=float).reshape(-1)
            q_sol = np.asarray(joint_angles, dtype=float).reshape(-1)
            n = min(q0.size, q_sol.size)
            same_as_start = float(np.max(np.abs(q0[:n] - q_sol[:n]))) < 1e-3
            print(f"Returned solution ~ starting_config? {same_as_start} (max|dq|={float(np.max(np.abs(q0[:n]-q_sol[:n]))):.6g})")

        # Multiple targets example
        multiple_targets = [
            ([0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]),
            ([0.6, 0.1, 0.4], [0.0, 0.0, 0.0, 1.0]),
            ([0.4, -0.1, 0.6], [0.0, 0.0, 0.0, 1.0])
        ]

        print("Solving for multiple targets...")
        targets_xyz = []
        fks_xyz = []
        err_norms = []
        for i, (pos, ori) in enumerate(multiple_targets):
            joint_angles = solve_position_iterative(rik, pos, ori, tolerance, iters=5)
            res = fk.compare(q=joint_angles, target_xyz=pos)
            print(f"Target {i+1}: {pos} -> Joint angles: {joint_angles}")
            print(f"           FK xyz: {res.fk_xyz.tolist()} | err_norm={res.error_norm:.6f}")
            targets_xyz.append(res.target_xyz)
            fks_xyz.append(res.fk_xyz)
            err_norms.append(res.error_norm)

        # Plot: target vs FK (xyz) and error norm
        if len(targets_xyz) > 0:
            targets_xyz = np.vstack(targets_xyz)  # (N,3)
            fks_xyz = np.vstack(fks_xyz)          # (N,3)
            err_norms = np.asarray(err_norms, dtype=float)
            idx = np.arange(1, targets_xyz.shape[0] + 1)

            fig1, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            labels = ["x", "y", "z"]
            for k in range(3):
                axs[k].plot(idx, targets_xyz[:, k], "o--", label=f"target_{labels[k]}")
                axs[k].plot(idx, fks_xyz[:, k], "s-", label=f"fk_{labels[k]}")
                axs[k].set_ylabel(f"{labels[k]} [m]")
                axs[k].grid(True)
                axs[k].legend(loc="best")
            axs[-1].set_xlabel("Target index")
            fig1.suptitle(f"RelaxedIK: target vs Drake FK({ee_link})")
            fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

            plt.show()
    
    except Exception as e:
        print(f"Error initializing Relaxed IK: {e}")
        print("Check that the config YAML exists and wrapper is properly imported")


def test_relaxed_ik_with_drake_pd():
    """
    Build a Drake world (Panda + fixed box), compute IK target joints with RelaxedIK,
    drive the robot with PD+gravity, then plot:
      - joint positions vs IK joint targets (7 subplots)
      - panda_hand world position (sim) vs IK-FK(panda_hand from q_r) vs target xyz
    """
    print("\n=== RelaxedIK + Drake PD world test ===")

    # Settings file for RelaxedIK
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_rubber_settings.yaml"))
    print(f"Using RelaxedIK config: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"RelaxedIK settings YAML not found: {config_path}")

    urdf_for_fk, ee_link, starting_config = _load_relaxedik_config(config_path)
    print(f"RelaxedIK-config EE link: {ee_link}")
    print(f"RelaxedIK-config URDF for Drake plant: {urdf_for_fk}")

    # IK target in world (for the EE link in the RelaxedIK config)
    target_xyz = np.array([0.6, 0.0, 0.2], dtype=float)
    target_quat_xyzw = [0.0, 0.0, 0.0, 1.0]
    tol = [0.0] * 6

    rik = RelaxedIKRust(setting_file_path=config_path)
    # Use non-zero tolerances + a few iterations so we don't silently fall back to start pose.
    tol = [0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01]
    q_r_full = np.asarray(
        solve_position_iterative(rik, target_xyz.tolist(), target_quat_xyzw, tol, iters=5),
        dtype=float,
    ).reshape(-1)
    print(f"RelaxedIK returned {q_r_full.size} joints: {q_r_full.tolist()}")
    if starting_config is not None:
        q0 = np.asarray(starting_config, dtype=float).reshape(-1)
        n = min(q0.size, q_r_full.size)
        print(f"max|q_r - starting_config| = {float(np.max(np.abs(q_r_full[:n] - q0[:n]))):.6g}")

    # Drake world (Panda + fixed box) using the SAME URDF that RelaxedIK used.
    robot_spec = panda_spec()
    panda_urdf = os.path.abspath(urdf_for_fk)
    fixed_box_sdf = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/boxes/fixed_box.sdf"))

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    panda_instances = Parser(plant).AddModelsFromUrl("file://" + panda_urdf)
    panda_id = panda_instances[0] if isinstance(panda_instances, list) else panda_instances
    Parser(plant).AddModelsFromUrl("file://" + fixed_box_sdf)
    plant.Finalize()

    nq = plant.num_positions(panda_id)
    nv = plant.num_velocities(panda_id)
    nu = plant.num_actuators(panda_id)

    # Desired joint targets: use the first nu values of RelaxedIK output (Panda arm actuators)
    q_des = q_r_full[:nu] if q_r_full.size >= nu else np.pad(q_r_full, (0, nu - q_r_full.size))
    q_des_source = builder.AddNamedSystem("q_des_source", ConstantVectorSource(q_des.tolist()))

    # Toy CERG-like q_v module (DSM=1, attraction only) to smoothly apply IK target
    qv_sys = builder.AddNamedSystem("toy_qv", ToyAttractionQv(nu=nu, dt=0.001, eta=1e-3))
    builder.Connect(q_des_source.get_output_port(), qv_sys.GetInputPort("q_r"))

    # PD + gravity controller
    ctrl = builder.AddNamedSystem(
        "PD+G controller",
        PD_gravity(plant, panda_id, kp=robot_spec.kp, kd=robot_spec.kd),
    )
    builder.Connect(qv_sys.GetOutputPort("q_v"), ctrl.GetInputPort("Desired_state"))
    builder.Connect(plant.get_state_output_port(panda_id), ctrl.GetInputPort("Current_state"))
    builder.Connect(ctrl.GetOutputPort("tau_u"), plant.get_actuation_input_port(panda_id))

    # Body pose extractor for panda_hand (actual sim)
    hand_extractor = builder.AddNamedSystem(
        "PandaHandPoseExtractor",
        BodyPoseExtractor(plant, panda_id, body_name=ee_link, robot_spec=robot_spec),
    )
    builder.Connect(plant.get_state_output_port(panda_id), hand_extractor.GetInputPort("joint_positions"))

    # Logger
    logger_signals = [
        LogSignal("state", plant.get_state_output_port(panda_id)),
        LogSignal("q_r", q_des_source.get_output_port()),
        LogSignal("q_v", qv_sys.GetOutputPort("q_v")),
        LogSignal("hand_pos", hand_extractor.GetOutputPort("body_world_positions")),
    ]
    mux_logger = add_mux_logger(builder, logger_signals, sample_period=0.001, name="mux_logger")

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()

    # Initialize robot at robot_spec.q0 (pad to nq)
    q0 = np.zeros(nq)
    if robot_spec.q0 is not None:
        q0[: min(len(robot_spec.q0), nq)] = np.asarray(robot_spec.q0)[: min(len(robot_spec.q0), nq)]
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
    plant.SetPositions(plant_context, panda_id, q0)

    # Initialize q_v to the current robot joint positions (actuated subset)
    qv_context = qv_sys.GetMyMutableContextFromRoot(diagram_context)
    qv_context.get_mutable_discrete_state_vector().SetFromVector(q0[:nu].tolist())

    simulator = Simulator(diagram, diagram_context)
    simulator.Initialize()
    simulator.AdvanceTo(10.0)

    # Extract logs
    state_raw, t_state = mux_logger.get(diagram_context, "state")
    q_r_raw, t_qr = mux_logger.get(diagram_context, "q_r")
    q_v_raw, t_qv = mux_logger.get(diagram_context, "q_v")
    hand_raw, t_hand = mux_logger.get(diagram_context, "hand_pos")

    state = state_raw.T  # (N, nq+nv)
    t = t_state
    q = state[:, :nq]
    q_r = q_r_raw.T  # (N, nu)
    q_v = q_v_raw.T  # (N, nu)
    hand_pos = hand_raw.T  # (N, 3)

    # IK-FK: compute panda_hand xyz from q_v (applied target) using DrakeFK (same URDF as simulation)
    fk = _make_fk(panda_urdf, ee_link)
    ik_fk_xyz = np.vstack([fk.fk_xyz(q_v[k, :]) for k in range(q_v.shape[0])])

    # Plot 1: joint positions (first 7 actuated joints) vs IK q_r and applied q_v
    nplot = nu
    fig1, axs1 = plt.subplots(nplot, 1, figsize=(12, 2.2 * nplot), sharex=True)
    if nplot == 1:
        axs1 = [axs1]
    fig1.suptitle("Joint positions vs RelaxedIK target (q_r) and toy q_v (applied)")
    for j in range(nplot):
        axs1[j].plot(t, q[:, j], label=f"q[{j}]", linewidth=2)
        axs1[j].plot(t_qr[: len(q_r)], q_r[:, j], label=f"q_r[{j}] (IK)", linestyle="--")
        axs1[j].plot(t_qv[: len(q_v)], q_v[:, j], label=f"q_v[{j}] (applied)", linestyle=":")
        axs1[j].grid(True)
        axs1[j].legend(loc="upper right", fontsize=8)
    axs1[-1].set_xlabel("time [s]")
    fig1.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Plot 2: panda_hand world position (actual sim) vs FK(q_v) vs target
    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    labels = ["x", "y", "z"]
    for i in range(3):
        axs2[i].plot(t_hand, hand_pos[:, i], label=f"{ee_link}_{labels[i]} (sim)")
        axs2[i].plot(t_qv[: len(ik_fk_xyz)], ik_fk_xyz[:, i], label=f"{ee_link}_{labels[i]} (FK(q_v))", linestyle=":")
        axs2[i].axhline(y=float(target_xyz[i]), color="black", linestyle="--", label=f"target_{labels[i]}")
        axs2[i].grid(True)
        axs2[i].legend(loc="upper right", fontsize=8)
    axs2[-1].set_xlabel("time [s]")
    fig2.suptitle(f"{ee_link} world position: sim vs FK(q_v) vs target")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.98])

    plt.show()


def test_dual_arm_setup():
    """Test dual-arm Relaxed IK setup"""
    
    print("\n=== Testing Dual-Arm Setup ===")
    
    # Use our custom settings file with custom initial pose
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_settings.yaml"))
    print(f"Using custom config path for dual-arm: {config_path}")
    
    try:
        left_arm_rik = RelaxedIKRust(setting_file_path=config_path)
        right_arm_rik = RelaxedIKRust(setting_file_path=config_path)
        print("Dual-arm Relaxed IK initialized with custom settings!")

        left_target_pos = [0.3, 0.2, 0.5]
        left_target_ori = [0.0, 0.0, 0.0, 1.0]
        right_target_pos = [0.3, -0.2, 0.5]
        right_target_ori = [0.0, 0.0, 0.0, 1.0]
        tolerance = [0.0]*6

        left_joints = left_arm_rik.solve_position(left_target_pos, left_target_ori, tolerance)
        right_joints = right_arm_rik.solve_position(right_target_pos, right_target_ori, tolerance)
        print(f"Left arm joints: {left_joints}")
        print(f"Right arm joints: {right_joints}")

        print("\n--- Alternative: Single solver approach ---")
        rik = RelaxedIKRust(setting_file_path=config_path)

        left_joints_alt = rik.solve_position(left_target_pos, left_target_ori, tolerance)
        right_joints_alt = rik.solve_position(right_target_pos, right_target_ori, tolerance)
        print(f"Left arm joints (single solver): {left_joints_alt}")
        print(f"Right arm joints (single solver): {right_joints_alt}")
        
    except Exception as e:
        print(f"Error with dual-arm setup: {e}")
        print("Check that the config YAML exists and wrapper is properly imported")


def create_sample_config():
    """Create a sample configuration file for the Panda FR3 with Relaxed IK"""
    
    sample_config = """# Custom Relaxed IK config for Panda FR3
urdf: panda.urdf
link_radius: 0.05 
base_links:
  - world
ee_links:
  - panda_hand
starting_config: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # Custom initial pose
obstacles:
"""
    
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_settings.yaml"))
    with open(config_file, 'w') as f:
        f.write(sample_config)
    
    print(f"Sample configuration saved to: {config_file}")
    print("Edit this file if you want to adjust optimization or collision settings.")


if __name__ == "__main__":
    print("=== Relaxed IK Simple Test ===")
    
    # Create sample configuration
    # create_sample_config()
    
    # Test basic functionality
    test_relaxed_ik_basic()

    # Drake world + PD controller tracking RelaxedIK targets
    test_relaxed_ik_with_drake_pd()
    
    # Test dual-arm setup
    test_dual_arm_setup()
    
    print("\n=== Test Complete ===")
