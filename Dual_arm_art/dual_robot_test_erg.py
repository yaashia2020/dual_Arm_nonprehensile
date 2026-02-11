#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual Panda FR3 Robots with ERG System - Based on test_erg.py with dual robots

This script creates a dual-arm manipulation system with:
- Two Panda FR3 robots mounted on a fixed box
- Movable box for manipulation
- Explicit Reference Governor (ERG) system for both robots
- RelaxedIK for box tracking
- Contact force monitoring
- Comprehensive logging and visualization

Based on test_erg.py architecture with dual robot support from dual_panda.py
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import csv  
from pydrake.all import *
# import pydot
from IPython.display import SVG, display
from trajectoryERG import ExplicitReferenceGovernor
import os
from pydrake.visualization import AddDefaultVisualization

from scipy.spatial.transform import Rotation as R



# Add Relaxed IK wrapper import
import sys
wrapper_dir = "/home/yaashia/dual_arm_nonprehensile/submodules/relaxed_ik_core/wrappers"
sys.path.insert(0, wrapper_dir)
from python_wrapper import RelaxedIKRust




# Import FCL system from fcl_test.py
from fcl_test import FCLLinkDistanceSystem

# Remove the import since we'll define PD_gravity locally
# from controllers.PID import PD_gravity

class PD_gravity(LeafSystem):
    """
    PD controller with gravity compensation for dual robot setup.
    Can be instantiated multiple times for different robots.
    """
    def __init__(self, plant, Kp, Kd, robot_id=None):
        super().__init__()
        
        self.plant = plant
        self.robot_id = robot_id
        self.Kp_ = np.asarray(Kp, dtype=float).reshape(-1)
        self.Kd_ = np.asarray(Kd, dtype=float).reshape(-1)

        if self.robot_id is None:
            raise ValueError("PD_gravity requires robot_id (ModelInstanceIndex).")

        # Per-robot dimensions
        self.nq = int(self.plant.num_positions(self.robot_id))
        self.nv = int(self.plant.num_velocities(self.robot_id))
        self.nu = int(self.plant.get_actuation_input_port(self.robot_id).size())

        if self.nu != self.nq:
            raise ValueError(f"Expected nu == nq for PD, got nu={self.nu}, nq={self.nq}")
        if self.Kp_.shape[0] != self.nq or self.Kd_.shape[0] != self.nq:
            raise ValueError(f"PD gains must have shape ({self.nq},); got Kp={self.Kp_.shape}, Kd={self.Kd_.shape}")
        
        # Input ports
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=self.nq)
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=self.nq + self.nv)
        
        # Output port for control torques
        self.DeclareVectorOutputPort("tau_u", size=self.nu, calc=self._calc_output)
        
        # Declare discrete state for storing computed torques
        self.DeclareDiscreteState(self.nu)
        
        # Periodic update event
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/1000,  # 1kHz update rate
            offset_sec=0.0,
            update=self._update_discrete_state
        )
    
    def _calc_output(self, context, output):
        """Calculate and output the control torques."""
        discrete_state = context.get_discrete_state_vector()
        output.SetFromVector(discrete_state.get_value())
    
    def _update_discrete_state(self, context, discrete_state):
        """Update the discrete state with new control torques."""
        # Get input values
        q_d = self._desired_state_port.Eval(context)
        q_full = self._current_state_port.Eval(context)
        
        # Extract position and velocity
        q = q_full[: self.nq]
        q_dot = q_full[self.nq : self.nq + self.nv]
        
        # Create plant context for gravity calculation
        plant_context = self.plant.CreateDefaultContext()
        
        # Set robot state in plant context
        self.plant.SetPositions(plant_context, self.robot_id, q)
        
        # Calculate gravity generalized forces for the *whole plant*.
        # NOTE: CalcGravityGeneralizedForces returns a vector indexed like generalized velocities (size = plant.num_velocities()).
        tau_g_full = -self.plant.CalcGravityGeneralizedForces(plant_context)
        tau_g_full = np.asarray(tau_g_full).reshape(-1)

        # Extract this model instance's slice from any nv-indexed array.
        # Drake exposes this slicing helper as GetVelocitiesFromArray(model_instance, array),
        # so we reuse it here because generalized forces share the same indexing as v.
        tau_g_model = np.asarray(self.plant.GetVelocitiesFromArray(self.robot_id, tau_g_full), dtype=float).reshape(-1)
        # PD is defined over positions; in most robot arms nv == nq. Keep a defensive trim if not.
        gravity = tau_g_model[: self.nq]
        
        # Calculate PD control torques
        tau = self.Kp_ * (q_d - q) - self.Kd_ * q_dot
        
        # Add gravity compensation
        tau = tau + gravity
        
        # Update discrete state
        discrete_state.get_mutable_vector().SetFromVector(tau)

def create_dual_robot_system_model(plant, scene_graph):
    """
    Add the dual Panda arm models to the plant and configure contact properties.
    
    Args:
        plant: The MultibodyPlant object to which the dual Panda arm models will be added.
        scene_graph: The SceneGraph object for visualization.
    
    Returns:
        Tuple containing the updated plant, scene_graph, and plant_context.
    """
    # Load Panda robot models (using different URDFs for dual robots)
    panda_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_drake.urdf"))
    panda_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_drake2.urdf"))

    
    # Add both Panda robots
    panda1 = Parser(plant).AddModelsFromUrl("file://" + panda_path_1)  # loads 'panda'
    panda2 = Parser(plant).AddModelsFromUrl("file://" + panda_path_2)  # loads 'panda_1'
    
    # Add the fixed box to the scene (acting as static ground)
    fixed_box_sdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/boxes/fixed_box.sdf"))
    fixed_box_sdf_url = "file://" + fixed_box_sdf_path
    fixed_box = Parser(plant).AddModelsFromUrl(fixed_box_sdf_url)
    
    # Get the fixed box model instance ID
    fixed_box_id = plant.GetModelInstanceByName("fixed_box")
    
    # Add the movable box to the scene (on top of the fixed box)
    movable_box_sdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/boxes/movable_box.sdf"))
    movable_box_sdf_url = "file://" + movable_box_sdf_path
    movable_box = Parser(plant).AddModelsFromUrl(movable_box_sdf_url)
    
    # Get the movable box model instance ID
    movable_box_id = plant.GetModelInstanceByName("movable_box")
    
    # Set the contact properties for interaction between the movable box and the ground
    plant.set_contact_surface_representation(mesh_type)  # Triangle or Polygon
    plant.set_contact_model(contact_model)  # Hydroelastic, Point, or HydroelasticWithFallback
    plant.set_discrete_contact_approximation(discrete_solver)
    
    
    plant.Finalize()
    
    # Position the movable box above the fixed box after plant is finalized
    # The fixed box is at Z=0.55 with height 0.1, so place movable box at Z=0.65
    plant_context = plant.CreateDefaultContext()
    initial_box_position = RigidTransform(p=[0.85, 0.0, 0.2])  # Positioned just above the fixed box
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("box_link", movable_box_id), initial_box_position)
    
    # Set default positions for both robots
    panda1_id = plant.GetModelInstanceByName("panda")
    panda2_id = plant.GetModelInstanceByName("panda_1")
    
    # Print the number of joints for each robot
    num_positions_1 = plant.num_positions(panda1_id)
    num_positions_2 = plant.num_positions(panda2_id)
    print(f"Robot 1 ('panda') has {num_positions_1} joints")
    print(f"Robot 2 ('panda_1') has {num_positions_2} joints")
    
    # Initial positions for both robots (slightly different to avoid collision)
    #init_pos1 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]
    #init_pos2 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]


    #init_pos1 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    #init_pos2 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

    init_pos1 = np.array([np.pi/2, -np.pi/4, 0, -3*np.pi/4,  np.pi/2, np.pi/2, -np.pi/4 ])
    init_pos2 = np.array([np.pi/2, -np.pi/4, 0, -3*np.pi/4,  -np.pi/2, np.pi/2, -np.pi/4 ])


    
    plant.SetDefaultPositions(panda1_id, init_pos1)
    plant.SetDefaultPositions(panda2_id, init_pos2)
    
    return plant, scene_graph, plant_context

####################################
# Configurations parameters
####################################
contact_model = ContactModel.kHydroelasticWithFallback  # Options: Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kTriangle  # Options: Triangle or Polygon
discrete_solver = DiscreteContactApproximation.kSap # Options:kTamsi, kSap, kLagged, kSimilar
realtime_factor = 1  # Real-time factor for simulation speed
time_step = 0.0005

# Visualization can fail in headless / restricted network environments.
# Use MESHCAT_VIS=0 to disable it without editing the file.
meshcat_visualisation = os.environ.get("MESHCAT_VIS", "1") not in ("0", "false", "False")
simulate = True

# Create system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
plant, scene_graph, plant_context = create_dual_robot_system_model(plant, scene_graph)

# Get the panda model instances for specific connections
panda1_id = plant.GetModelInstanceByName("panda")
panda2_id = plant.GetModelInstanceByName("panda_1")
num_positions_1 = plant.num_positions(panda1_id)
num_velocities_1 = plant.num_velocities(panda1_id)
num_positions_2 = plant.num_positions(panda2_id)
num_velocities_2 = plant.num_velocities(panda2_id)

# Get the movable box model instance ID
movable_box_id = plant.GetModelInstanceByName("movable_box")

print(f"Robot 1: {num_positions_1} positions, {num_velocities_1} velocities")
print(f"Robot 2: {num_positions_2} positions, {num_velocities_2} velocities")
print(f"Movable box ID: {movable_box_id}")

######################################################################################################
#              ##################ERG System################
######################################################################################################
class ERG(LeafSystem):
    def __init__(self, plant, robot_id, erg_name="erg"):
        super().__init__()  # Don't forget to initialize the base class.
        self.plant = plant
        self.robot_id = robot_id
        self.erg_name = erg_name
        self.nq = int(self.plant.num_positions(self.robot_id))
        self.nv = int(self.plant.num_velocities(self.robot_id))
        self.nu = int(self.plant.get_actuation_input_port(self.robot_id).size())
        if self.nu != self.nq:
            raise ValueError(f"[{erg_name}] expected nu == nq, got nu={self.nu}, nq={self.nq}")

        self._state_port = self.DeclareVectorInputPort(name="state", size=self.nq + self.nv)
        self._tau_port = self.DeclareVectorInputPort(name="tau", size=self.nu)
        self._qr_port = self.DeclareVectorInputPort(name="q_r", size=self.nq)
        self._box_state_port = self.DeclareVectorInputPort(name="box_state", size=13)

        # Discrete state stores q_v (the filtered reference). Default zeros, but we will
        # set it from the connected plant state at Simulator.Initialize().
        state_index = self.DeclareDiscreteState(self.nq)
        self.DeclareStateOutputPort("q_v_filtered", state_index)  # One output: y=x.
        
        # Add output port for calculated energy
        self.DeclareVectorOutputPort("calculated_energy", size=1, calc=self.output_energy)
        self.DeclareVectorOutputPort("dsm", size=1, calc=self.output_dsm)
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.refrence) # Call the Update method defined below.

        # At initialization, sync q_v to the robot's actual initial joint positions so the controller
        # does not see q_v_filtered = 0 at t=0.
        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_qv)
        self.erg = ExplicitReferenceGovernor(
            robust_delta_tau_=0.1, kappa_tau_=1.0,
            robust_delta_q_=0.1, kappa_q_=15.0, robust_delta_dq_=0.1, kappa_dq_=7.0,
            robust_delta_dp_EE_=0.01, kappa_dp_EE_=7.0, kappa_terminal_energy_=7.5, FD_=1.0,
            name=erg_name)

        # Initialize a flag to check if it's the first update
        self.first_update = True
        self.q_v_ = np.zeros(self.nq)
        # Debug option: breaking in a LeafSystem update will halt the whole simulation.
        self.break_on_dsm_zero = False
    
    def _initialize_qv(self, context, discrete_state):
        """Initialize q_v from the measured robot joint positions at t=0."""
        state = self._state_port.Eval(context)
        q = np.asarray(state[: self.nq], dtype=float)
        self.q_v_ = q.copy()
        # Write into discrete state so output port is correct immediately.
        discrete_state.get_mutable_vector().SetFromVector(self.q_v_)
        # No need to re-init again in the first periodic callback.
        self.first_update = False

    def refrence(self, context, discrete_state):
        # Evaluate the input ports
        state = self._state_port.Eval(context)
        q = state[: self.nq]
        dq = state[self.nq : self.nq + self.nv]
        tau = self._tau_port.Eval(context)
        q_r = self._qr_port.Eval(context)
        box_state = self._box_state_port.Eval(context)
        # Extract and post-process box position (subtract 0.11 from x)
        box_position = np.array(box_state[4:7], dtype=float)
        box_position[0] -= 0.11

        self.q_v = context.get_discrete_state_vector().CopyToVector()              
        # Initialize q_v_ only at the first callback
        if self.first_update:
            #self.q_v_ = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # Use all 9 joints
            #self.q_v_ = [np.pi / 2, -np.pi / 4, 0, -3 * np.pi / 4, np.pi / 2, np.pi / 2, -np.pi / 4]
            q = state[: self.nq]
            self.q_v_ = q.copy()  # sync ERG’s internal reference to reality
            self.first_update = False

            #self.first_update = False
        
        # Update box position in ERG
        self.q_v_ = self.erg.get_qv(q, dq, tau, q_r, self.q_v_, box_position)
        
        # Calculate energy from trajectory predictions
        self.calculated_energy = self.erg.get_energy(q, dq, tau, q_r, self.q_v_, box_position)
        self.current_dsm = self.erg.get_last_dsm()
        if self.break_on_dsm_zero and np.isclose(self.current_dsm, 0.0):
            print(f"[DSM breakpoint][{self.erg_name}] t={context.get_time():.4f}, dsm={self.current_dsm:.6f}")
            breakpoint()

        # Write into the output vector.
        discrete_state.get_mutable_vector().SetFromVector(self.q_v_)
        
    def output_energy(self, context, output):
        # Return the calculated energy from the ERG system
        if hasattr(self, 'calculated_energy'):
            output.SetAtIndex(0, self.calculated_energy)
        else:
            output.SetAtIndex(0, 0.0)

    def output_dsm(self, context, output):
        # Return latest DSM from trajectoryERG.
        if hasattr(self, 'current_dsm'):
            output.SetAtIndex(0, self.current_dsm)
        else:
            output.SetAtIndex(0, 0.0)

######################################################################################################
#              ##################Relaxed IK System for Box Tracking################
######################################################################################################
class DualRelaxedIKBoxTracker(LeafSystem):
    def __init__(self, plant, plant_context, panda1_id, panda2_id):
        super().__init__()

        # Store plant and context
        self.plant = plant
        self.plant_context = plant_context
        self.panda1_id = panda1_id
        self.panda2_id = panda2_id
        self.nq1 = int(self.plant.num_positions(self.panda1_id))
        self.nq2 = int(self.plant.num_positions(self.panda2_id))

        # Input port for box state
        self._box_state_port = self.DeclareVectorInputPort(name="box_state", size=13)

        # State for both robots (q1 + q2)
        state_index = self.DeclareDiscreteState(self.nq1 + self.nq2)
        self.DeclareStateOutputPort("ik_joint_targets", state_index)

        # Output port for the cartesian targets that IK is solving for:
        # [x1, y1, z1, x2, y2, z2]
        self.DeclareVectorOutputPort(
            "ee_target_positions", size=6, calc=self.CalcTargetPositions
        )

        # Periodic update for IK solving
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,
            offset_sec=0.0,
            update=self.solve_ik)

        # Initialize Relaxed IK
        try:
            config_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_settings.yaml"))
            config_path2 = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda2_settings.yaml"))

            self.rik1 = RelaxedIKRust(setting_file_path=config_path)
            self.rik2 = RelaxedIKRust(setting_file_path=config_path2)
            print("Relaxed IK initialized successfully for both robots!")
            self.ik_available = True

            # === Save initial EE orientations ===
            panda1_id = self.plant.GetModelInstanceByName("panda")
            panda2_id = self.plant.GetModelInstanceByName("panda_1")

            # Get default positions (set in create_dual_robot_system_model)
            q1 = self.plant.GetDefaultPositions(panda1_id)
            q2 = self.plant.GetDefaultPositions(panda2_id)

            # Build a context with those defaults
            ctx = self.plant.CreateDefaultContext()
            self.plant.SetPositions(ctx, panda1_id, q1)
            self.plant.SetPositions(ctx, panda2_id, q2)

            # Snapshot EE poses
            ee1_pose = self.plant.EvalBodyPoseInWorld(ctx, self.plant.GetBodyByName("rubber_pad", panda1_id))
            ee2_pose = self.plant.EvalBodyPoseInWorld(ctx, self.plant.GetBodyByName("rubber_pad", panda2_id))

            self.ee1_initial_quat = R.from_matrix(ee1_pose.rotation().matrix()).as_quat()
            self.ee2_initial_quat = R.from_matrix(ee2_pose.rotation().matrix()).as_quat()

            # ✅ Force robot1 to share robot2's orientation
            self.ee1_initial_quat = self.ee2_initial_quat.copy()

            print("Saved initial orientations (robot1 forced to match robot2)")

        except Exception as e:
            print(f"Failed to initialize Relaxed IK: {e}")
            self.ik_available = False
            self.ee1_initial_quat = [0, 0, 0, 1]
            self.ee2_initial_quat = [0, 0, 0, 1]

        # IK tolerances
        self.tolerance = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]

    def CalcTargetPositions(self, context, output):
        """Compute the same EE xyz targets used by solve_ik()."""
        box_state = self._box_state_port.Eval(context)
        x_b, y_b, z_b = box_state[4:7]
        t = context.get_time()

        box_size = np.array([0.22, 0.30, 0.20])  # from movable_box.sdf
        half_extents = box_size / 2.0
        pad_offset = 0.03

        z_target = z_b
        if t > 6.0:
            z_target = z_b + 0.4

        target_position1 = [x_b - half_extents[0] - pad_offset, y_b, z_target]
        target_position2 = [x_b + half_extents[0] + pad_offset, y_b, z_target]

        output.SetFromVector(
            np.array(
                [
                    target_position1[0],
                    target_position1[1],
                    target_position1[2],
                    target_position2[0],
                    target_position2[1],
                    target_position2[2],
                ],
                dtype=float,
            )
        )

    def solve_ik(self, context, discrete_state):
        """Solve IK for both robots to grab the box from opposite sides."""
        if not self.ik_available:
            return

        # Get current box state
        box_state = self._box_state_port.Eval(context)
        x_b, y_b, z_b = box_state[4:7]

        # Simulation time
        t = context.get_time()

        print("time", t)

        # --- Base box geometry ---
        box_size = np.array([0.22, 0.30, 0.20])  # from movable_box.sdf
        half_extents = box_size / 2.0
        pad_offset = 0.03  # clearance from box face

        # --- Dynamic Z target ---
        z_target = z_b
        if t > 6.0:
            z_target = z_b+0.4   # after 5s, lift EE target 3 cm

        # --- Define target positions in the box frame ---
        target_position1 = [x_b - half_extents[0] - pad_offset, y_b, z_target]
        target_position2 = [x_b + half_extents[0] + pad_offset, y_b, z_target]

        # --- Define orientations relative to box frame ---
        orientation1 = R.from_euler("y", 90, degrees=True).as_quat()  # palm toward +x
        orientation2 = R.from_euler("y", -90, degrees=True).as_quat()  # palm toward -x

        # Current combined state
        current_state = discrete_state.get_mutable_vector()

        print(f"[t={t:.2f}] Robot1 target:", target_position1, orientation1)
        print(f"[t={t:.2f}] Robot2 target:", target_position2, orientation2)

        try:
            q1_solution = self.rik1.solve_position(target_position1, orientation1, self.tolerance)
            q2_solution = self.rik2.solve_position(target_position2, orientation2, self.tolerance)

            q1_solution = np.asarray(q1_solution, dtype=float).reshape(-1)
            q2_solution = np.asarray(q2_solution, dtype=float).reshape(-1)
            if q1_solution.shape[0] != self.nq1 or q2_solution.shape[0] != self.nq2:
                raise ValueError(
                    f"IK returned shapes q1={q1_solution.shape}, q2={q2_solution.shape}, "
                    f"expected ({self.nq1},) and ({self.nq2},)"
                )
            solution = np.concatenate([q1_solution, q2_solution])
            current_state.SetFromVector(solution)

        except Exception as e:
            print(f"IK solving failed: {e}")

######################################################################################################
#              ##################Contact Force Converter################
######################################################################################################
class DualContactForceConverter(LeafSystem):
    def __init__(self, plant, scene_graph, panda1_id, panda2_id):
        super().__init__()
        
        self.plant = plant
        self.scene_graph = scene_graph
        self.panda1_id = panda1_id
        self.panda2_id = panda2_id
        self.DeclareAbstractInputPort("contact_results", AbstractValue.Make(ContactResults()))
        self.DeclareVectorOutputPort("contact_forces", size=6, calc=self.CalcOutput)  # 3 for each robot
        
        # Store the robot model instance IDs for contact detection
        # We'll use body names to identify which robot each contact involves
        
        # Contact categorization
        self.robot1_contacts = 0
        self.robot2_contacts = 0
        self.hand_contacts = 0
        self.finger_contacts = 0
    
    def CalcOutput(self, context, output):
        contact_results = self.GetInputPort("contact_results").Eval(context)
        
        # Initialize contact forces for both robots
        robot1_force = np.zeros(3)
        robot2_force = np.zeros(3)
        
        # Process contacts for both robots
        for i in range(contact_results.num_point_pair_contacts()):
            contact_info = contact_results.point_pair_contact_info(i)
            body_a = contact_info.bodyA_index()
            body_b = contact_info.bodyB_index()
            
            # Get body names
            body_a_name = self.plant.get_body(body_a).name()
            body_b_name = self.plant.get_body(body_b).name()
            
            # Check if contact involves robot 1 (panda)
            if "panda" in body_a_name.lower() and "panda_1" not in body_a_name.lower():
                force = contact_info.contact_force()
                robot1_force += np.array([force[0], force[1], force[2]])
                self.robot1_contacts += 1
            elif "panda" in body_b_name.lower() and "panda_1" not in body_b_name.lower():
                force = contact_info.contact_force()
                robot1_force += np.array([force[0], force[1], force[2]])
                self.robot1_contacts += 1
            
            # Check if contact involves robot 2 (panda_1)
            elif "panda_1" in body_a_name.lower():
                force = contact_info.contact_force()
                robot2_force += np.array([force[0], force[1], force[2]])
                self.robot2_contacts += 1
            elif "panda_1" in body_b_name.lower():
                force = contact_info.contact_force()
                robot2_force += np.array([force[0], force[1], force[2]])
                self.robot2_contacts += 1
        
        # Combine forces for output (6 values: [robot1_x, robot1_y, robot1_z, robot2_x, robot2_y, robot2_z])
        combined_forces = np.concatenate([robot1_force, robot2_force])
        output.SetFromVector(combined_forces)
        


######################################################################################################
#              ##################Panda Link7 Pose Extractor for Both Robots################
######################################################################################################
class DualPandaLink7PoseExtractor(LeafSystem):
    def __init__(self, plant, panda1_id, panda2_id):
        super().__init__()
        self.plant = plant
        self.panda1_id = panda1_id
        self.panda2_id = panda2_id
        self.nq1 = int(self.plant.num_positions(self.panda1_id))
        self.nv1 = int(self.plant.num_velocities(self.panda1_id))
        self.nq2 = int(self.plant.num_positions(self.panda2_id))
        self.nv2 = int(self.plant.num_velocities(self.panda2_id))
        
        # Input ports for both robots
        self.DeclareVectorInputPort("robot1_joint_positions", size=self.nq1 + self.nv1)
        self.DeclareVectorInputPort("robot2_joint_positions", size=self.nq2 + self.nv2)
        
        # Output ports for both robots (6 values: [robot1_x, robot1_y, robot1_z, robot2_x, robot2_y, robot2_z])
        self.DeclareVectorOutputPort("panda_link7_world_positions", size=6, calc=self.CalcOutput)
        self.DeclareVectorOutputPort("rubber_pad_world_positions", size=6, calc=self.CalcRubberPadOutput)
        
        self.temp_context = plant.CreateDefaultContext()
    
    def CalcOutput(self, context, output):
        # Get joint positions for both robots
        robot1_state = self.GetInputPort("robot1_joint_positions").Eval(context)
        robot2_state = self.GetInputPort("robot2_joint_positions").Eval(context)
        
        # Extract joint positions
        robot1_joints = robot1_state[: self.nq1]
        robot2_joints = robot2_state[: self.nq2]
        
        # Set positions for robot 1
        self.plant.SetPositions(self.temp_context, self.panda1_id, robot1_joints)
        panda1_link7_body = self.plant.GetBodyByName("panda_link7", self.panda1_id)  # Specify robot 1
        panda1_link7_pose = self.plant.EvalBodyPoseInWorld(self.temp_context, panda1_link7_body)
        robot1_translation = panda1_link7_pose.translation()
        
        # Set positions for robot 2
        self.plant.SetPositions(self.temp_context, self.panda2_id, robot2_joints)
        panda2_link7_body = self.plant.GetBodyByName("panda_link7", self.panda2_id)  # Specify robot 2
        panda2_link7_pose = self.plant.EvalBodyPoseInWorld(self.temp_context, panda2_link7_body)
        robot2_translation = panda2_link7_pose.translation()
        
        # Combine translations for output
        combined_translation = np.concatenate([robot1_translation, robot2_translation])
        output.SetFromVector([combined_translation[0], combined_translation[1], combined_translation[2],
                             combined_translation[3], combined_translation[4], combined_translation[5]])

    def CalcRubberPadOutput(self, context, output):
        # Get joint positions for both robots
        robot1_state = self.GetInputPort("robot1_joint_positions").Eval(context)
        robot2_state = self.GetInputPort("robot2_joint_positions").Eval(context)

        # Extract joint positions
        robot1_joints = robot1_state[: self.nq1]
        robot2_joints = robot2_state[: self.nq2]

        # Set positions for robot 1
        self.plant.SetPositions(self.temp_context, self.panda1_id, robot1_joints)
        panda1_pad_body = self.plant.GetBodyByName("rubber_pad", self.panda1_id)
        panda1_pad_pose = self.plant.EvalBodyPoseInWorld(self.temp_context, panda1_pad_body)
        robot1_translation = panda1_pad_pose.translation()

        # Set positions for robot 2
        self.plant.SetPositions(self.temp_context, self.panda2_id, robot2_joints)
        panda2_pad_body = self.plant.GetBodyByName("rubber_pad", self.panda2_id)
        panda2_pad_pose = self.plant.EvalBodyPoseInWorld(self.temp_context, panda2_pad_body)
        robot2_translation = panda2_pad_pose.translation()

        combined_translation = np.concatenate([robot1_translation, robot2_translation])
        output.SetFromVector(
            [
                combined_translation[0],
                combined_translation[1],
                combined_translation[2],
                combined_translation[3],
                combined_translation[4],
                combined_translation[5],
            ]
        )

######################################################################################################
#              ##################Robot Link Index Accessor################
######################################################################################################
class RobotLinkIndexAccessor(LeafSystem):
    """
    Provides access to body indexes and link information for both robots.
    Useful for getting specific link indexes like panda_link7 for both robots.
    """
    def __init__(self, plant, panda1_id, panda2_id):
        super().__init__()
        self.plant = plant
        self.panda1_id = panda1_id
        self.panda2_id = panda2_id
        
        # Output port for link indexes (2 values: [robot1_link7_idx, robot2_link7_idx])
        self.DeclareVectorOutputPort("link7_indexes", size=2, calc=self.CalcOutput)
        
        # Get the body indexes for panda_link7 of both robots
        self._get_link_indexes()
    
    def _get_link_indexes(self):
        """Get the body indexes for panda_link7 of both robots."""
        try:
            # Get panda_link7 body for robot 1
            self.robot1_link7_body = self.plant.GetBodyByName("panda_link7", self.panda1_id)
            self.robot1_link7_index = self.robot1_link7_body.index()
            
            # Get panda_link7 body for robot 2
            self.robot2_link7_body = self.plant.GetBodyByName("panda_link7", self.panda2_id)
            self.robot2_link7_index = self.robot2_link7_body.index()
            
            print(f"Robot 1 panda_link7 body index: {self.robot1_link7_index}")
            print(f"Robot 2 panda_link7 body index: {self.robot2_link7_index}")
            
        except Exception as e:
            print(f"Error getting link indexes: {e}")
            self.robot1_link7_index = -1
            self.robot2_link7_index = -1
    
    def CalcOutput(self, context, output):
        """Output the link7 indexes for both robots."""
        indexes = [self.robot1_link7_index, self.robot2_link7_index]
        output.SetFromVector(indexes)
    
    def get_robot1_link7_index(self):
        """Get the body index for robot 1's panda_link7."""
        return self.robot1_link7_index
    
    def get_robot2_link7_index(self):
        """Get the body index for robot 2's panda_link7."""
        return self.robot2_link7_index
    
    def get_robot1_link7_body(self):
        """Get the body object for robot 1's panda_link7."""
        return self.robot1_link7_body
    
    def get_robot2_link7_body(self):
        """Get the body object for robot 2's panda_link7."""
        return self.robot2_link7_body
    
    def get_all_link_info(self):
        """Get comprehensive link information for both robots."""
        info = {
            'robot1': {
                'model_id': self.panda1_id,
                'link7_body': self.robot1_link7_body,
                'link7_index': self.robot1_link7_index,
                'link7_name': 'panda_link7'
            },
            'robot2': {
                'model_id': self.panda2_id,
                'link7_body': self.robot2_link7_body,
                'link7_index': self.robot2_link7_index,
                'link7_name': 'panda_link7'
            }
        }
        return info

######################################################################################################
#              ##################Main System Setup################
######################################################################################################

# Create a simple system to split IK output for dual robots
class IKSplitter(LeafSystem):
    def __init__(self, nq1, nq2):
        super().__init__()
        self.nq1 = int(nq1)
        self.nq2 = int(nq2)
        self.DeclareVectorInputPort("ik_joint_targets", size=self.nq1 + self.nq2)  # Full IK output
        self.DeclareVectorOutputPort("robot1_targets", size=self.nq1, calc=self.CalcRobot1Output)
        self.DeclareVectorOutputPort("robot2_targets", size=self.nq2, calc=self.CalcRobot2Output)
    
    def CalcRobot1Output(self, context, output):
        full_targets = self.GetInputPort("ik_joint_targets").Eval(context)
        robot1_targets = full_targets[: self.nq1]
        output.SetFromVector(robot1_targets)
    
    def CalcRobot2Output(self, context, output):
        full_targets = self.GetInputPort("ik_joint_targets").Eval(context)
        robot2_targets = full_targets[self.nq1 : self.nq1 + self.nq2]
        output.SetFromVector(robot2_targets)

# Add all systems to the diagram
dual_ik_tracker = builder.AddSystem(DualRelaxedIKBoxTracker(plant, plant_context, panda1_id, panda2_id))
ik_splitter = builder.AddSystem(IKSplitter(num_positions_1, num_positions_2))

# Create two separate ERG systems (one for each robot)
erg1 = builder.AddSystem(ERG(plant, panda1_id, "robot1"))
erg2 = builder.AddSystem(ERG(plant, panda2_id, "robot2"))

dual_contact_converter = builder.AddSystem(DualContactForceConverter(plant, scene_graph, panda1_id, panda2_id))
dual_pose_extractor = builder.AddSystem(DualPandaLink7PoseExtractor(plant, panda1_id, panda2_id))
link_index_accessor = builder.AddSystem(RobotLinkIndexAccessor(plant, panda1_id, panda2_id))

# Add FCL distance systems for both robots
fcl_robot1 = builder.AddSystem(FCLLinkDistanceSystem(plant, panda1_id, movable_box_id))
fcl_robot2 = builder.AddSystem(FCLLinkDistanceSystem(plant, panda2_id, movable_box_id))


# Add PD(+G) controllers for both robots
if num_positions_1 == 7:
    Kp1 = np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0])
    Kd1 = np.array([8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0])
else:
    Kp1 = np.full(num_positions_1, 100.0)
    Kd1 = np.full(num_positions_1, 5.0)

if num_positions_2 == 7:
    Kp2 = np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0])
    Kd2 = np.array([8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0])
else:
    Kp2 = np.full(num_positions_2, 100.0)
    Kd2 = np.full(num_positions_2, 5.0)

controller1 = builder.AddNamedSystem("PD+G controller 1", PD_gravity(plant, Kp1, Kd1, panda1_id))
controller2 = builder.AddNamedSystem("PD+G controller 2", PD_gravity(plant, Kp2, Kd2, panda2_id))

# Add all loggers
tau_logger1 = LogVectorOutput(controller1.GetOutputPort("tau_u"), builder)
tau_logger1.set_name("tau_logger1")
tau_logger2 = LogVectorOutput(controller2.GetOutputPort("tau_u"), builder)
tau_logger2.set_name("tau_logger2")

state_logger1 = LogVectorOutput(plant.get_state_output_port(panda1_id), builder)
state_logger1.set_name("state_logger1")
state_logger2 = LogVectorOutput(plant.get_state_output_port(panda2_id), builder)
state_logger2.set_name("state_logger2")

erg_logger1 = LogVectorOutput(erg1.GetOutputPort("q_v_filtered"), builder)
erg_logger1.set_name("erg_logger1")
erg_logger2 = LogVectorOutput(erg2.GetOutputPort("q_v_filtered"), builder)
erg_logger2.set_name("erg_logger2")
erg_dsm_logger1 = LogVectorOutput(erg1.GetOutputPort("dsm"), builder)
erg_dsm_logger1.set_name("erg_dsm_logger1")
erg_dsm_logger2 = LogVectorOutput(erg2.GetOutputPort("dsm"), builder)
erg_dsm_logger2.set_name("erg_dsm_logger2")
ik_target_xyz_logger = LogVectorOutput(dual_ik_tracker.GetOutputPort("ee_target_positions"), builder)
ik_target_xyz_logger.set_name("ik_target_xyz_logger")
ik_ref_logger1 = LogVectorOutput(ik_splitter.GetOutputPort("robot1_targets"), builder)
ik_ref_logger1.set_name("ik_ref_logger1")
ik_ref_logger2 = LogVectorOutput(ik_splitter.GetOutputPort("robot2_targets"), builder)
ik_ref_logger2.set_name("ik_ref_logger2")

box_pos_logger = LogVectorOutput(plant.get_state_output_port(movable_box_id), builder)
box_pos_logger.set_name("box_pos_logger")

contact_logger = LogVectorOutput(dual_contact_converter.GetOutputPort("contact_forces"), builder)
contact_logger.set_name("contact_logger")

pose_logger = LogVectorOutput(dual_pose_extractor.GetOutputPort("panda_link7_world_positions"), builder)
pose_logger.set_name("pose_logger")
rpad_logger = LogVectorOutput(dual_pose_extractor.GetOutputPort("rubber_pad_world_positions"), builder)
rpad_logger.set_name("rpad_logger")

link_index_logger = LogVectorOutput(link_index_accessor.GetOutputPort("link7_indexes"), builder)
link_index_logger.set_name("link_index_logger")

# Add FCL distance loggers for both robots
fcl_logger1 = LogVectorOutput(fcl_robot1.get_output_port(0), builder)
fcl_logger1.set_name("fcl_logger1")
fcl_logger2 = LogVectorOutput(fcl_robot2.get_output_port(0), builder)
fcl_logger2.set_name("fcl_logger2")



# Connect all systems
# Box state to IK tracker
builder.Connect(plant.get_state_output_port(movable_box_id), dual_ik_tracker.GetInputPort("box_state"))

# Connect IK system to ERG (replacing the trajectory) - following test_erg.py pattern
builder.Connect(dual_ik_tracker.GetOutputPort("ik_joint_targets"), ik_splitter.GetInputPort("ik_joint_targets"))

# Split IK output for ERG
builder.Connect(ik_splitter.GetOutputPort("robot1_targets"), erg1.GetInputPort("q_r"))
builder.Connect(ik_splitter.GetOutputPort("robot2_targets"), erg2.GetInputPort("q_r"))

# Box state to ERG
builder.Connect(plant.get_state_output_port(movable_box_id), erg1.GetInputPort("box_state"))
builder.Connect(plant.get_state_output_port(movable_box_id), erg2.GetInputPort("box_state"))

# Connect plant state and actuation to ERG (following test_erg.py pattern)
builder.Connect(plant.get_state_output_port(panda1_id), erg1.GetInputPort("state"))
builder.Connect(plant.GetOutputPort("panda_net_actuation"), erg1.GetInputPort("tau"))
builder.Connect(plant.get_state_output_port(panda2_id), erg2.GetInputPort("state"))
builder.Connect(plant.GetOutputPort("panda_net_actuation"), erg2.GetInputPort("tau"))

# ERG outputs to controllers
builder.Connect(erg1.GetOutputPort("q_v_filtered"), controller1.GetInputPort("Desired_state"))
builder.Connect(erg2.GetOutputPort("q_v_filtered"), controller2.GetInputPort("Desired_state"))

# Robot states to controllers
builder.Connect(plant.get_state_output_port(panda1_id), controller1.GetInputPort("Current_state"))
builder.Connect(plant.get_state_output_port(panda2_id), controller2.GetInputPort("Current_state"))

# Controllers to plant
builder.Connect(controller1.GetOutputPort("tau_u"), plant.get_actuation_input_port(panda1_id))
builder.Connect(controller2.GetOutputPort("tau_u"), plant.get_actuation_input_port(panda2_id))

# Contact results to converter
builder.Connect(plant.get_contact_results_output_port(), dual_contact_converter.GetInputPort("contact_results"))

# Robot states to pose extractor
builder.Connect(plant.get_state_output_port(panda1_id), dual_pose_extractor.GetInputPort("robot1_joint_positions"))
builder.Connect(plant.get_state_output_port(panda2_id), dual_pose_extractor.GetInputPort("robot2_joint_positions"))

# Connect plant state to FCL distance systems
builder.Connect(plant.get_state_output_port(), fcl_robot1.GetInputPort("x"))
builder.Connect(plant.get_state_output_port(), fcl_robot2.GetInputPort("x"))



# Add visualization
if meshcat_visualisation:
    AddDefaultVisualization(builder=builder)

# Build the diagram
diagram = builder.Build()

# Create simulator
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(realtime_factor)
simulator.Initialize()

# Run simulation
print("Starting dual robot simulation...")
sim_duration = float(os.environ.get("SIM_DURATION", "10.0"))
simulator.AdvanceTo(sim_duration)

print("Simulation completed!")

# Get logged data
diagram_context = simulator.get_mutable_context()
log_x1 = state_logger1.FindLog(diagram_context)
log_x2 = state_logger2.FindLog(diagram_context)
log_tau1 = tau_logger1.FindLog(diagram_context)
log_tau2 = tau_logger2.FindLog(diagram_context)
log_erg1 = erg_logger1.FindLog(diagram_context)
log_erg2 = erg_logger2.FindLog(diagram_context)
log_dsm1 = erg_dsm_logger1.FindLog(diagram_context)
log_dsm2 = erg_dsm_logger2.FindLog(diagram_context)
log_ik_target_xyz = ik_target_xyz_logger.FindLog(diagram_context)
log_ik_ref1 = ik_ref_logger1.FindLog(diagram_context)
log_ik_ref2 = ik_ref_logger2.FindLog(diagram_context)
log_box = box_pos_logger.FindLog(diagram_context)
log_contact = contact_logger.FindLog(diagram_context)
log_pose = pose_logger.FindLog(diagram_context)
log_rpad = rpad_logger.FindLog(diagram_context)
log_fcl1 = fcl_logger1.FindLog(diagram_context)
log_fcl2 = fcl_logger2.FindLog(diagram_context)


# Extract time data
t_time = log_x1.sample_times()

# Extract joint data for both robots
data_q1 = log_x1.data().transpose()[:, 0:7]   # Robot 1 joints
data_q2 = log_x2.data().transpose()[:, 0:7]   # Robot 2 joints
data_qdot1 = log_x1.data().transpose()[:, 7:14]  # Robot 1 velocities
data_qdot2 = log_x2.data().transpose()[:, 7:14]  # Robot 2 velocities

# Extract torques for both robots
data_tau1 = log_tau1.data().transpose()
data_tau2 = log_tau2.data().transpose()

# Extract ERG outputs for both robots
data_erg1 = log_erg1.data().transpose()
data_erg2 = log_erg2.data().transpose()
t_erg1 = log_erg1.sample_times()
t_erg2 = log_erg2.sample_times()
data_dsm1 = log_dsm1.data().transpose()[:, 0]
data_dsm2 = log_dsm2.data().transpose()[:, 0]
t_dsm1 = log_dsm1.sample_times()
t_dsm2 = log_dsm2.sample_times()

# Extract IK cartesian xyz targets for both robots
data_ik_target_xyz = log_ik_target_xyz.data().transpose()  # shape: (N, 6)
t_ik_target_xyz = log_ik_target_xyz.sample_times()

# Extract RelaxedIK references (q_r) for both robots
data_ik_ref1 = log_ik_ref1.data().transpose()
data_ik_ref2 = log_ik_ref2.data().transpose()
t_ik_ref1 = log_ik_ref1.sample_times()
t_ik_ref2 = log_ik_ref2.sample_times()

# Extract box position data
data_box_pos = log_box.data().transpose()

# Extract contact force data
data_contact_forces = log_contact.data().transpose()

# Extract pose data for both robots
data_pose = log_pose.data().transpose()

# Extract rubber_pad xyz data for both robots
data_rpad = log_rpad.data().transpose()  # shape: (N, 6)
t_rpad = log_rpad.sample_times()

# Extract FCL distance data for both robots
data_fcl1 = log_fcl1.data().transpose()  # Robot 1 FCL distances (4 links)
data_fcl2 = log_fcl2.data().transpose()  # Robot 2 FCL distances (4 links)



print("\n=== Dual Robot Simulation Results ===")
print(f"Robot 1 final joint positions: {data_q1[-1, :]}")
print(f"Robot 2 final joint positions: {data_q2[-1, :]}")
print(f"Box final position: {data_box_pos[-1, 4:7]}")  # x, y, z position
print(f"Final contact forces - Robot 1: {data_contact_forces[-1, :3]}, Robot 2: {data_contact_forces[-1, 3:6]}")
print(f"Final FCL distances - Robot 1: {data_fcl1[-1, :]} (link7, hand, leftfinger, rightfinger)")
print(f"Final FCL distances - Robot 2: {data_fcl2[-1, :]} (link7, hand, leftfinger, rightfinger)")
print(f"Final DSM - Robot 1: {data_dsm1[-1]:.4f}, Robot 2: {data_dsm2[-1]:.4f}")

# Extract final rubber_pad poses (position + orientation) for both robots.
plant_context_final = plant.GetMyContextFromRoot(diagram_context)
robot1_pad_pose = plant.EvalBodyPoseInWorld(
    plant_context_final, plant.GetBodyByName("rubber_pad", panda1_id)
)
robot2_pad_pose = plant.EvalBodyPoseInWorld(
    plant_context_final, plant.GetBodyByName("rubber_pad", panda2_id)
)

robot1_pad_pos = robot1_pad_pose.translation()
robot2_pad_pos = robot2_pad_pose.translation()

print(f"Robot 1 final rubber_pad position: {robot1_pad_pos}")
print(f"Robot 2 final rubber_pad position: {robot2_pad_pos}")
print(f"Robot 1 final rubber_pad quaternion [w, x, y, z]: {robot1_pad_pose.rotation().ToQuaternion().wxyz()}")
print(f"Robot 2 final rubber_pad quaternion [w, x, y, z]: {robot2_pad_pose.rotation().ToQuaternion().wxyz()}")


# Create plots for both robots
plt.figure(figsize=(14, 12))

# Robot 1 joint positions
plt.subplot(3, 3, 1)
for i in range(7):  # All 9 joints (7 arm + 2 gripper)
    plt.plot(t_time, data_q1[:, i], label=f'Joint {i+1}')
plt.title('Robot 1 Joint Positions')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.legend()
plt.grid(True)

# Robot 2 joint positions
plt.subplot(3, 3, 2)
for i in range(7):  # All 9 joints (7 arm + 2 gripper)
    plt.plot(t_time, data_q2[:, i], label=f'Joint {i+1}')
plt.title('Robot 2 Joint Positions')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.legend()
plt.grid(True)

# Box position
plt.subplot(3, 3, 3)
box_positions = data_box_pos[:, 4:7]  # x, y, z
plt.plot(t_time, box_positions[:, 0], label='X')
plt.plot(t_time, box_positions[:, 1], label='Y')
plt.plot(t_time, box_positions[:, 2], label='Z')
plt.title('Box Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

# Robot 1 torques
plt.subplot(3, 3, 4)
for i in range(7):  # All 9 joints (7 arm + 2 gripper)
    plt.plot(t_time, data_tau1[:, i], label=f'Joint {i+1}')
plt.title('Robot 1 Torques')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

# Robot 2 torques
plt.subplot(3, 3, 5)
for i in range(7):  # All 9 joints (7 arm + 2 gripper)
    plt.plot(t_time, data_tau2[:, i], label=f'Joint {i+1}')
plt.title('Robot 2 Torques')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

# Contact forces
plt.subplot(3, 3, 6)
plt.plot(t_time, data_contact_forces[:, 0], label='Robot 1 X')
plt.plot(t_time, data_contact_forces[:, 1], label='Robot 1 Y')
plt.plot(t_time, data_contact_forces[:, 2], label='Robot 1 Z')
plt.plot(t_time, data_contact_forces[:, 3], label='Robot 2 X')
plt.plot(t_time, data_contact_forces[:, 4], label='Robot 2 Y')
plt.plot(t_time, data_contact_forces[:, 5], label='Robot 2 Z')
plt.title('Contact Forces')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.grid(True)

# FCL distances for Robot 1
plt.subplot(3, 3, 7)
fcl_labels = ['link7', 'hand', 'leftfinger', 'rightfinger']
for i in range(4):
    plt.plot(t_time, data_fcl1[:, i], label=fcl_labels[i])
plt.axhline(0.0, color='r', linestyle='--', alpha=0.6)
plt.title('Robot 1 FCL Distances to Box')
plt.xlabel('Time (s)')
plt.ylabel('Signed Distance (m)')
plt.legend()
plt.grid(True)

# FCL distances for Robot 2
plt.subplot(3, 3, 8)
for i in range(4):
    plt.plot(t_time, data_fcl2[:, i], label=fcl_labels[i])
plt.axhline(0.0, color='r', linestyle='--', alpha=0.6)
plt.title('Robot 2 FCL Distances to Box')
plt.xlabel('Time (s)')
plt.ylabel('Signed Distance (m)')
plt.legend()
plt.grid(True)

# Combined minimum FCL distances
plt.subplot(3, 3, 7)
min_fcl1 = np.min(data_fcl1, axis=1)  # Minimum distance per timestep for robot 1
min_fcl2 = np.min(data_fcl2, axis=1)  # Minimum distance per timestep for robot 2
plt.plot(t_time, min_fcl1, label='Robot 1 Min Distance', linewidth=2)
plt.plot(t_time, min_fcl2, label='Robot 2 Min Distance', linewidth=2)
plt.axhline(0.0, color='r', linestyle='--', alpha=0.6)
plt.title('Minimum FCL Distances to Box')
plt.xlabel('Time (s)')
plt.ylabel('Min Signed Distance (m)')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Plot joint-level comparison: Actual q vs ERG command vs RelaxedIK reference.
joint_limits_lower = np.zeros(7)
joint_limits_upper = np.zeros(7)
try:
    for j in range(7):
        joint_name = f"panda_joint{j+1}"
        joint = plant.GetJointByName(joint_name, panda1_id)
        joint_limits_lower[j] = joint.position_lower_limits()[0]
        joint_limits_upper[j] = joint.position_upper_limits()[0]
except Exception as e:
    print(f"Could not read joint limits from plant ({e}); using Panda defaults.")
    joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

fig_cmp, axes_cmp = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
for j in range(7):
    row = 0 if j < 4 else 1
    col = j if j < 4 else j - 4
    ax = axes_cmp[row, col]
    lim_low = joint_limits_lower[j]
    lim_high = joint_limits_upper[j]
    ax.axhspan(
        lim_low,
        lim_high,
        color="0.85",
        alpha=0.25,
        label="Joint limits" if j == 0 else None,
        zorder=0,
    )
    ax.axhline(lim_low, color="0.5", linestyle="-.", linewidth=1.0)
    ax.axhline(lim_high, color="0.5", linestyle="-.", linewidth=1.0)
    ax.plot(t_time, data_q1[:, j], color="k", linewidth=1.8, label="Robot1 actual q")
    ax.plot(t_erg1, data_erg1[:, j], color="tab:blue", linestyle="--", linewidth=1.4, label="Robot1 ERG cmd")
    ax.plot(t_ik_ref1, data_ik_ref1[:, j], color="tab:green", linestyle=":", linewidth=1.4, label="Robot1 RelaxedIK ref")
    ax.plot(t_time, data_q2[:, j], color="0.45", linewidth=1.8, label="Robot2 actual q")
    ax.plot(t_erg2, data_erg2[:, j], color="tab:orange", linestyle="--", linewidth=1.4, label="Robot2 ERG cmd")
    ax.plot(t_ik_ref2, data_ik_ref2[:, j], color="tab:red", linestyle=":", linewidth=1.4, label="Robot2 RelaxedIK ref")
    ax.set_title(f"Joint {j+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.grid(True, alpha=0.3)

# Use last empty subplot for a compact legend.
axes_cmp[1, 3].axis("off")
handles, labels = axes_cmp[0, 0].get_legend_handles_labels()
axes_cmp[1, 3].legend(handles, labels, loc="center", frameon=True)
fig_cmp.suptitle("Joint Tracking: Actual vs ERG vs RelaxedIK Reference (Both Robots)")
fig_cmp.tight_layout()

# Plot DSM from trajectoryERG output.
plt.figure(figsize=(10, 4))
plt.plot(t_dsm1, data_dsm1, label="Robot 1 DSM", linewidth=2.0)
plt.plot(t_dsm2, data_dsm2, label="Robot 2 DSM", linewidth=2.0)
plt.axhline(0.0, color="k", linestyle="--", alpha=0.6, linewidth=1.0)
plt.title("TrajectoryERG Dynamic Safety Margin (DSM)")
plt.xlabel("Time (s)")
plt.ylabel("DSM")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Compare IK xyz targets vs actual rubber_pad world xyz (both robots).
fig_xyz_cmp, axes_xyz_cmp = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
labels = ["X (m)", "Y (m)", "Z (m)"]
for k in range(3):
    # Robot 1: target vs actual
    axes_xyz_cmp[k].plot(
        t_ik_target_xyz, data_ik_target_xyz[:, k],
        color="tab:blue", linestyle="--", linewidth=2.0, label="Robot 1 target" if k == 0 else None
    )
    axes_xyz_cmp[k].plot(
        t_rpad, data_rpad[:, k],
        color="tab:blue", linestyle="-", linewidth=2.0, label="Robot 1 rubber_pad" if k == 0 else None
    )

    # Robot 2: target vs actual
    axes_xyz_cmp[k].plot(
        t_ik_target_xyz, data_ik_target_xyz[:, 3 + k],
        color="tab:orange", linestyle="--", linewidth=2.0, label="Robot 2 target" if k == 0 else None
    )
    axes_xyz_cmp[k].plot(
        t_rpad, data_rpad[:, 3 + k],
        color="tab:orange", linestyle="-", linewidth=2.0, label="Robot 2 rubber_pad" if k == 0 else None
    )

    axes_xyz_cmp[k].set_ylabel(labels[k])
    axes_xyz_cmp[k].grid(True, alpha=0.3)

axes_xyz_cmp[0].set_title("IK Target XYZ vs rubber_pad XYZ (World)")
axes_xyz_cmp[-1].set_xlabel("Time (s)")
axes_xyz_cmp[0].legend()
fig_xyz_cmp.tight_layout()

plt.show()



print("Dual robot simulation and analysis completed!") 
