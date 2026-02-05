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
from trajectoryERG import CompliantERG
from CERG_Setup import ErgParams, ContactParams, panda_spec
import os
from pydrake.visualization import AddDefaultVisualization


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
        self.Kp_ = Kp
        self.Kd_ = Kd
        
        # Input ports - match original design: 9 joints for desired, 18 for current state
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=9)  # 9 joints (7 arm + 2 gripper)
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=18)  # 9 pos + 9 vel
        
        # Output port for control torques - 9 joints
        self.DeclareVectorOutputPort("tau_u", size=9, calc=self._calc_output)
        
        # Declare discrete state for storing computed torques - 9 joints
        self.DeclareDiscreteState(9)
        
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
        q_d = self._desired_state_port.Eval(context)  # 9 joints
        q_full = self._current_state_port.Eval(context)  # 18 states (9 pos + 9 vel)
        
        # Extract position and velocity
        q = q_full[:9]   # First 9 elements are positions
        q_dot = q_full[9:18]  # Last 9 elements are velocities
        
        # Create plant context for gravity calculation
        plant_context = self.plant.CreateDefaultContext()
        
        # Set robot state in plant context
        if self.robot_id is not None:
            # For dual robot setup, we need to set the state for the specific robot
            # First get the current full plant state
            full_state = self.plant.GetPositionsAndVelocities(plant_context)
            
            # Calculate the starting index for this robot in the full state vector
            # This assumes robots are added sequentially to the plant
            if self.robot_id == self.plant.GetModelInstanceByName("panda"):
                # First robot - start from beginning
                start_idx = 0
            elif self.robot_id == self.plant.GetModelInstanceByName("panda_1"):
                # Second robot - start after first robot's states
                start_idx = 18  # First robot has 18 states (9 pos + 9 vel)
            else:
                # Fallback - assume first 9 elements
                start_idx = 0
            
            # Set the state for this specific robot
            robot_state = q_full[:9]  # Just the 9 joint positions for robot
            self.plant.SetPositions(plant_context, self.robot_id, robot_state)
        
        # Calculate gravity compensation
        gravity_full = -self.plant.CalcGravityGeneralizedForces(plant_context)
        gravity_full_np = np.array(gravity_full).flatten()
        
        # Extract gravity for this robot
        if self.robot_id is not None:
            # Get the starting index for this robot's gravity forces
            if self.robot_id == self.plant.GetModelInstanceByName("panda"):
                # First robot - first 9 elements
                gravity = gravity_full_np[:9]
            elif self.robot_id == self.plant.GetModelInstanceByName("panda_1"):
                # Second robot - elements after first robot
                gravity = gravity_full_np[9:18]  # Assuming 9 joints per robot
            else:
                # Fallback
                gravity = gravity_full_np[:9]
        else:
            # Assume first 9 elements for backward compatibility
            gravity = gravity_full_np[:9]
        
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
    panda_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_fr3.urdf"))
    panda_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_fr3_2.urdf"))

    
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
    init_pos1 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]
    init_pos2 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]
    
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

meshcat_visualisation = True
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
    def __init__(self):
        super().__init__()  # Don't forget to initialize the base class.
        self._state_port = self.DeclareVectorInputPort(name="state", size=18)
        self._tau_port = self.DeclareVectorInputPort(name="tau", size=9)
        self._qr_port = self.DeclareVectorInputPort(name="q_r", size=9)  # Back to 9 joints
        self._box_state_port = self.DeclareVectorInputPort(name="box_state", size=13)

        state_index = self.DeclareDiscreteState(9)  # Back to 9 joints
        self.DeclareStateOutputPort("q_v_filtered", state_index)  # One output: y=x.
        
        # Add output port for calculated energy
        self.DeclareVectorOutputPort("calculated_energy", size=1, calc=self.output_energy)
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.refrence) # Call the Update method defined below.
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_drake.urdf"))
        robot_spec = panda_spec(urdf_path)
        erg_params = ErgParams()
        contact_params = ContactParams()
        self.erg = CompliantERG(
            robot_spec=robot_spec,
            erg_params=erg_params,
            contact_params=contact_params)

        # Initialize a flag to check if it's the first update
        self.first_update = True

    def refrence(self, context, discrete_state):
        # Evaluate the input ports
        state = self._state_port.Eval(context)
        q = state[:9]  # First 9 elements are positions
        dq = state[9:18]  # Last 9 elements are velocities
        tau = self._tau_port.Eval(context)
        q_r = self._qr_port.Eval(context)
        box_state = self._box_state_port.Eval(context)
        # Extract and post-process box position (subtract 0.11 from x)
        box_position = np.array(box_state[4:7], dtype=float)
        box_position[0] -= 0.11

        self.q_v = context.get_discrete_state_vector().CopyToVector()              
        # Initialize q_v_ only at the first callback
        if self.first_update:
            self.q_v_ = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]  # Use all 9 joints    
            self.first_update = False
        
        # Update box position in ERG
        self.q_v_ = self.erg.get_qv(q, dq, tau, q_r, self.q_v_, box_position)
        
        # Calculate energy from trajectory predictions
        self.calculated_energy = self.erg.get_energy(q, dq, tau, q_r, self.q_v_, box_position)

        # Write into the output vector.
        discrete_state.get_mutable_vector().SetFromVector(self.q_v_)
        
    def output_energy(self, context, output):
        # Return the calculated energy from the ERG system
        if hasattr(self, 'calculated_energy'):
            output.SetAtIndex(0, self.calculated_energy)
        else:
            output.SetAtIndex(0, 0.0)

######################################################################################################
#              ##################Relaxed IK System for Box Tracking################
######################################################################################################
class DualRelaxedIKBoxTracker(LeafSystem):
    def __init__(self, plant, plant_context):
        super().__init__()
        
        # Store plant and context references
        self.plant = plant
        self.plant_context = plant_context
        
        # Declare input port for box state (13 values: 7 pose + 6 velocities)
        self._box_state_port = self.DeclareVectorInputPort(name="box_state", size=13)
        
        # State for both robots (9 joint angles each)
        state_index = self.DeclareDiscreteState(18)  # 9 + 9 for both robots
        self.DeclareStateOutputPort("ik_joint_targets", state_index)
        
        # Periodic update for IK solving
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # 10Hz IK updates
            offset_sec=0.0,
            update=self.solve_ik)
        
        # Initialize Relaxed IK for both robots
        try:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_settings.yaml"))
            self.rik1 = RelaxedIKRust(setting_file_path=config_path)
            self.rik2 = RelaxedIKRust(setting_file_path=config_path)
            print("Relaxed IK initialized successfully for both robots!")
            self.ik_available = True
        except Exception as e:
            print(f"Failed to initialize Relaxed IK: {e}")
            self.ik_available = False
        
        # Tolerance for IK solving
        self.tolerance = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]  # x,y,z and rx,ry,rz tolerances
        
        # Target positions for both robots (slightly offset to avoid collision)
        self.target_position1 = [0.6, 0.0, 0.2]  # Robot 1 target
        self.target_position2 = [0.6, 0.0, 0.2]  # Robot 2 target (same target for now)
    
    def solve_ik(self, context, discrete_state):
        """Solve IK for both robots to track the box."""
        if not self.ik_available:
            return
        
        # Get current box state
        box_state = self._box_state_port.Eval(context)
        box_pos = box_state[4:7]  # Extract position (x, y, z)
        
        # Apply offset for both robots (subtract 0.11 from x position)
        target_position1 = [box_pos[0] - 0.11, box_pos[1], box_pos[2]]
        target_position2 = [box_pos[0] - 0.11, box_pos[1], box_pos[2]]
        
        # Get current robot states
        current_state = discrete_state.get_mutable_vector()
        q1_current = current_state.value()[:9]   # First 9 joints for robot 1
        q2_current = current_state.value()[9:18] # Next 9 joints for robot 2
        
        try:
            # Solve IK for robot 1 - provide target position, orientation, and tolerance
            # Use default orientation (identity quaternion) for now
            default_orientation = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
            q1_solution = self.rik1.solve_position(target_position1, default_orientation, self.tolerance)
            
            # Solve IK for robot 2 - provide target position, orientation, and tolerance
            q2_solution = self.rik2.solve_position(target_position2, default_orientation, self.tolerance)
            
            # Combine arm solutions with current gripper positions
            q1_full = np.concatenate([q1_solution, q1_current[7:9]])  # Arm + gripper
            q2_full = np.concatenate([q2_solution, q2_current[7:9]])  # Arm + gripper
            
            # Update the state with both solutions
            solution = np.concatenate([q1_full, q2_full])
            current_state.SetFromVector(solution)
            
        except Exception as e:
            print(f"IK solving failed: {e}")
            # Keep current joint positions if IK fails

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
        
        # Input ports for both robots
        self.DeclareVectorInputPort("robot1_joint_positions", size=18)
        self.DeclareVectorInputPort("robot2_joint_positions", size=18)
        
        # Output ports for both robots (6 values: [robot1_x, robot1_y, robot1_z, robot2_x, robot2_y, robot2_z])
        self.DeclareVectorOutputPort("panda_link7_world_positions", size=6, calc=self.CalcOutput)
        
        self.temp_context = plant.CreateDefaultContext()
    
    def CalcOutput(self, context, output):
        # Get joint positions for both robots
        robot1_state = self.GetInputPort("robot1_joint_positions").Eval(context)
        robot2_state = self.GetInputPort("robot2_joint_positions").Eval(context)
        
        # Extract joint positions (first 9 values)
        robot1_joints = robot1_state[:9]
        robot2_joints = robot2_state[:9]
        
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
    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort("ik_joint_targets", size=18)  # Full IK output
        self.DeclareVectorOutputPort("robot1_targets", size=9, calc=self.CalcRobot1Output)
        self.DeclareVectorOutputPort("robot2_targets", size=9, calc=self.CalcRobot2Output)
    
    def CalcRobot1Output(self, context, output):
        full_targets = self.GetInputPort("ik_joint_targets").Eval(context)
        robot1_targets = full_targets[:9]  # First 9 elements for robot 1
        output.SetFromVector(robot1_targets)
    
    def CalcRobot2Output(self, context, output):
        full_targets = self.GetInputPort("ik_joint_targets").Eval(context)
        robot2_targets = full_targets[9:18]  # Last 9 elements for robot 2
        output.SetFromVector(robot2_targets)

# Add all systems to the diagram
dual_ik_tracker = builder.AddSystem(DualRelaxedIKBoxTracker(plant, plant_context))
ik_splitter = builder.AddSystem(IKSplitter())

# Create two separate ERG systems (one for each robot) following test_erg.py pattern
erg1 = builder.AddSystem(ERG())
erg2 = builder.AddSystem(ERG())

dual_contact_converter = builder.AddSystem(DualContactForceConverter(plant, scene_graph, panda1_id, panda2_id))
dual_pose_extractor = builder.AddSystem(DualPandaLink7PoseExtractor(plant, panda1_id, panda2_id))
link_index_accessor = builder.AddSystem(RobotLinkIndexAccessor(plant, panda1_id, panda2_id))

# Add FCL distance systems for both robots
fcl_robot1 = builder.AddSystem(FCLLinkDistanceSystem(plant, panda1_id, movable_box_id))
fcl_robot2 = builder.AddSystem(FCLLinkDistanceSystem(plant, panda2_id, movable_box_id))


# Add PID controllers for both robots
Kp1 =  np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0, 120.0, 120.0])  # 9 joints
Kd1 =  np.array([8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0, 5.0, 5.0])  # 9 joints
Kp2 =  np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0, 120.0, 120.0])  # 9 joints
Kd2 =  np.array([8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0, 5.0, 5.0])  # 9 joints

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

box_pos_logger = LogVectorOutput(plant.get_state_output_port(movable_box_id), builder)
box_pos_logger.set_name("box_pos_logger")

contact_logger = LogVectorOutput(dual_contact_converter.GetOutputPort("contact_forces"), builder)
contact_logger.set_name("contact_logger")

pose_logger = LogVectorOutput(dual_pose_extractor.GetOutputPort("panda_link7_world_positions"), builder)
pose_logger.set_name("pose_logger")

link_index_logger = LogVectorOutput(link_index_accessor.GetOutputPort("link7_indexes"), builder)
link_index_logger.set_name("link_index_logger")

# Add FCL distance loggers for both robots
fcl_logger1 = LogVectorOutput(fcl_robot1.get_output_port(0), builder)
fcl_logger1.set_name("fcl_logger1")
fcl_logger2 = LogVectorOutput(fcl_robot2.get_output_port(0), builder)
fcl_logger2.set_name("fcl_logger2")

# Add energy loggers for both ERG systems
energy_logger1 = LogVectorOutput(erg1.GetOutputPort("calculated_energy"), builder)
energy_logger1.set_name("energy_logger1")
energy_logger2 = LogVectorOutput(erg2.GetOutputPort("calculated_energy"), builder)
energy_logger2.set_name("energy_logger2")



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
AddDefaultVisualization(builder=builder)

# Build the diagram
diagram = builder.Build()

# Create simulator
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(realtime_factor)
simulator.Initialize()

# Run simulation
print("Starting dual robot simulation...")
simulator.AdvanceTo(2.0) 

print("Simulation completed!")

# Get logged data
diagram_context = simulator.get_mutable_context()
log_x1 = state_logger1.FindLog(diagram_context)
log_x2 = state_logger2.FindLog(diagram_context)
log_tau1 = tau_logger1.FindLog(diagram_context)
log_tau2 = tau_logger2.FindLog(diagram_context)
log_erg1 = erg_logger1.FindLog(diagram_context)
log_erg2 = erg_logger2.FindLog(diagram_context)
log_box = box_pos_logger.FindLog(diagram_context)
log_contact = contact_logger.FindLog(diagram_context)
log_pose = pose_logger.FindLog(diagram_context)
log_fcl1 = fcl_logger1.FindLog(diagram_context)
log_fcl2 = fcl_logger2.FindLog(diagram_context)
log_energy1 = energy_logger1.FindLog(diagram_context)
log_energy2 = energy_logger2.FindLog(diagram_context)


# Extract time data
t_time = log_x1.sample_times()

# Extract joint data for both robots
data_q1 = log_x1.data().transpose()[:, 0:9]   # Robot 1 joints
data_q2 = log_x2.data().transpose()[:, 0:9]   # Robot 2 joints
data_qdot1 = log_x1.data().transpose()[:, 9:18]  # Robot 1 velocities
data_qdot2 = log_x2.data().transpose()[:, 9:18]  # Robot 2 velocities

# Extract torques for both robots
data_tau1 = log_tau1.data().transpose()
data_tau2 = log_tau2.data().transpose()

# Extract ERG outputs for both robots
data_erg1 = log_erg1.data().transpose()
data_erg2 = log_erg2.data().transpose()

# Extract box position data
data_box_pos = log_box.data().transpose()

# Extract contact force data
data_contact_forces = log_contact.data().transpose()

# Extract pose data for both robots
data_pose = log_pose.data().transpose()

# Extract FCL distance data for both robots
data_fcl1 = log_fcl1.data().transpose()  # Robot 1 FCL distances (4 links)
data_fcl2 = log_fcl2.data().transpose()  # Robot 2 FCL distances (4 links)

# Extract energy data for both robots
data_energy1 = log_energy1.data().transpose()  # Robot 1 energy
data_energy2 = log_energy2.data().transpose()  # Robot 2 energy



print("\n=== Dual Robot Simulation Results ===")
print(f"Robot 1 final joint positions: {data_q1[-1, :]}")
print(f"Robot 2 final joint positions: {data_q2[-1, :]}")
print(f"Box final position: {data_box_pos[-1, 4:7]}")  # x, y, z position
print(f"Final contact forces - Robot 1: {data_contact_forces[-1, :3]}, Robot 2: {data_contact_forces[-1, 3:6]}")
print(f"Final FCL distances - Robot 1: {data_fcl1[-1, :]} (link7, hand, leftfinger, rightfinger)")
print(f"Final FCL distances - Robot 2: {data_fcl2[-1, :]} (link7, hand, leftfinger, rightfinger)")

# Check for torque limit violations
print("\n=== Torque Limit Violation Analysis ===")
# Panda robot torque limits (Nm) for 7 arm joints
torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# Check Robot 1 torque violations
max_torques_1 = np.max(np.abs(data_tau1[:, :7]), axis=0)  # Max absolute torque for each joint
violations_found_1 = False
for i in range(7):
    if max_torques_1[i] > torque_limits[i]:
        if not violations_found_1:
            print("Robot 1 Torque Violations:")
            violations_found_1 = True
        violation_percent = (max_torques_1[i] / torque_limits[i]) * 100
        print(f"  JOINT {i+1} VIOLATION: {max_torques_1[i]:.2f} Nm (limit: {torque_limits[i]:.1f} Nm, {violation_percent:.1f}% over limit)")

# Check Robot 2 torque violations
max_torques_2 = np.max(np.abs(data_tau2[:, :7]), axis=0)  # Max absolute torque for each joint
violations_found_2 = False
for i in range(7):
    if max_torques_2[i] > torque_limits[i]:
        if not violations_found_2:
            print("\nRobot 2 Torque Violations:")
            violations_found_2 = True
        violation_percent = (max_torques_2[i] / torque_limits[i]) * 100
        print(f"  JOINT {i+1} VIOLATION: {max_torques_2[i]:.2f} Nm (limit: {torque_limits[i]:.1f} Nm, {violation_percent:.1f}% over limit)")

# Check for any violations during simulation
violations_1 = np.any(np.abs(data_tau1[:, :7]) > torque_limits, axis=1)
violations_2 = np.any(np.abs(data_tau2[:, :7]) > torque_limits, axis=1)

if np.any(violations_1):
    violation_times_1 = t_time[violations_1]
    print(f"\nRobot 1 had torque violations at {len(violation_times_1)} time steps")
    print(f"First violation at t={violation_times_1[0]:.3f}s, Last violation at t={violation_times_1[-1]:.3f}s")

if np.any(violations_2):
    violation_times_2 = t_time[violations_2]
    print(f"\nRobot 2 had torque violations at {len(violation_times_2)} time steps")
    print(f"First violation at t={violation_times_2[0]:.3f}s, Last violation at t={violation_times_2[-1]:.3f}s")

if not violations_found_1 and not violations_found_2:
    print("No torque limit violations detected for either robot.")


# Create plots for both robots
plt.figure(figsize=(18, 12))

# Robot 1 joint positions
plt.subplot(3, 3, 1)
for i in range(9):  # All 9 joints (7 arm + 2 gripper)
    plt.plot(t_time, data_q1[:, i], label=f'Joint {i+1}')
plt.title('Robot 1 Joint Positions')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.legend()
plt.grid(True)

# Robot 2 joint positions
plt.subplot(3, 3, 2)
for i in range(9):  # All 9 joints (7 arm + 2 gripper)
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
for i in range(9):  # All 9 joints (7 arm + 2 gripper)
    plt.plot(t_time, data_tau1[:, i], label=f'Joint {i+1}')
plt.title('Robot 1 Torques')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

# Robot 2 torques
plt.subplot(3, 3, 5)
for i in range(9):  # All 9 joints (7 arm + 2 gripper)
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
plt.subplot(3, 3, 9)
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
plt.show()

# Create separate energy plots
plt.figure(figsize=(12, 8))

# Individual energy plots
plt.subplot(2, 2, 1)
plt.plot(t_time, data_energy1.flatten(), label='Robot 1 Energy', linewidth=2)
plt.title('Robot 1 ERG Energy')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t_time, data_energy2.flatten(), label='Robot 2 Energy', linewidth=2)
plt.title('Robot 2 ERG Energy')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.legend()
plt.grid(True)

# Combined energy plot
plt.subplot(2, 2, 3)
total_energy = data_energy1.flatten() + data_energy2.flatten()
plt.plot(t_time, total_energy, label='Total Energy', linewidth=2, color='red')
plt.title('Total System Energy')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy')
plt.legend()
plt.grid(True)

# Energy difference plot
plt.subplot(2, 2, 4)
energy_diff = data_energy1.flatten() - data_energy2.flatten()
plt.plot(t_time, energy_diff, label='Energy Difference (R1-R2)', linewidth=2, color='green')
plt.title('Energy Difference Between Robots')
plt.xlabel('Time (s)')
plt.ylabel('Energy Difference')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



print("Dual robot simulation and analysis completed!") 