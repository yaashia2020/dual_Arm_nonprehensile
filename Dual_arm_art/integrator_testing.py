import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Enable interactive plotting to keep windows open
plt.ion()  
from pydrake.all import *
# import pydot
from IPython.display import SVG, display
from scipy.spatial.transform import Rotation as R

# Import CompliantERG from trajectoryERG
from trajectoryERG import CompliantERG
from CERG_Setup import ErgParams, ContactParams, panda_spec

# Import Z-axis integrator
from z_axis_integrator import make_integrate_z_two_in_block

# Add Relaxed IK wrapper import
import sys
wrapper_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "submodules/relaxed_ik_core/wrappers")
sys.path.insert(0, wrapper_dir)
from python_wrapper import RelaxedIKRust

def create_system_model(plant, scene_graph):
    """
    Add the Panda arm model to the plant and configure contact properties.
    
    Args:
        plant: The MultibodyPlant object to which the Panda arm model will be added.
        scene_graph: The SceneGraph object for visualization.
    
    Returns:
        Tuple containing the updated plant and scene_graph.
    """
    urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_drake.urdf"))
    urdf = "file://" + urdf_path
    arm = Parser(plant).AddModelsFromUrl(urdf)
    
    # Add the fixed box to the scene (acting as static ground)
    fixed_box_sdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/boxes/fixed_box.sdf"))
    fixed_box_sdf_url = "file://" + fixed_box_sdf_path
    fixed_box = Parser(plant).AddModelsFromUrl(fixed_box_sdf_url)
    
    # Get the fixed box model instance ID
    fixed_box_id = plant.GetModelInstanceByName("fixed_box")
    
    # The fixed box is already static due to <static>true</static> in the SDF file
    # No need to weld it manually - Drake handles this automatically
    
    # Add the middle fixed box (between ground and movable box)
    middle_fixed_box_sdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/boxes/middle_fixed_box.sdf"))
    middle_fixed_box_sdf_url = "file://" + middle_fixed_box_sdf_path
    middle_fixed_box = Parser(plant).AddModelsFromUrl(middle_fixed_box_sdf_url)
    
    # Get the middle fixed box model instance ID
    middle_fixed_box_id = plant.GetModelInstanceByName("middle_fixed_box")
    
    # Add the movable box to the scene (on top of the middle fixed box)
    movable_box_sdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/boxes/movable_box.sdf"))
    movable_box_sdf_url = "file://" + movable_box_sdf_path
    movable_box = Parser(plant).AddModelsFromUrl(movable_box_sdf_url)
    
    # Get the movable box model instance ID
    movable_box_id = plant.GetModelInstanceByName("movable_box")
    
    # Add the back fixed box (behind the stacked boxes)
    back_fixed_box_sdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/boxes/back_fixed_box.sdf"))
    back_fixed_box_sdf_url = "file://" + back_fixed_box_sdf_path
    back_fixed_box = Parser(plant).AddModelsFromUrl(back_fixed_box_sdf_url)
    
    # Get the back fixed box model instance ID
    back_fixed_box_id = plant.GetModelInstanceByName("back_fixed_box")
    
    # Set the contact properties for interaction between the movable box and the ground
    plant.set_contact_surface_representation(mesh_type)  # Triangle or Polygon
    plant.set_contact_model(contact_model)  # Hydroelastic, Point, or HydroelasticWithFallback
    plant.set_discrete_contact_approximation(discrete_solver)
    plant.Finalize()
    
    # Position the movable box above the middle fixed box after plant is finalized
    # Middle fixed box: X=0.7, Z=0.25, height=0.4, so top surface is at Z=0.45
    # Movable box height=0.2, so center should be at Z=0.45 + 0.1 = 0.55
    plant_context = plant.CreateDefaultContext()
    initial_box_position = RigidTransform(p=[0.7, 0.0, 0.55])  # Positioned just above the middle fixed box
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("box_link", movable_box_id), initial_box_position)
    
    # Set default positions for the robot (not the box)
    panda_id = plant.GetModelInstanceByName("panda")
    
    # Debug: Check how many positions the robot actually has
    num_robot_positions = int(plant.num_positions(panda_id))
    # print(f"Robot has {num_robot_positions} positions")
    # print(f"Type of num_robot_positions: {type(num_robot_positions)}")
    
    # Debug: Print the IDs and box setup
    # print(f"Fixed box ID: {fixed_box_id}")
    # print(f"Middle fixed box ID: {middle_fixed_box_id}")
    # print(f"Movable box ID: {movable_box_id}")
    # print(f"Back fixed box ID: {back_fixed_box_id}")
    # print("Box setup:")
    # print("  - Fixed box (ground): Z=0.1")
    # print("  - Middle fixed box: X=0.7, Z=0.25, height=0.4m")
    # print("  - Movable box: X=0.7, Z=0.55, height=0.2m")
    # print("  - Back fixed box: X=1.0, Z=0.4, height=0.8m (behind in X-axis)")
    
    # Set appropriate number of default positions
    if num_robot_positions == 7:
        # 7-joint robot (arm only)
        plant.SetDefaultPositions(panda_id, [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    elif num_robot_positions == 9:
        # 9-joint robot (arm + gripper)
        plant.SetDefaultPositions(panda_id, [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0])
    else:
        # Unknown number of joints - use zeros
        # print(f"Warning: Unknown number of joints ({num_robot_positions}), using zeros")
        plant.SetDefaultPositions(panda_id, [0.0] * num_robot_positions)
    
    return plant, scene_graph, plant_context, num_robot_positions, fixed_box_id, middle_fixed_box_id, movable_box_id, back_fixed_box_id
####################################
# Configurations parameters
####################################
contact_model = ContactModel.kHydroelasticWithFallback  # Options: Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kTriangle  # Options: Triangle or Polygon
discrete_solver = DiscreteContactApproximation.kTamsi # Options:kTamsi, kSap, kLagged, kSimilar
realtime_factor = 1  # Real-time factor for simulation speed
time_step = 0.001

meshcat_visualisation = True
simulate = True

# Create system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
plant, scene_graph, plant_context, num_robot_positions, fixed_box_id, middle_fixed_box_id, movable_box_id, back_fixed_box_id = create_system_model(plant, scene_graph)
# Set the initial joint position of the robot otherwise it will correspond to zero positions
# plant_context = plant.CreateDefaultContext()  # This is now returned from create_system_model

# Get the panda model instance for specific connections
panda_id = plant.GetModelInstanceByName("panda")
num_positions = num_robot_positions
num_velocities = num_robot_positions

# Define initial joint configuration for robot initialization
# First 7 joints have specified values, additional joints (like gripper) are set to 0
base_joint_config = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
trajInit_ = np.zeros(num_robot_positions)
trajInit_[:7] = base_joint_config

# Get the movable box model instance for specific connections
movable_box_id = plant.GetModelInstanceByName("movable_box")

# Debug: Print movable box state structure
# print(f"\n=== Movable Box State Debug ===")
# print(f"Movable box model instance ID: {movable_box_id}")
# print(f"Number of positions: {plant.num_positions(movable_box_id)}")
# print(f"Number of velocities: {plant.num_velocities(movable_box_id)}")
# print(f"Total state size: {plant.num_positions(movable_box_id) +  plant.num_velocities(movable_box_id)}")

# # Get the actual state to see its structure
# box_state = plant.get_state_output_port(movable_box_id).Eval(plant_context)
# print(f"Actual state vector size: {len(box_state)}")
# print(f"State vector: {box_state}")

# Try to get pose information
try:
    box_body = plant.GetBodyByName("box_link", movable_box_id)
    box_pose = plant.EvalBodyPoseInWorld(plant_context, box_body)
    # print(f"Box pose translation: {box_pose.translation()}")
    # print(f"Box pose rotation (quaternion): {box_pose.rotation().ToQuaternion().wxyz()}")
except Exception as e:
    # print(f"Could not get box pose: {e}")
    pass

# print("=" * 40)

######################################################################################################
#              ##################Relaxed IK System for Box Tracking################
######################################################################################################
class RelaxedIKBoxTracker(LeafSystem):
    def __init__(self, plant, plant_context, num_joints):
        super().__init__()
        self.plant = plant
        self.plant_context = plant_context
        self.num_joints = num_joints

        # ===== timing & motion params =====
        self.t_orient     = 2.0    # s: Phase 1 — orientation-only
        self.t_lift_start = 6.0    # s: start lifting
        self.lift_amount  = 0.40   # m: lift after t_lift_start

        # box geometry (from movable_box.sdf: size [0.22, 0.30, 0.20])
        self.box_half_x   = 0.11   # 0.22 / 2
        self.pad_close    = 0.03   # m: approach pad during Phase 2/3
        self.standoff_far = 0.12   # m: initial hold distance during Phase 1

        # IK tolerances: [px, py, pz, rx, ry, rz]
        self.tol_default  = [0.01, 0.01, 0.01, 0.10, 0.10, 0.10]
        self.tol_orient   = [0.02, 0.02, 0.02, 0.03, 0.03, 0.03]  # tighter orientation in Phase 1

        # desired world-fixed EE orientation (palm toward +X): +90° about Y
        self.desired_quat_xyzw = R.from_euler("x", 90, degrees=True).as_quat().tolist()

        # cached "hold" position for Phase 1 (filled on first tick)
        self._hold_pos = None

        # ---- IO ----
        self._box_state_port = self.DeclareVectorInputPort(name="box_state", size=13)
        self._z_adjusted_target_port = self.DeclareVectorInputPort(name="z_adjusted_target", size=3)
        state_index = self.DeclareDiscreteState(num_joints)
        self.DeclareStateOutputPort("ik_joint_targets", state_index)

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01, offset_sec=0.0, update=self.solve_ik
        )

        # ---- RelaxedIK init ----
        try:
            config_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_rubber_settings.yaml")
            )
            self.rik = RelaxedIKRust(setting_file_path=config_path)
            self.ik_available = True
            # print("Relaxed IK initialized successfully for box tracking!")
        except Exception as e:
            # print(f"Failed to initialize Relaxed IK: {e}")
            self.ik_available = False

        # fallback joints
        if num_joints == 7:
            self.fallback_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        elif num_joints == 9:
            self.fallback_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]
        else:
            self.fallback_joints = [0.0] * num_joints
            # print(f"Warning: Using zero fallback joints for {num_joints} joints")

    def solve_ik(self, context, discrete_state):
        if not self.ik_available:
            discrete_state.get_mutable_vector().SetFromVector(self.fallback_joints)
            return

        try:
            t = context.get_time()
            box_state = self._box_state_port.Eval(context)
            x_b, y_b, z_b = box_state[4:7]

            # ===== Get current EE pose from plant context at t=0 =====
            if self._hold_pos is None:
                # Get the panda robot model instance directly
                try:
                    robot_id = self.plant.GetModelInstanceByName("panda")
                except Exception:
                    # Fallback: get first non-world model instance
                    robot_id = None
                    for i in range(self.plant.num_model_instances()):
                        mi = ModelInstanceIndex(i)
                        if mi != self.plant.world_frame().model_instance():
                            robot_id = mi
                            break
                    
                    if robot_id is None:
                        # print("Warning: Could not find robot model instance")
                        discrete_state.get_mutable_vector().SetFromVector(self.fallback_joints)
                        return

                # Find end-effector link name (from parsed RelaxedIK config if available)
                ee_name = None
                if hasattr(self, "config_info") and "ee_links" in self.config_info and len(self.config_info["ee_links"]) > 0:
                    ee_name = self.config_info["ee_links"][0]
                else:
                    ee_name = "rubber_pad"  # default

                ee_body = self.plant.GetBodyByName(ee_name, robot_id)
                ee_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, ee_body)
                self._hold_pos = ee_pose.translation().tolist()

                # print(f"[Init] Using actual EE start pos as hold position: {self._hold_pos}")

            # ===== Get z-adjusted target from z_integrator =====
            # Get integrated Z offset (default to 0 if not available)
            z_integrated_offset = 0.0
            try:
                if self._z_adjusted_target_port.HasValue(context):
                    z_integrator_output = self._z_adjusted_target_port.Eval(context)
                    z_integrated_offset = z_integrator_output[2]  # The integrated Z offset
            except:
                pass  # Use default 0.0
            
            # ===== Phase selection =====
            if t <= self.t_orient:
                # Phase 1: orientation-only — hold position fixed, aim for desired orientation
                target_position  = self._hold_pos
                target_quat_xyzw = self.desired_quat_xyzw
                tol              = self.tol_orient
            else:
                # Phase 2/3: face-tracking (XY) + optional Z lift after t_lift_start
                # Compute base target Z
                z_target_base = z_b if t < self.t_lift_start else (z_b + self.lift_amount)
                
                # Add integrated offset from z_integrator
                z_target = z_target_base + z_integrated_offset
                
                x_face   = x_b - (self.box_half_x + self.pad_close)
                target_position  = [x_face, y_b, z_target]
                target_quat_xyzw = self.desired_quat_xyzw
                tol              = self.tol_default

            # ---- Debug ----
            phase = (
                "orient-only" if t <= self.t_orient
                else ("approach" if t < self.t_lift_start else "lift")
            )
            # print(f"\n=== RelaxedIK (t={t:.2f}) Phase: {phase} ===")
            # print(f"Box: [{x_b:.3f}, {y_b:.3f}, {z_b:.3f}]  Target: {target_position}")
            # print(f"Quat(xyzw): {target_quat_xyzw}")

            # ---- Solve IK ----
            joint_solution = self.rik.solve_position(target_position, target_quat_xyzw, tol)

            # ---- Pad to full joint count ----
            if len(joint_solution) == self.num_joints:
                full_solution = joint_solution
            elif len(joint_solution) == 7 and self.num_joints == 9:
                full_solution = list(joint_solution) + [0.0, 0.0]
            else:
                # print(f"Warning: Solution length {len(joint_solution)} != expected {self.num_joints}, using fallback")
                full_solution = self.fallback_joints

            discrete_state.get_mutable_vector().SetFromVector(full_solution)

        except Exception as e:
            # print(f"IK solving failed: {e}")
            discrete_state.get_mutable_vector().SetFromVector(self.fallback_joints)

######################################################################################################
#              ##################Planner Trapezoidal motion profile ################
######################################################################################################
# DELETED - Not using motion profile, using RelaxedIK instead



######################################################################################################
#                       ################## Trajectory-Based ERG ################
######################################################################################################
class ERG(LeafSystem):
    def __init__(self, num_joints):
        super().__init__()  # Don't forget to initialize the base class.
        
        self.num_joints = num_joints
        self.state_size = num_joints * 2  # positions + velocities
        
        self._state_port = self.DeclareVectorInputPort(name="state", size=self.state_size)
        self._tau_port = self.DeclareVectorInputPort(name="tau", size=num_joints)
        self._qr_port = self.DeclareVectorInputPort(name="q_r", size=num_joints)
        self._box_state_port = self.DeclareVectorInputPort(name="box_state", size=13)

        state_index = self.DeclareDiscreteState(num_joints)
        self.DeclareStateOutputPort("q_v_filtered", state_index)  # One output: y=x.
        
        # Add output port for calculated energy
        self.DeclareVectorOutputPort("calculated_energy", size=1, calc=self.output_energy)
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.refrence) # Call the Update method defined below.
        # Use the same URDF as the main plant
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
        q = state[:self.num_joints]
        dq = state[self.num_joints:]
        tau = self._tau_port.Eval(context)
        q_r = self._qr_port.Eval(context)
        box_state = self._box_state_port.Eval(context)
        # Extract and post-process box position (subtract 0.11 from x)
        box_position = np.array(box_state[4:7], dtype=float)
        box_position[0] -= 0.11

        self.q_v = context.get_discrete_state_vector().CopyToVector()              
        # Initialize q_v_ only at the first callback
        if self.first_update:
            self.q_v_ = trajInit_  # Use all 9 joints    
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
#                                  ########PD+G controller#######   Check input output
######################################################################################################
class PD_gravity(LeafSystem):
    def __init__(self, num_joints):
        super().__init__()
        
        self.num_joints = num_joints
        self.state_size = num_joints * 2  # positions + velocities
        
        # Declare input ports with dynamic sizes
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=num_joints)
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=self.state_size)
        
        # Set gains based on number of joints
        if num_joints == 7:
            # 7-joint robot (arm only)
            self.Kp_ = [120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0]
            self.Kd_ = [8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0]
        elif num_joints == 9:
            # 9-joint robot (arm + gripper)
            self.Kp_ = [120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0, 120, 120]
            self.Kd_ = [8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0, 5, 5]
        else:
            # Default gains for unknown number of joints
            self.Kp_ = [100.0] * num_joints
            self.Kd_ = [5.0] * num_joints
            # print(f"Warning: Using default gains for {num_joints} joints")

        # Declare discrete state and output with dynamic size
        state_index = self.DeclareDiscreteState(num_joints)
        self.DeclareStateOutputPort("tau_u", state_index)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/1000,  # One second time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.compute_tau_u) # Call the Update method defined below.
        
    def compute_tau_u(self, context, discrete_state):
        # Evaluate the input ports
        self.q_d = self._desired_state_port.Eval(context)  # num_joints
        self.q = self._current_state_port.Eval(context)    # 2*num_joints states (pos + vel)
        
        # Get gravity for entire plant and extract robot portion
        gravity_full = -plant.CalcGravityGeneralizedForces(plant_context)
        # Convert to numpy array and extract first num_joints elements for robot
        gravity_full_np = np.array(gravity_full).flatten()
        gravity = gravity_full_np[:self.num_joints]  # Extract correct number of elements
        
        # Calculate torque for all joints
        tau = self.Kp_ * (self.q_d - self.q[:self.num_joints]) - self.Kd_ * self.q[self.num_joints:]
        tau = tau + gravity  # Add gravity for all joints
        
        discrete_state.get_mutable_vector().SetFromVector(tau)

######################################################################################################
#                                  ########Contact Force Converter#######
######################################################################################################
class ContactForceConverter(LeafSystem):
    def __init__(self, plant):
        super().__init__()
        self.plant = plant
        
        # Input port for contact results
        self.DeclareAbstractInputPort("contact_results", 
                                    plant.get_contact_results_output_port().Allocate())
        
        # Output port for contact forces (3D vector)
        self.DeclareVectorOutputPort("contact_forces", size=3, calc=self.calc_output)
    
    def calc_output(self, context, output):
        # Get contact results from input port
        contact_results = self.GetInputPort("contact_results").Eval(context)
        
        # Initialize output to zero
        output.SetFromVector([0.0, 0.0, 0.0])
        
        # Check if there are any contacts
        if contact_results.num_point_pair_contacts() == 0:
            # print("No contacts detected")
            return
        
        # Get number of contacts
        num_contacts = contact_results.num_point_pair_contacts()
        # print(f"Total contacts detected: {num_contacts}")
        
        # Define robot links to check for contacts - including all hand components
        robot_links = [
            "panda_link1", "panda_link2", "panda_link3", "panda_link4", 
            "panda_link5", "panda_link6", "panda_link7", 
            "panda_hand", "panda_finger_joint1", "panda_finger_joint2",
            "panda_leftfinger", "panda_rightfinger"
        ]
        object_name = "box_link"
        
        # Sum up all contact forces between robot and box
        total_force = [0.0, 0.0, 0.0]
        robot_box_contacts = 0
        
        # Track contacts by component type
        arm_contacts = 0
        hand_contacts = 0
        finger_contacts = 0
        
        for i in range(num_contacts):
            contact_info = contact_results.point_pair_contact_info(i)
            
            bodyA_idx = contact_info.bodyA_index()
            bodyB_idx = contact_info.bodyB_index()
            
            bodyA_name = self.plant.get_body(bodyA_idx).name()
            bodyB_name = self.plant.get_body(bodyB_idx).name()
            
            # print(f"Contact {i}: {bodyA_name} <-> {bodyB_name}")
            
            # Check if this contact is between any robot link and box
            if ((bodyA_name == object_name or bodyB_name == object_name) and
                (bodyA_name in robot_links or bodyB_name in robot_links)):
                
                robot_box_contacts += 1
                force = contact_info.contact_force()
                # print(f"  Robot-Box contact detected! Force: [{force[0]:.3f}, {force[1]:.3f}, {force[2]:.3f}]")
                
                # Categorize contact type
                contact_body = bodyA_name if bodyA_name in robot_links else bodyB_name
                if "finger" in contact_body:
                    finger_contacts += 1
                    # print(f"    -> Finger contact detected on {contact_body}")
                elif contact_body == "panda_hand":
                    hand_contacts += 1
                    # print(f"    -> Hand contact detected on {contact_body}")
                else:
                    arm_contacts += 1
                    # print(f"    -> Arm link contact detected on {contact_body}")
                
                # Extract contact force and add to total
                total_force[0] += force[0]
                total_force[1] += force[1]
                total_force[2] += force[2]
        
        # print(f"Robot-Box contacts: {robot_box_contacts}")
        # print(f"  - Arm link contacts: {arm_contacts}")
        # print(f"  - Hand contacts: {hand_contacts}")
        # print(f"  - Finger contacts: {finger_contacts}")
        # print(f"Total force: [{total_force[0]:.3f}, {total_force[1]:.3f}, {total_force[2]:.3f}]")
        
        # Set the total contact force
        output.SetFromVector(total_force)

######################################################################################################
#                                  ##################################
######################################################################################################
######################################################################################################
#                                  ########Box Position Extractor#######
######################################################################################################
# Get the box model instance ID for the position extractor
box_id = plant.GetModelInstanceByName("movable_box")

# Create systems
erg_system = builder.AddNamedSystem("Trajectory-based ERG", ERG(num_robot_positions))
pid_controller = builder.AddNamedSystem("PD+G controller", PD_gravity(num_robot_positions))

# Add new systems for IK-based control
# Remove the custom box position extractor and use plant state output directly
ik_box_tracker = builder.AddNamedSystem("Relaxed IK Box Tracker", RelaxedIKBoxTracker(plant, plant_context, num_robot_positions))

# Add contact force converter system
contact_force_converter = builder.AddNamedSystem("Contact Force Converter", ContactForceConverter(plant))

# Connect plant state output to IK system (box state will be extracted inside the system)
builder.Connect(plant.get_state_output_port(box_id), 
               ik_box_tracker.GetInputPort("box_state"))

# Connect plant state output to ERG system (box state will be extracted inside the system)
builder.Connect(plant.get_state_output_port(box_id), 
               erg_system.GetInputPort("box_state"))

# Connect contact results to contact force converter
builder.Connect(plant.get_contact_results_output_port(),
               contact_force_converter.GetInputPort("contact_results"))

# Connect IK system to ERG (replacing the trajectory)
builder.Connect(ik_box_tracker.GetOutputPort("ik_joint_targets"), 
               erg_system.GetInputPort("q_r"))

# Keep the original connections for the rest of the system
builder.Connect(plant.get_state_output_port(panda_id), pid_controller.GetInputPort("Current_state"))
builder.Connect(erg_system.GetOutputPort("q_v_filtered"), pid_controller.GetInputPort("Desired_state"))
builder.Connect(pid_controller.GetOutputPort("tau_u"), plant.get_actuation_input_port(panda_id))
builder.Connect(plant.get_state_output_port(panda_id), erg_system.GetInputPort("state"))
builder.Connect(plant.GetOutputPort("panda_net_actuation"), erg_system.GetInputPort("tau"))

# Connect to visualizer
if meshcat_visualisation:
    meshcat = StartMeshcat()
    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    # print(f"MeshCat visualization available at: {meshcat.web_url()}")

logger_x = LogVectorOutput(plant.get_state_output_port(), builder) #state
logger_tau = LogVectorOutput(pid_controller.GetOutputPort("tau_u"), builder) #tau_u
logger_qv = LogVectorOutput(erg_system.GetOutputPort("q_v_filtered"), builder) #q_v
logger_qr = LogVectorOutput(ik_box_tracker.GetOutputPort("ik_joint_targets"), builder) #ik targets
logger_box_pos = LogVectorOutput(plant.get_state_output_port(box_id), builder) #box position

# Add loggers for energy and contact forces
logger_energy = LogVectorOutput(erg_system.GetOutputPort("calculated_energy"), builder) #calculated energy from ERG
logger_contact_forces = LogVectorOutput(contact_force_converter.GetOutputPort("contact_forces"), builder) #contact forces between robot and box

# Create a LeafSystem to extract panda_link7 world positions from joint states
class PandaLink7PoseExtractor(LeafSystem):
    def __init__(self, plant, panda_id, num_joints):
        super().__init__()
        self.plant = plant
        self.panda_id = panda_id
        self.num_joints = num_joints
        
        # Input port for joint positions (dynamic based on num_joints)
        # The plant state output contains: [q1, q2, ..., qn, dq1, dq2, ..., dqn]
        # So total size is num_joints * 2
        self.DeclareVectorInputPort("joint_positions", size=num_joints * 2)  # positions + velocities
        
        # Output port for panda_link7 world positions (X, Y, Z)
        self.DeclareVectorOutputPort("panda_link7_world_positions", size=3, calc=self.CalcOutput)
        
        # Create a temporary context for pose evaluation
        self.temp_context = plant.CreateDefaultContext()
    
    def CalcOutput(self, context, output):
        # Get joint positions from input port (dynamic state: num_joints positions + num_joints velocities)
        full_state = self.GetInputPort("joint_positions").Eval(context)
        
        # Debug: Print state information
        # print(f"Full state size: {len(full_state)}")
        # print(f"Expected num_joints: {self.num_joints}")
        # print(f"Plant num_positions for panda: {self.plant.num_positions(self.panda_id)}")
        
        # Extract only joint positions (first num_joints elements)
        # The state vector is [q1, q2, ..., qn, dq1, dq2, ..., dqn]
        joint_positions = full_state[:self.num_joints]
        
        # print(f"Extracted joint_positions size: {len(joint_positions)}")
        # print(f"Joint positions: {joint_positions}")
        
        # Set the plant to these joint positions in temporary context
        # Use the correct model instance and context
        try:
            self.plant.SetPositions(self.temp_context, self.panda_id, joint_positions)
            # print("SetPositions successful!")
        except Exception as e:
            # print(f"SetPositions failed: {e}")
            # Fallback: try without model instance
            try:
                self.plant.SetPositions(self.temp_context, joint_positions)
                # print("SetPositions successful (without model instance)!")
            except Exception as e2:
                # print(f"SetPositions failed even without model instance: {e2}")
                return
        
        # Get panda_link7 body
        panda_link7_body = self.plant.GetBodyByName("panda_link7")
        
        # Evaluate panda_link7 world pose using EvalBodyPoseInWorld
        panda_link7_pose = self.plant.EvalBodyPoseInWorld(self.temp_context, panda_link7_body)
        
        # Extract translation (X, Y, Z positions)
        translation = panda_link7_pose.translation()
        
        # Set output to world positions
        output.SetFromVector([translation[0], translation[1], translation[2]])

# Add PandaLink7PoseExtractor to the diagram
panda_link7_extractor = builder.AddNamedSystem("PandaLink7PoseExtractor", 
                                              PandaLink7PoseExtractor(plant, panda_id, num_robot_positions))

# Debug: Check plant state output port size
panda_state_port = plant.get_state_output_port(panda_id)
# print(f"Panda state output port size: {panda_state_port.size()}")
# print(f"Expected input port size: {num_robot_positions * 2}")
# print(f"Plant num_positions for panda: {plant.num_positions(panda_id)}")
# print(f"Plant num_velocities for panda: {plant.num_velocities(panda_id)}")
# print(f"Our num_robot_positions: {num_robot_positions}")

# Check if the input port size matches
extractor_input_port = panda_link7_extractor.GetInputPort("joint_positions")
# print(f"Extractor input port size: {extractor_input_port.size()}")
# print(f"Port sizes match: {panda_state_port.size() == extractor_input_port.size()}")

# Connect robot joint states to PandaLink7PoseExtractor
builder.Connect(plant.get_state_output_port(panda_id), 
               panda_link7_extractor.GetInputPort("joint_positions"))

# Add logger for panda_link7 world positions from extractor
logger_panda_link7_world = LogVectorOutput(panda_link7_extractor.GetOutputPort("panda_link7_world_positions"), builder)

# Add binary contact detector using Drake's built-in contact detection
class ContactDetectorForIntegrator(LeafSystem):
    """
    Detects contact between robot end-effector and movable box using Drake's contact results.
    Outputs binary contact: 1.0 if contact exists, 0.0 otherwise.
    """
    def __init__(self, plant, movable_box_id):
        super().__init__()
        self.plant = plant
        self.movable_box_id = movable_box_id
        
        # Input: contact results from plant
        self.DeclareAbstractInputPort("contact_results", 
                                       plant.get_contact_results_output_port().Allocate())
        # Output: binary contact value (0.0 or 1.0)
        self.DeclareVectorOutputPort("contact", size=1, calc=self.CalcOutput)
    
    def CalcOutput(self, context, output):
        # Get contact results from plant
        contact_results = self.GetInputPort("contact_results").Eval(context)
        
        # Check for contacts
        num_contacts = contact_results.num_point_pair_contacts()
        
        # Get current time
        t = context.get_time()
        
        # Get box body indices
        box_body_indices = []
        for body_idx in range(self.plant.num_bodies()):
            body = self.plant.get_body(BodyIndex(body_idx))
            if body.model_instance() == self.movable_box_id:
                box_body_indices.append(body_idx)
        
        # Get robot end-effector body indices - rubber pad and panda_hand
        robot_body_indices = []
        
        # Find rubber pad
        try:
            body = self.plant.GetBodyByName("rubber_pad")
            robot_body_indices.append(body.index())
            # Only print once
            if not hasattr(self, '_printed_rubber_pad'):
                print(f"Found rubber_pad at body index {body.index()}")
                self._printed_rubber_pad = True
        except:
            if not hasattr(self, '_printed_warning'):
                print("Warning: rubber_pad not found in plant!")
                self._printed_warning = True
        
        # Find panda_hand
        try:
            body = self.plant.GetBodyByName("panda_hand")
            robot_body_indices.append(body.index())
            # Only print once
            if not hasattr(self, '_printed_panda_hand'):
                print(f"Found panda_hand at body index {body.index()}")
                self._printed_panda_hand = True
        except:
            if not hasattr(self, '_printed_warning_hand'):
                print("Warning: panda_hand not found in plant!")
                self._printed_warning_hand = True
        
        # If neither found, print available bodies
        if len(robot_body_indices) == 0 and not hasattr(self, '_printed_available_bodies'):
            print("Available robot bodies:")
            panda_id = self.plant.GetModelInstanceByName("panda")
            for i in range(self.plant.num_bodies()):
                body = self.plant.get_body(BodyIndex(i))
                if body.model_instance() == panda_id:
                    print(f"  - {body.name()} (index {i})")
            self._printed_available_bodies = True
        
        # Log all contact information
        for i in range(num_contacts):
            contact_info = contact_results.point_pair_contact_info(i)
            bodyA_idx = contact_info.bodyA_index()
            bodyB_idx = contact_info.bodyB_index()
            
            bodyA_name = self.plant.get_body(BodyIndex(bodyA_idx)).name()
            bodyB_name = self.plant.get_body(BodyIndex(bodyB_idx)).name()
            
            # Also get model instance for debugging
            bodyA_model = self.plant.get_body(BodyIndex(bodyA_idx)).model_instance()
            bodyB_model = self.plant.get_body(BodyIndex(bodyB_idx)).model_instance()
            
            # Determine contact type
            is_robot_A = bodyA_idx in robot_body_indices
            is_robot_B = bodyB_idx in robot_body_indices
            is_box_A = bodyA_idx in box_body_indices
            is_box_B = bodyB_idx in box_body_indices
            
            contact_type = "unknown"
            if is_robot_A and is_robot_B:
                contact_type = "robot-self"
            elif is_box_A and is_box_B:
                contact_type = "box-box"
            elif (is_robot_A and is_box_B) or (is_robot_B and is_box_A):
                contact_type = "robot-box"
            else:
                contact_type = "other"
            
            # Get contact force
            force = contact_info.contact_force()
            force_magnitude = np.linalg.norm([force[0], force[1], force[2]])
            
            # Log this contact
            if not hasattr(self, '_contact_log'):
                self._contact_log = []
            
            self._contact_log.append({
                'time': t,
                'bodyA': bodyA_name,
                'bodyB': bodyB_name,
                'bodyA_idx': bodyA_idx,
                'bodyB_idx': bodyB_idx,
                'bodyA_model': str(bodyA_model),
                'bodyB_model': str(bodyB_model),
                'type': contact_type,
                'force_x': force[0],
                'force_y': force[1],
                'force_z': force[2],
                'force_magnitude': force_magnitude
            })
        
        if num_contacts == 0:
            output.set_value([0.0])
            return
        
        # Check if any contact is between robot (rubber_pad) and box ONLY
        # Reject: self-contacts (robot-robot) and box-to-box contacts
        contact_detected = False
        for i in range(num_contacts):
            contact_info = contact_results.point_pair_contact_info(i)
            bodyA_idx = contact_info.bodyA_index()
            bodyB_idx = contact_info.bodyB_index()
            
            # Check if contact involves both robot and box
            is_robot_A = bodyA_idx in robot_body_indices
            is_robot_B = bodyB_idx in robot_body_indices
            is_box_A = bodyA_idx in box_body_indices
            is_box_B = bodyB_idx in box_body_indices
            
            # Skip self-contacts (robot-robot)
            if is_robot_A and is_robot_B:
                continue
            
            # Skip box-to-box contacts
            if is_box_A and is_box_B:
                continue
            
            # Only accept robot (rubber_pad) contacting box
            if (is_robot_A and is_box_B) or (is_robot_B and is_box_A):
                contact_detected = True
                # Get body names for debugging
                bodyA_name = self.plant.get_body(BodyIndex(bodyA_idx)).name()
                bodyB_name = self.plant.get_body(BodyIndex(bodyB_idx)).name()
                print(f"Contact detected: {bodyA_name} <-> {bodyB_name}")
                break
        
        contact_value = 1.0 if contact_detected else 0.0
        output.set_value([contact_value])

# Create contact detector - binary detection using Drake contact results
# Only detects contact between rubber_pad and movable box
contact_detector = builder.AddSystem(ContactDetectorForIntegrator(plant, movable_box_id))
builder.Connect(plant.get_contact_results_output_port(),
                contact_detector.GetInputPort("contact_results"))

# Add Z-axis integrator system with two inputs
z_integrator = builder.AddNamedSystem("ZAxisIntegrator", 
                                     make_integrate_z_two_in_block(
                                         Ki_z=0.7,  # Integral gain for Z (reduced for stability)
                                         z_min=-0.5,  # Minimum Z limit
                                         z_max=0.2,   # Maximum Z limit
                                         Kaw_z=0.1,   # Anti-windup gain
                                         error_mode="a_minus_b",  # a.z - b.z
                                         passthrough_xy_from="a",  # Use X,Y from first input
                                         name="ZAxisIntegrator"
                                     ))

# First input: RelaxedIK target position (where we want the box to be)
# We need to extract target Z from RelaxedIK
# For now, use box Z + lift_amount as target (will be refined)
class TargetPositionExtractor(LeafSystem):
    def __init__(self, t_lift_start, lift_amount):
        super().__init__()
        self.t_lift_start = t_lift_start
        self.lift_amount = lift_amount
        
        # Input: box state
        self.DeclareVectorInputPort("box_state", size=13)
        # Output: target 3D position
        self.DeclareVectorOutputPort("target_position", size=3, calc=self.CalcOutput)
    
    def CalcOutput(self, context, output):
        # Get box state
        box_state = self.GetInputPort("box_state").Eval(context)
        
        # We need to get current time to calculate target
        t = context.get_time()
        
        # Extract box position (indices 4:7 in 13-state vector)
        x_b, y_b, z_b = box_state[4:7]
        
        # Calculate target Z (same logic as RelaxedIK)
        z_target = z_b if t < self.t_lift_start else (z_b + self.lift_amount)
        
        # Target position
        target_pos = np.array([x_b, y_b, z_target])
        output.set_value(target_pos)

# Create target position extractor
target_extractor = builder.AddSystem(TargetPositionExtractor(
    t_lift_start=6.0,
    lift_amount=0.40
))
# Connect box state to target extractor (same as IK tracker)
builder.Connect(plant.get_state_output_port(box_id),
                target_extractor.GetInputPort("box_state"))

# First input: target Z position
builder.Connect(target_extractor.GetOutputPort("target_position"),
               z_integrator.GetInputPort("a"))

# Second input: movable box world positions (extract position components from 13-vector state)
class BoxPositionExtractor(LeafSystem):
    def __init__(self):
        super().__init__()
        # Input port for 13-element box state
        self.DeclareVectorInputPort("box_state", size=13)
        # Output port for 3-element position (X, Y, Z)
        self.DeclareVectorOutputPort("box_position", size=3, calc=self.CalcOutput)
    
    def CalcOutput(self, context, output):
        # Get the full box state
        box_state = self.GetInputPort("box_state").Eval(context)
        # Extract position (indices 4:7) - same as in test_erg.py
        position = box_state[4:7]
        output.SetFromVector(position)

box_position_extractor = builder.AddSystem(BoxPositionExtractor())
builder.Connect(plant.get_state_output_port(movable_box_id), 
               box_position_extractor.GetInputPort("box_state"))
builder.Connect(box_position_extractor.GetOutputPort("box_position"), 
               z_integrator.GetInputPort("b"))

# Connect contact detector to z_integrator
builder.Connect(contact_detector.GetOutputPort("contact"),
               z_integrator.GetInputPort("contact"))

# Connect z_integrator output to IK system (for z-adjusted target with integrated offset)
builder.Connect(z_integrator.GetOutputPort("u"),
               ik_box_tracker.GetInputPort("z_adjusted_target"))

# Add loggers for Z-axis integrator output and contact flag
logger_z_integrated = LogVectorOutput(z_integrator.GetOutputPort("u"), builder)
logger_contact_flag = LogVectorOutput(contact_detector.GetOutputPort("contact"), builder)

# Finalize the diagram
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

####################################
# Run Simple Simulation
####################################
if simulate:
    simulator = Simulator(diagram, diagram_context)
    # print(f"Initial positions: {trajInit_}")
    plant.SetPositions(plant_context, panda_id, trajInit_)
    simulator.set_target_realtime_rate(realtime_factor)
    simulator.set_publish_every_time_step(True)
    simulator.Initialize()

    # Define the step size and simulation time
    kStep = 0.001
    sim_time = 10.0  # Fixed simulation time since no trajectory duration
    simulator_context = simulator.get_mutable_context()

    # print(f"Starting simulation for {sim_time} seconds...")
    # print(f"Initial` positions: {trajInit_}")
    # print("Using Relaxed IK to track box position!")

    # Run simulation
    while simulator_context.get_time() < sim_time:
         next_time = min(sim_time, simulator_context.get_time() + kStep)
         simulator.AdvanceTo(next_time)

    # print("Simulation completed!")
    
    # Evaluate and print the final pose of the movable box

    
    # Record and publish MeshCat visualization
    if meshcat_visualisation:
        # print("Recording simulation for MeshCat replay...")
        meshcat.StartRecording()
        simulator.AdvanceTo(sim_time)  # Adjust this time as needed
        meshcat.PublishRecording()
        # print("MeshCat recording published!")
        
        # Save HTML recording
        html_path = os.path.join(os.path.dirname(__file__), "meshcat_recording_erg.html")
        html_data = meshcat.StaticHtml()
        # print("Recording size (characters):", len(html_data))
        
        if len(html_data) > 0:
            with open(html_path, "w") as f:
                f.write(html_data)
            # print(f"Recording saved to: {html_path}")
        else:
            # print("MeshCat recording appears to be empty. Did any geometry move?")
            pass

# Save contact log to file
if hasattr(contact_detector, '_contact_log') and len(contact_detector._contact_log) > 0:
    contact_log_path = os.path.join(os.path.dirname(__file__), "contact_log.csv")
    print(f"\nSaving contact log to: {contact_log_path}")
    
    import csv
    with open(contact_log_path, 'w', newline='') as csvfile:
        fieldnames = ['time', 'bodyA', 'bodyB', 'bodyA_idx', 'bodyB_idx', 'bodyA_model', 'bodyB_model', 'type', 'force_x', 'force_y', 'force_z', 'force_magnitude']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for contact in contact_detector._contact_log:
            writer.writerow(contact)
    
    print(f"Saved {len(contact_detector._contact_log)} contact events")
    
    # Print summary
    contact_types = {}
    for contact in contact_detector._contact_log:
        ct = contact['type']
        contact_types[ct] = contact_types.get(ct, 0) + 1
    
    print("\nContact type summary:")
    for ctype, count in contact_types.items():
        print(f"  {ctype}: {count}")
else:
    print("\nNo contacts logged")

# Assuming you have the following limits defined somewhere in your script
limit_q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
limit_q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
limit_dq = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
limit_tau = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

log_x = logger_x.FindLog(diagram_context)
log_tau = logger_tau.FindLog(diagram_context)
log_qv = logger_qv.FindLog(diagram_context)
log_qr = logger_qr.FindLog(diagram_context)
log_box_pos = logger_box_pos.FindLog(diagram_context)
log_energy = logger_energy.FindLog(diagram_context)
log_contact_forces = logger_contact_forces.FindLog(diagram_context)
log_panda_link7_world = logger_panda_link7_world.FindLog(diagram_context)
log_z_integrated = logger_z_integrated.FindLog(diagram_context)
log_contact_flag = logger_contact_flag.FindLog(diagram_context)

t_time = log_x.sample_times()

# Debug: Check plant state structure
# print(f"\n=== Plant State Structure Debug ===")
# print(f"Total plant state size: {log_x.data().shape}")
# print(f"Robot positions: {num_robot_positions}")
# print(f"Robot velocities: {num_robot_positions}")
# print(f"Box positions: {plant.num_positions(movable_box_id)}")
# print(f"Box velocities: {plant.num_velocities(movable_box_id)}")
# print(f"Expected total state size: {num_robot_positions * 2 + plant.num_positions(movable_box_id) + plant.num_velocities(movable_box_id)}")

# Extract only robot joint positions and velocities from the full plant state
# Plant state structure: [robot_positions, robot_velocities, box_positions, box_velocities]
robot_positions_start = 0
robot_positions_end = num_robot_positions
robot_velocities_start = num_robot_positions
robot_velocities_end = num_robot_positions * 2

data_q = log_x.data().transpose()[:, robot_positions_start:robot_positions_end]  # Robot joint positions only
data_qdot = log_x.data().transpose()[:, robot_velocities_start:robot_velocities_end]  # Robot joint velocities only
# print(f"Extracted data_q shape: {data_q.shape}")
# print(f"Extracted data_qdot shape: {data_qdot.shape}")
# print("=" * 50)
data_tau = log_tau.data().transpose()  # Selecting only the first 7 columns (joints)
data_qv = log_qv.data().transpose()  # Selecting only the first 7 columns (joints)
data_qr = log_qr.data().transpose()  # Selecting only the first 7 columns (joints)
data_box_pos = log_box_pos.data().transpose() # box position
data_energy = log_energy.data().transpose() # system energy
data_contact_forces = log_contact_forces.data().transpose() # contact forces
data_panda_link7_world = log_panda_link7_world.data().transpose() # panda_link7 world positions
data_z_integrated = log_z_integrated.data().transpose() # Z-axis integrator output
data_contact_flag = log_contact_flag.data().transpose() # contact flag (1 when contact, 0 when no contact)

# Debug: Print contact detection stats
print(f"\n=== Contact Detection Debug ===")
print(f"Contact flag shape: {data_contact_flag.shape}")
print(f"Contact flag range: [{np.min(data_contact_flag):.2f}, {np.max(data_contact_flag):.2f}]")
print(f"Fraction of time with contact: {np.sum(data_contact_flag > 0.5) / len(data_contact_flag) * 100:.1f}%")
print(f"Integrated Z range: [{np.min(data_z_integrated[:, 2]):.4f}, {np.max(data_z_integrated[:, 2]):.4f}]")
print("=" * 50)

# Extract panda_link7 positions using the new PoseExtractor system
# print("Extracting panda_link7 positions using PandaLink7PoseExtractor...")
# print(f"Panda link7 world positions data shape: {data_panda_link7_world.shape}")

# The new system directly gives us X, Y, Z positions
# data_panda_link7_world shape: [time_steps, 3] where 3 = [X, Y, Z]
panda_link7_positions = data_panda_link7_world

# print(f"panda_link7 positions shape: {panda_link7_positions.shape}")
# print(f"Sample panda_link7 positions: {panda_link7_positions[:5]}")

# Print the last values of joint positions from logger data
# print(f"\n=== Final Joint Positions ===")
# print(f"Joint positions at end of simulation:")
# print(f"Final joint positions: {data_q[-1, :]}")
# print("=" * 30)

# Get body poses at the end of simulation using the last joint positions from logger
# print(f"\n=== Final Body Poses (using logged joint positions) ===")
try:
    # Set the plant to the final joint positions from logger
    final_joint_positions = data_q[-1, :]
    
    # Debug: Check vector sizes
    # print(f"data_q shape: {data_q.shape}")
    # print(f"final_joint_positions length: {len(final_joint_positions)}")
    # print(f"Plant num_positions for panda: {plant.num_positions(panda_id)}")
    # print(f"Expected num_robot_positions: {num_robot_positions}")
    # print(f"final_joint_positions: {final_joint_positions}")
    
    # Ensure we have the right number of positions
    if len(final_joint_positions) != plant.num_positions(panda_id):
        # print(f"Warning: Position vector length mismatch!")
        # print(f"  Logger data length: {len(final_joint_positions)}")
        # print(f"  Plant expects: {plant.num_positions(panda_id)}")
        
        # Truncate or pad as needed
        if len(final_joint_positions) > plant.num_positions(panda_id):
            final_joint_positions = final_joint_positions[:plant.num_positions(panda_id)]
            # print(f"  Truncated to: {final_joint_positions}")
        else:
            # Pad with zeros
            padding = [0.0] * (plant.num_positions(panda_id) - len(final_joint_positions))
            final_joint_positions = np.concatenate([final_joint_positions, padding])
            # print(f"  Padded to: {final_joint_positions}")
    
    plant.SetPositions(plant_context, panda_id, final_joint_positions)
    
    # Get panda_link7 pose
    panda_link7_body = plant.GetBodyByName("panda_link7")
    panda_link7_pose = plant.EvalBodyPoseInWorld(plant_context, panda_link7_body)
    panda_link7_translation = panda_link7_pose.translation()
    
    # Get panda_hand pose
    panda_hand_body = plant.GetBodyByName("panda_hand")
    panda_hand_pose = plant.EvalBodyPoseInWorld(plant_context, panda_hand_body)
    panda_hand_translation = panda_hand_pose.translation()
    
    # print(f"panda_link7 translation: {panda_link7_translation}")
    # print(f"panda_hand translation: {panda_hand_translation}")
    # print("=" * 30)
    
except Exception as e:
    # print(f"Error getting final body poses: {e}")
    pass

# Create a figure for panda_link7 world positions
fig_panda_link7, axs_panda_link7 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig_panda_link7.suptitle('Panda Link7 World Positions (XYZ) - Direct Pose Extraction')

# Plot panda_link7 positions
axs_panda_link7[0].plot(t_time, panda_link7_positions[:, 0], label='Position X', linestyle='-', color='red', linewidth=2)
axs_panda_link7[1].plot(t_time, panda_link7_positions[:, 1], label='Position Y', linestyle='-', color='green', linewidth=2)
axs_panda_link7[2].plot(t_time, panda_link7_positions[:, 2], label='Position Z', linestyle='-', color='blue', linewidth=2)

axs_panda_link7[0].set_ylabel('X Position [m]')
axs_panda_link7[1].set_ylabel('Y Position [m]')
axs_panda_link7[2].set_ylabel('Z Position [m]')
axs_panda_link7[2].set_xlabel('Time [s]')

for ax in axs_panda_link7:
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_xlim([t_time[1], t_time[-1]])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Print summary statistics for panda_link7 positions
# print("\n=== Panda Link7 Position Summary (Direct Pose Extraction) ===")
# print(f"Initial position: [{panda_link7_positions[0, 0]:.4f}, {panda_link7_positions[0, 1]:.4f}, {panda_link7_positions[0, 2]:.4f}] m")
# print(f"Final position: [{panda_link7_positions[-1, 0]:.4f}, {panda_link7_positions[-1, 1]:.4f}, {panda_link7_positions[-1, 2]:.4f}] m")
# print(f"Total displacement: {np.linalg.norm(panda_link7_positions[-1] - panda_link7_positions[0]):.4f} m")
# print(f"X range: [{np.min(panda_link7_positions[:, 0]):.4f}, {np.max(panda_link7_positions[:, 0]):.4f}] m")
# print(f"Y range: [{np.min(panda_link7_positions[:, 1]):.4f}, {np.max(panda_link7_positions[:, 1]):.4f}] m")
# print(f"Z range: [{np.min(panda_link7_positions[:, 2]):.4f}, {np.max(panda_link7_positions[:, 2]):.4f}] m")

# Identify modified joints (comparing with initial configuration)
modified_indices = np.where(trajInit_ != 0)[0]  # Find non-zero initial positions

# Joint names and number of modified positions
joint_names = ['Panda joint 1', 'Panda joint 2', 'Panda joint 3', 'Panda joint 4', 'Panda joint 5', 'Panda joint 6', 'Panda joint 7', 'panda_figer1', 'panda_figer2']
modified_joint_names = [joint_names[i] for i in modified_indices]

num_positions = len(modified_indices)  # Number of modified joints

# Create a figure for position plots for modified joints
fig_pos, axs_pos = plt.subplots(num_positions, 1, figsize=(12, 3*num_positions), sharex=True)
fig_pos.suptitle('Joint Positions')

for idx, joint_idx in enumerate(modified_indices):
    axs_pos[idx].plot(t_time, data_q[:, joint_idx], label='Joint position', linestyle='-')
    axs_pos[idx].plot(t_time, data_qv[:, joint_idx], label='Filtered reference position', linestyle='--')
    axs_pos[idx].plot(t_time, data_qr[:, joint_idx], label='Reference position', linestyle=':')
    axs_pos[idx].set_ylabel('Position [rad]')
    axs_pos[idx].set_title(f'{modified_joint_names[idx]}')
    axs_pos[idx].grid(True)
    # Set x-axis limits
    axs_pos[idx].set_xlim([t_time[1], t_time[-1]])

# Add legend to the top of the first subplot
axs_pos[0].legend(loc='upper right')

# Set common xlabel
axs_pos[-1].set_xlabel('Time [s]')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create a figure for velocity plots for modified joints
fig_vel, axs_vel = plt.subplots(num_positions, 1, figsize=(12, 3*num_positions), sharex=True)
fig_vel.suptitle('Joint Velocities')

for idx, joint_idx in enumerate(modified_indices):
    axs_vel[idx].plot(t_time, data_qdot[:, joint_idx], label='Joint velocity', linestyle='-')
    axs_vel[idx].plot(t_time, limit_dq[joint_idx] * np.ones_like(t_time), 'r--', label='Velocity limit')
    axs_vel[idx].plot(t_time, -limit_dq[joint_idx] * np.ones_like(t_time), 'r--')
    axs_vel[idx].set_ylabel('Velocity [rad/s]')
    axs_vel[idx].set_title(f'{modified_joint_names[idx]}')
    axs_vel[idx].grid(True)
    # Set x-axis limits
    axs_vel[idx].set_xlim([t_time[1], t_time[-1]])

# Add legend to the top of the first subplot
axs_vel[0].legend(loc='upper right')

# Set common xlabel
axs_vel[-1].set_xlabel('Time [s]')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create a figure for torque plots for modified joints
fig_tau, axs_tau = plt.subplots(num_positions, 1, figsize=(12, 3*num_positions), sharex=True)
fig_tau.suptitle('Joint Torques')

for idx, joint_idx in enumerate(modified_indices):
    axs_tau[idx].plot(t_time, data_tau[:, joint_idx], label='Applied torque', linestyle='-')
    axs_tau[idx].plot(t_time, limit_tau[joint_idx] * np.ones_like(t_time), 'r--', label='Torque limit')
    axs_tau[idx].plot(t_time, -limit_tau[joint_idx] * np.ones_like(t_time), 'r--')
    axs_tau[idx].set_ylabel('Torque [Nm]')
    axs_tau[idx].set_title(f'{modified_joint_names[idx]}')
    axs_tau[idx].grid(True)
    # Set x-axis limits
    axs_tau[idx].set_xlim([t_time[1], t_time[-1]])

# Add legend to the top of the first subplot
axs_tau[0].legend(loc='upper right')

# Set common xlabel
axs_tau[-1].set_xlabel('Time [s]')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create a figure for box poses (all bodies)
fig_box_poses, axs_box_poses = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig_box_poses.suptitle('Movable Box Position (XYZ)')

# The state data contains: [quaternion_w, quaternion_x, quaternion_y, quaternion_z, position_x, position_y, position_z]
# Extract positions only
position_data = data_box_pos[:, 4:7]    # Next 3 elements are spatial position

# Plot positions
axs_box_poses[0].plot(t_time, position_data[:, 0], label='Position X', linestyle='-', color='red')
axs_box_poses[1].plot(t_time, position_data[:, 1], label='Position Y', linestyle='-', color='green')
axs_box_poses[2].plot(t_time, position_data[:, 2], label='Position Z', linestyle='-', color='blue')

axs_box_poses[0].set_ylabel('X Position [m]')
axs_box_poses[1].set_ylabel('Y Position [m]')
axs_box_poses[2].set_ylabel('Z Position [m]')
axs_box_poses[2].set_xlabel('Time [s]')

for ax in axs_box_poses:
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_xlim([t_time[1], t_time[-1]])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create 4 clean graphs with consistent color scheme
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Robot Control Performance', fontsize=16)

# Define consistent color scheme for all 7 joints
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']

# Graph 1: Commanded (IK solutions) vs ERG output (aux reference/v)
ax1.set_title('Commanded vs ERG Output (Joint Positions)', fontsize=14)
for i in range(7):
    ax1.plot(t_time, data_qr[:, i], label=f'{joint_names[i]} (IK)', color=colors[i], linestyle='-', linewidth=2)
    ax1.plot(t_time, data_qv[:, i], label=f'{joint_names[i]} (ERG)', color=colors[i], linestyle='--', linewidth=2)
ax1.set_ylabel('Position [rad]')
ax1.grid(True)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.set_xlim([t_time[1], t_time[-1]])

# Graph 2: Joint Velocities
ax2.set_title('Joint Velocities', fontsize=14)
for i in range(7):
    ax2.plot(t_time, data_qdot[:, i], label=f'{joint_names[i]}', color=colors[i], linewidth=2)
ax2.set_ylabel('Velocity [rad/s]')
ax2.grid(True)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_xlim([t_time[1], t_time[-1]])

# Graph 3: Joint Torques
ax3.set_title('Joint Torques', fontsize=14)
for i in range(7):
    ax3.plot(t_time, data_tau[:, i], label=f'{joint_names[i]}', color=colors[i], linewidth=2)
ax3.set_ylabel('Torque [Nm]')
ax3.set_xlabel('Time [s]')
ax3.grid(True)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.set_xlim([t_time[1], t_time[-1]])

# Graph 4: Joint Positions (Actual)
ax4.set_title('Actual Joint Positions', fontsize=14)
for i in range(7):
    ax4.plot(t_time, data_q[:, i], label=f'{joint_names[i]}', color=colors[i], linewidth=2)
ax4.set_ylabel('Position [rad]')
ax4.set_xlabel('Time [s]')
ax4.grid(True)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.set_xlim([t_time[1], t_time[-1]])

plt.tight_layout()
plt.show()



# Create a figure for system energy
fig_energy, axs_energy = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig_energy.suptitle('System Energy')

# Plot total energy
axs_energy[0].plot(t_time, data_energy[:, 0], label='Total Energy', linestyle='-', color='blue')
axs_energy[0].set_ylabel('Energy [J]')
axs_energy[0].grid(True)
axs_energy[0].legend(loc='upper right')

# Plot kinetic and potential energy if available
if data_energy.shape[1] > 1:
    axs_energy[1].plot(t_time, data_energy[:, 1], label='Kinetic Energy', linestyle='-', color='red')
    axs_energy[1].plot(t_time, data_energy[:, 2], label='Potential Energy', linestyle='-', color='green')
    axs_energy[1].set_ylabel('Energy [J]')
    axs_energy[1].grid(True)
    axs_energy[1].legend(loc='upper right')

axs_energy[-1].set_xlabel('Time [s]')
axs_energy[-1].set_xlim([t_time[1], t_time[-1]])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create a figure for contact forces
fig_contact, axs_contact = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig_contact.suptitle('Contact Forces')

# Plot contact forces (assuming first 3 components are x, y, z forces)
if data_contact_forces.shape[1] >= 3:
    axs_contact[0].plot(t_time, data_contact_forces[:, 0], label='Contact Force X', linestyle='-', color='red')
    axs_contact[1].plot(t_time, data_contact_forces[:, 1], label='Contact Force Y', linestyle='-', color='green')
    axs_contact[2].plot(t_time, data_contact_forces[:, 2], label='Contact Force Z', linestyle='-', color='blue')
    
    axs_contact[0].set_ylabel('Force X [N]')
    axs_contact[1].set_ylabel('Force Y [N]')
    axs_contact[2].set_ylabel('Force Z [N]')
    
    for ax in axs_contact:
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.set_xlim([t_time[1], t_time[-1]])
    
    axs_contact[2].set_xlabel('Time [s]')
else:
    # If contact forces data structure is different, plot all available components
    for i in range(min(3, data_contact_forces.shape[1])):
        axs_contact[i].plot(t_time, data_contact_forces[:, i], label=f'Contact Force {i+1}', linestyle='-')
        axs_contact[i].set_ylabel(f'Force {i+1} [N]')
        axs_contact[i].grid(True)
        axs_contact[i].legend(loc='upper right')
        axs_contact[i].set_xlim([t_time[1], t_time[-1]])
    
    axs_contact[2].set_xlabel('Time [s]')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create a figure for Z-axis integrator output (only Z, with contact detection markers)
fig_z_integrator, ax_z = plt.subplots(1, 1, figsize=(12, 6))
fig_z_integrator.suptitle('Z-Axis Integrator Output (Only Integrated Z)')

# Plot Z-axis integrator output (only Z component)
if data_z_integrated.shape[1] >= 3:
    ax_z.plot(t_time, data_z_integrated[:, 2], label='Integrated Z', linestyle='-', color='blue', linewidth=2)
    
    # Find when contact transitions from 0 to 1
    contact_values = data_contact_flag.flatten()
    contact_transitions = []
    for i in range(1, len(contact_values)):
        # Check if contact just became active (0 -> 1)
        if contact_values[i] > 0.5 and contact_values[i-1] <= 0.5:
            contact_transitions.append(t_time[i])
    
    # Add vertical lines at contact detection points
    for transition_time in contact_transitions:
        ax_z.axvline(x=transition_time, color='red', linestyle='--', linewidth=1.5, 
                     label='Contact detected' if transition_time == contact_transitions[0] else '')
    
    ax_z.set_ylabel('Integrated Z [m]')
    ax_z.set_xlabel('Time [s]')
    ax_z.grid(True)
    ax_z.legend(loc='upper right')
    ax_z.set_xlim([t_time[1], t_time[-1]])
    
    print(f"\nContact detected at times: {contact_transitions}")
else:
    # If Z-axis integrator data structure is different, plot the last component
    ax_z.plot(t_time, data_z_integrated[:, -1], label='Integrated Z', linestyle='-', color='blue', linewidth=2)
    ax_z.set_ylabel('Integrated Z [m]')
    ax_z.set_xlabel('Time [s]')
    ax_z.grid(True)
    ax_z.legend(loc='upper right')
    ax_z.set_xlim([t_time[1], t_time[-1]])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Keep plots open
print("\nPlots displayed. Close windows or press Enter to continue...")
input()

# Print summary statistics for energy and contact forces
# print("\n=== Energy and Contact Forces Summary ===")
# print(f"Final total energy: {data_energy[-1, 0]:.4f} J")
# if data_energy.shape[1] > 1:
#     print(f"Final kinetic energy: {data_energy[-1, 1]:.4f} J")
#     print(f"Final potential energy: {data_energy[-1, 2]:.4f} J")

# print(f"Max contact force magnitude: {np.max(np.linalg.norm(data_contact_forces, axis=1)):.4f} N")
# print(f"Average contact force magnitude: {np.mean(np.linalg.norm(data_contact_forces, axis=1)):.4f} N")

# Print summary statistics for Z-axis integrator
# print("\n=== Z-Axis Integrator Summary ===")
# if data_z_integrated.shape[1] >= 3:
#     print(f"Final integrated position: [{data_z_integrated[-1, 0]:.4f}, {data_z_integrated[-1, 1]:.4f}, {data_z_integrated[-1, 2]:.4f}] m")
#     print(f"X range: [{np.min(data_z_integrated[:, 0]):.4f}, {np.max(data_z_integrated[:, 0]):.4f}] m")
#     print(f"Y range: [{np.min(data_z_integrated[:, 1]):.4f}, {np.max(data_z_integrated[:, 1]):.4f}] m")
#     print(f"Z range: [{np.min(data_z_integrated[:, 2]):.4f}, {np.max(data_z_integrated[:, 2]):.4f}] m")
# else:
#     print(f"Z-axis integrator output shape: {data_z_integrated.shape}")
#     print(f"Final integrated values: {data_z_integrated[-1, :]}")
