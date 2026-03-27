import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

# Enable interactive plotting to keep windows open
plt.ion()  
from pydrake.all import *
# import pydot
from IPython.display import SVG, display
from scipy.spatial.transform import Rotation as R

# Import ExplicitReferenceGovernor from trajectoryERG
from trajectoryERG import ExplicitReferenceGovernor

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

        # Expose the box position as used *inside* the ERG (after local preprocessing / offsets)
        # This makes it easy to log + plot what Trajectory ERG is actually seeing.
        self._box_position_used_in_erg = np.zeros(3)
        self.DeclareVectorOutputPort(
            "box_position_used_in_erg", size=3, calc=self.output_box_position_used_in_erg
        )
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.refrence) # Call the Update method defined below.
        # Use the same URDF as the main plant
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_drake.urdf"))
        
        self.erg = ExplicitReferenceGovernor(
            robust_delta_tau_=0.1, kappa_tau_=1.0,
            robust_delta_q_=0.1, kappa_q_=15.0, robust_delta_dq_=0.1, kappa_dq_=7.0,
            robust_delta_dp_EE_=0.01, kappa_dp_EE_=7.0, kappa_terminal_energy_=7.5, FD_=1.0,
            num_joints=self.num_joints, urdf_path=urdf_path)

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
        # Box center position (true world position, used for FCL collision box placement)
        box_position_center = np.array(box_state[4:7], dtype=float)

        # Box position used for constraints / face-tracking (subtract half-length in X)
        box_position = box_position_center.copy()
        box_position[0] -= 0.11
        self._box_position_used_in_erg = box_position.copy()

        self.q_v = context.get_discrete_state_vector().CopyToVector()              
        # Initialize q_v_ only at the first callback
        if self.first_update:
            self.q_v_ = trajInit_  # Use all 9 joints    
            self.first_update = False
        
        # Update box position in ERG
        self.q_v_ = self.erg.get_qv(
            q, dq, tau, q_r, self.q_v_,
            box_position,
            box_position_fcl=box_position_center,
        )
        
        # Calculate energy from trajectory predictions
        self.calculated_energy = self.erg.get_energy(
            q, dq, tau, q_r, self.q_v_,
            box_position,
            box_position_fcl=box_position_center,
        )

        # Write into the output vector.
        discrete_state.get_mutable_vector().SetFromVector(self.q_v_)
        
    def output_energy(self, context, output):
        # Return the calculated energy from the ERG system
        if hasattr(self, 'calculated_energy'):
            output.SetAtIndex(0, self.calculated_energy)
        else:
            output.SetAtIndex(0, 0.0)

    def output_box_position_used_in_erg(self, context, output):
        output.SetFromVector(self._box_position_used_in_erg)


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

# -------------------- FCL Boxes Visualization (Meshcat) --------------------
# Visualize the same simple box primitives used by trajERG's FCL collision checking.
if meshcat_visualisation:
    class FCLBoxesMeshcatVisualizer(LeafSystem):
        def __init__(self, plant, panda_id, movable_box_id, num_robot_positions, meshcat):
            super().__init__()
            self.plant = plant
            self.panda_id = panda_id
            self.movable_box_id = movable_box_id
            self.num_robot_positions = num_robot_positions
            self.meshcat = meshcat
            self.temp_context = plant.CreateDefaultContext()

            # Inputs
            self.DeclareVectorInputPort("robot_state", size=num_robot_positions * 2)
            self.DeclareVectorInputPort("box_state", size=13)

            # Publish periodically
            self.DeclarePeriodicPublishEvent(
                period_sec=0.01, offset_sec=0.0, publish=self.DoPublish
            )

            # Build the list of robot bodies to visualize (match trajERG logic)
            names = ["panda_link7", "panda_hand"]
            for n in ["panda_leftfinger", "panda_rightfinger"]:
                try:
                    self.plant.GetBodyByName(n)
                    names.append(n)
                except Exception:
                    pass

            # Always include rubber pad if present (even if fingers exist)
            for pad in ["rubber_pad", "panda_rubber_pad", "gripper_pad"]:
                try:
                    self.plant.GetBodyByName(pad)
                    if pad not in names:
                        names.append(pad)
                    break
                except Exception:
                    continue

            self.robot_body_names = names

            # Create Meshcat objects once
            for name in self.robot_body_names:
                if "finger" in name.lower():
                    dims = (0.05, 0.05, 0.05)
                elif ("rubber" in name.lower()) or ("pad" in name.lower()):
                    dims = (0.08, 0.08, 0.08)
                else:
                    dims = (0.10, 0.10, 0.10)

                self.meshcat.SetObject(
                    f"fcl/robot/{name}",
                    Box(*dims),
                    Rgba(1.0, 0.2, 0.2, 0.35),
                )

            # Box used for collision checking (movable_box.sdf: [0.22, 0.30, 0.20])
            self.meshcat.SetObject(
                "fcl/box",
                Box(0.22, 0.30, 0.20),
                Rgba(0.2, 0.2, 1.0, 0.25),
            )

        # Drake's periodic publish callback is invoked with (context) in pydrake.
        def DoPublish(self, context):
            # Robot bodies
            robot_state = self.GetInputPort("robot_state").Eval(context)
            q = robot_state[: self.num_robot_positions]

            try:
                self.plant.SetPositions(self.temp_context, self.panda_id, q)
            except Exception:
                # Fallback if model instance overload isn't available
                try:
                    self.plant.SetPositions(self.temp_context, q)
                except Exception:
                    return

            for name in self.robot_body_names:
                try:
                    body = self.plant.GetBodyByName(name)
                    X_WB = self.plant.EvalBodyPoseInWorld(self.temp_context, body)
                    self.meshcat.SetTransform(f"fcl/robot/{name}", X_WB)
                except Exception:
                    continue

            # Moving box (pose from its state)
            try:
                box_state = self.GetInputPort("box_state").Eval(context)
                q_wxyz = box_state[0:4]
                p_xyz = box_state[4:7]
                X_WBox = RigidTransform(RotationMatrix(Quaternion(q_wxyz)), p_xyz)
                self.meshcat.SetTransform("fcl/box", X_WBox)
            except Exception:
                pass

    fcl_viz = builder.AddNamedSystem(
        "FCLBoxesMeshcatVisualizer",
        FCLBoxesMeshcatVisualizer(
            plant=plant,
            panda_id=panda_id,
            movable_box_id=movable_box_id,
            num_robot_positions=num_robot_positions,
            meshcat=meshcat,
        ),
    )
    builder.Connect(
        plant.get_state_output_port(panda_id),
        fcl_viz.GetInputPort("robot_state"),
    )
    builder.Connect(
        plant.get_state_output_port(movable_box_id),
        fcl_viz.GetInputPort("box_state"),
    )

logger_x = LogVectorOutput(plant.get_state_output_port(), builder) #state
logger_tau = LogVectorOutput(pid_controller.GetOutputPort("tau_u"), builder) #tau_u
logger_qv = LogVectorOutput(erg_system.GetOutputPort("q_v_filtered"), builder) #q_v
logger_qr = LogVectorOutput(ik_box_tracker.GetOutputPort("ik_joint_targets"), builder) #ik targets
logger_box_pos = LogVectorOutput(plant.get_state_output_port(box_id), builder) #box position
logger_box_pos_used_erg = LogVectorOutput(erg_system.GetOutputPort("box_position_used_in_erg"), builder)

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

# Extract rubber_pad world position (similar to panda_link7 extractor)
class RubberPadPoseExtractor(LeafSystem):
    def __init__(self, plant, panda_id, num_robot_positions):
        super().__init__()
        self.plant = plant
        self.panda_id = panda_id
        self.num_robot_positions = num_robot_positions
        self.temp_context = plant.CreateDefaultContext()

        self.DeclareVectorInputPort("joint_positions", size=num_robot_positions * 2)
        self.DeclareVectorOutputPort("rubber_pad_world_positions", size=3, calc=self.CalcOutput)

        # Cache body selection (best-effort)
        self._rubber_pad_body_name = None
        for name in ["rubber_pad", "panda_rubber_pad", "gripper_pad", "panda_hand"]:
            try:
                self.plant.GetBodyByName(name)
                self._rubber_pad_body_name = name
                break
            except Exception:
                continue

        if self._rubber_pad_body_name is None:
            self._rubber_pad_body_name = "panda_hand"  # last resort

    def CalcOutput(self, context, output):
        full_state = self.GetInputPort("joint_positions").Eval(context)
        joint_positions = full_state[:self.num_robot_positions]

        # Update plant positions
        try:
            self.plant.SetPositions(self.temp_context, self.panda_id, joint_positions)
        except Exception:
            try:
                self.plant.SetPositions(self.temp_context, joint_positions)
            except Exception:
                output.SetFromVector([0.0, 0.0, 0.0])
                return

        try:
            body = self.plant.GetBodyByName(self._rubber_pad_body_name)
            pose = self.plant.EvalBodyPoseInWorld(self.temp_context, body)
            p = pose.translation()
            output.SetFromVector([p[0], p[1], p[2]])
        except Exception:
            output.SetFromVector([0.0, 0.0, 0.0])

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

# Add RubberPadPoseExtractor to the diagram and logger
rubber_pad_extractor = builder.AddNamedSystem(
    "RubberPadPoseExtractor", RubberPadPoseExtractor(plant, panda_id, num_robot_positions)
)
builder.Connect(
    plant.get_state_output_port(panda_id),
    rubber_pad_extractor.GetInputPort("joint_positions"),
)
logger_rubber_pad_world = LogVectorOutput(
    rubber_pad_extractor.GetOutputPort("rubber_pad_world_positions"), builder
)

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
                                         z_min=0.0,   # No negative integration — only correct slip downward
                                         z_max=5.0,   # Maximum Z limit
                                         Kaw_z=0.7,   # Anti-windup gain (matched to Ki_z)
                                         error_mode="a_minus_b",  # a.z - b.z
                                         passthrough_xy_from="a",  # Use X,Y from first input
                                         name="ZAxisIntegrator"
                                     ))

# First input: freeze box Z at first contact, then hold it as the reference.
# Before contact: outputs current box Z so ez=0 (no integration).
# After first contact: holds the frozen Z — any slip below it drives the integrator up.
class TargetPositionExtractor(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort("box_state", size=13)
        self.DeclareVectorInputPort("contact", size=1)
        # discrete state: [frozen_z, contact_ever]
        self.DeclareDiscreteState(2)
        self.DeclareVectorOutputPort("target_position", size=3, calc=self.CalcOutput)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.001, offset_sec=0.0, update=self.Update
        )

    def Update(self, context, discrete_state):
        s = context.get_discrete_state_vector().CopyToVector()
        frozen_z, contact_ever = s[0], s[1]
        contact = self.GetInputPort("contact").Eval(context)[0]
        if contact_ever < 0.5 and contact > 0.5:
            box_state = self.GetInputPort("box_state").Eval(context)
            frozen_z = float(box_state[6]) + 0.4  # target = contact height + lift offset
            contact_ever = 1.0
        discrete_state.get_mutable_vector().SetFromVector([frozen_z, contact_ever])

    def CalcOutput(self, context, output):
        box_state = self.GetInputPort("box_state").Eval(context)
        x_b, y_b, z_b = box_state[4:7]
        s = context.get_discrete_state_vector().CopyToVector()
        frozen_z, contact_ever = s[0], s[1]
        z_ref = frozen_z if contact_ever > 0.5 else z_b
        output.SetFromVector([x_b, y_b, z_ref])

target_extractor = builder.AddSystem(TargetPositionExtractor())
builder.Connect(plant.get_state_output_port(box_id),
                target_extractor.GetInputPort("box_state"))
builder.Connect(contact_detector.GetOutputPort("contact"),
                target_extractor.GetInputPort("contact"))

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
log_box_pos_used_erg = logger_box_pos_used_erg.FindLog(diagram_context)
log_energy = logger_energy.FindLog(diagram_context)
log_contact_forces = logger_contact_forces.FindLog(diagram_context)
log_panda_link7_world = logger_panda_link7_world.FindLog(diagram_context)
log_rubber_pad_world = logger_rubber_pad_world.FindLog(diagram_context)
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
data_box_pos_used_erg = log_box_pos_used_erg.data().transpose()  # box position as seen inside ERG (shifted)
data_energy = log_energy.data().transpose() # system energy
data_contact_forces = log_contact_forces.data().transpose() # contact forces
data_panda_link7_world = log_panda_link7_world.data().transpose() # panda_link7 world positions
data_rubber_pad_world = log_rubber_pad_world.data().transpose() # rubber_pad world positions
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

####################################
# Debug prints: box lift analysis
####################################
box_z = data_box_pos[:, 6]
pad_z = data_rubber_pad_world[:, 2]
integ_z = data_z_integrated[:, 2]
contact_vals = data_contact_flag.flatten()

# Find first contact time
contact_idx = np.where(contact_vals > 0.5)[0]
t_contact = t_time[contact_idx[0]] if len(contact_idx) > 0 else None
box_z_at_contact = box_z[contact_idx[0]] if t_contact is not None else None

# Find t=6s (lift start) index
lift_idx = np.searchsorted(t_time, 6.0)
t_end_idx = -1

print("\n=== Box Lift Debug ===")
if t_contact is not None:
    print(f"First contact at t={t_contact:.2f}s,  box_z={box_z_at_contact:.4f}m")
    print(f"Frozen Z reference (target) = box_z_at_contact + 0.4 = {box_z_at_contact + 0.4:.4f}m")
else:
    print("No contact detected during simulation!")
print(f"Box Z at t=0s   : {box_z[0]:.4f}m")
print(f"Box Z at t=6s   : {box_z[lift_idx]:.4f}m")
print(f"Box Z at t=end  : {box_z[t_end_idx]:.4f}m")
print(f"Box Z lift achieved: {box_z[t_end_idx] - box_z[0]:.4f}m  (expected ~0.4m)")
print(f"Rubber pad Z at t=0s  : {pad_z[0]:.4f}m")
print(f"Rubber pad Z at t=end : {pad_z[t_end_idx]:.4f}m")
print(f"Rubber pad Z lift     : {pad_z[t_end_idx] - pad_z[0]:.4f}m")
print(f"Integrator u[2] at end: {integ_z[t_end_idx]:.4f}m")
print(f"Box Z gap from target at end: {(box_z_at_contact + 0.4 if box_z_at_contact else 0) - box_z[t_end_idx]:.4f}m")
print("=" * 40)

####################################
# ERG energy limit
####################################
E_max = erg_system.erg.E_max_  # from trajectoryERG.py line 202

####################################
# Plot 1: All world positions (X, Y, Z) — link7, rubber_pad, box — in one figure
####################################
fig_world, axs_world = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig_world.suptitle('World Positions: Link7 / Rubber Pad / Box', fontsize=13)
labels_xyz = ['X [m]', 'Y [m]', 'Z [m]']
for i in range(3):
    axs_world[i].plot(t_time, panda_link7_positions[:, i], label='link7',      color='steelblue',  linewidth=2)
    axs_world[i].plot(t_time, data_rubber_pad_world[:, i],  label='rubber_pad', color='darkorange',  linewidth=2)
    axs_world[i].plot(t_time, data_box_pos[:, 4+i],         label='box',        color='green',       linewidth=2)
    axs_world[i].set_ylabel(labels_xyz[i])
    axs_world[i].grid(True, alpha=0.3)
    axs_world[i].set_xlim([t_time[0], t_time[-1]])
axs_world[0].legend(loc='upper right')
axs_world[2].set_xlabel('Time [s]')
plt.tight_layout()
plt.show()

####################################
# Plot 2: Joint positions — all 7 joints, each with q / q_v / q_r + limits
####################################
fig_joints, axs_joints = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
fig_joints.suptitle('Joint Positions: actual (q) / ERG ref (q_v) / IK cmd (q_r) + limits', fontsize=13)
colors7 = plt.cm.tab10(np.linspace(0, 0.7, 7))
for i in range(7):
    ax = axs_joints[i]
    ax.plot(t_time, data_q[:, i],  color=colors7[i], linewidth=2,   linestyle='-',  label='q (actual)')
    ax.plot(t_time, data_qv[:, i], color=colors7[i], linewidth=1.5, linestyle='--', label='q_v (ERG)')
    ax.plot(t_time, data_qr[:, i], color=colors7[i], linewidth=1.5, linestyle=':',  label='q_r (IK)')
    ax.axhline(limit_q_min[i], color='red',  linewidth=1.0, linestyle='--', alpha=0.7, label='limits')
    ax.axhline(limit_q_max[i], color='red',  linewidth=1.0, linestyle='--', alpha=0.7)
    ax.set_ylabel(f'J{i+1} [rad]', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_time[0], t_time[-1]])
axs_joints[0].legend(loc='upper right', fontsize=8, ncol=4)
axs_joints[6].set_xlabel('Time [s]')
plt.tight_layout()
plt.show()

####################################
# Plot 3: Energy + Emax in one graph
####################################
fig_energy, ax_energy = plt.subplots(1, 1, figsize=(12, 5))
fig_energy.suptitle('ERG Energy vs Emax', fontsize=13)
ax_energy.plot(t_time, data_energy[:, 0], color='steelblue', linewidth=2, label='Energy')
ax_energy.axhline(E_max, color='red', linewidth=1.5, linestyle='--', label=f'E_max = {E_max:.3f}')
ax_energy.set_ylabel('Energy [J]')
ax_energy.set_xlabel('Time [s]')
ax_energy.grid(True, alpha=0.3)
ax_energy.legend(loc='upper right')
ax_energy.set_xlim([t_time[0], t_time[-1]])
plt.tight_layout()
plt.show()

####################################
# Plot 4: Z integrator + box Z + contact flag
####################################
fig_z, axs_z = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig_z.suptitle('Z Integrator / Box Z / Contact', fontsize=13)
axs_z[0].plot(t_time, integ_z,          color='darkorange', linewidth=2, label='u[2] (integrated Z)')
axs_z[0].plot(t_time, pad_z,            color='steelblue',  linewidth=2, label='rubber_pad Z')
if box_z_at_contact is not None:
    axs_z[0].axhline(box_z_at_contact + 0.4, color='green', linewidth=1.5, linestyle='--', label=f'target Z={box_z_at_contact+0.4:.3f}')
axs_z[0].set_ylabel('Z [m]')
axs_z[0].legend(fontsize=9)
axs_z[0].grid(True, alpha=0.3)
axs_z[1].plot(t_time, box_z, color='green', linewidth=2, label='box Z')
if box_z_at_contact is not None:
    axs_z[1].axhline(box_z_at_contact + 0.4, color='red', linewidth=1.5, linestyle='--', label='target Z')
axs_z[1].set_ylabel('Box Z [m]')
axs_z[1].legend(fontsize=9)
axs_z[1].grid(True, alpha=0.3)
axs_z[2].plot(t_time, contact_vals, color='purple', linewidth=2, drawstyle='steps-post', label='contact')
axs_z[2].set_ylim(-0.1, 1.2)
axs_z[2].set_ylabel('contact')
axs_z[2].set_xlabel('Time [s]')
axs_z[2].grid(True, alpha=0.3)
for ax in axs_z:
    ax.set_xlim([t_time[0], t_time[-1]])
plt.tight_layout()
plt.show()

####################################
# Plot 5: Contact forces
####################################
fig_contact, axs_contact = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
fig_contact.suptitle('Contact Forces (X, Y, Z)', fontsize=13)
for i, (label, color) in enumerate([('Fx', 'red'), ('Fy', 'green'), ('Fz', 'blue')]):
    axs_contact[i].plot(t_time, data_contact_forces[:, i], color=color, linewidth=2, label=label)
    axs_contact[i].set_ylabel(f'{label} [N]')
    axs_contact[i].grid(True, alpha=0.3)
    axs_contact[i].set_xlim([t_time[0], t_time[-1]])
axs_contact[0].legend()
axs_contact[2].set_xlabel('Time [s]')
plt.tight_layout()
plt.show()

print("\nPlots displayed. Close windows or press Enter to continue...")
input()

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

