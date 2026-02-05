import numpy as np
import time
import csv  
import os
import sys

# Make imports robust in both "Run" and "Debug" (debugpy may not put this file's
# directory on sys.path in some launch configurations).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pydrake.all import *
# import pydot
from IPython.display import SVG, display

# Import CompliantERG from trajectoryERG
from trajectoryERG import CompliantERG
from CERG_Setup import ErgParams, ContactParams, panda_spec
from plotting import BodyPoseExtractor, add_mux_logger, LogSignal
from plot_results import plot_results
from controllers.pd_gravity import PD_gravity as SharedPDGravity


# Add Relaxed IK wrapper import
wrapper_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "submodules/relaxed_ik_core/wrappers")
sys.path.insert(0, wrapper_dir)
from python_wrapper import RelaxedIKRust

# Robot spec + shared initial joint configuration from setup
ROBOT_SPEC = panda_spec()
trajInit_ = np.array(
    ROBOT_SPEC.q0 if ROBOT_SPEC.q0 is not None else np.zeros(ROBOT_SPEC.num_positions),
    dtype=float,
)

# -----------------------------
# Box tracking target definition
# -----------------------------
# The box state position (indices 4:7 of the 13D state) is the *box link origin* in world frame.
# Use these offsets to choose what point we want the end-effector to track.
# - For box center-of-mass (typical): [0, 0, 0]
# - For top-center (avoid floor/table penetration): [0, 0, +box_half_height + clearance]
# NOTE: movable_box.sdf size in x is 0.22, so half-length is 0.11.
# Using -0.11 in x targets the "near" face (assuming -x points from box toward robot).
BOX_TARGET_OFFSET_W = np.array([-0.11, 0.0, 0.0], dtype=float)

# -----------------------------
# Tracking mode (box vs fixed point)
# -----------------------------
# Track the movable box (spawned in the plant) with an offset.
USE_BOX_TARGET = True
FIXED_TARGET_W = np.array([0.7, 0.1, 0.22], dtype=float)

# When there is no real box in the plant, ERG (and some debug utilities) still expect a "box_state"
# vector. Drake's free-body state output for a floating body is typically 13D:
#   [quat_w, quat_x, quat_y, quat_z, pos_x, pos_y, pos_z, v_x, v_y, v_z, w_x, w_y, w_z]
# We'll synthesize a fixed state centered at FIXED_TARGET_W with identity quaternion and zero velocity.
FIXED_BOX_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

def _make_fixed_box_state(size: int) -> np.ndarray:
    """
    Build a fixed 'box_state' vector sized to whatever the input port expects.
    - First 4 entries: quaternion (wxyz)
    - Next 3 entries: position (xyz)
    - Remaining entries: zeros
    """
    size = int(size)
    x = np.zeros(size, dtype=float)
    if size >= 4:
        x[:4] = FIXED_BOX_QUAT_WXYZ
    if size >= 7:
        x[4:7] = FIXED_TARGET_W
    return x

# Prevent IK from ever targeting below the "floor/table" top surface.
# fixed_box.sdf is a 0.1-thick box centered at z=0, so its top is at z=+0.05.
FLOOR_TOP_Z = 0
MIN_IK_TARGET_Z = FLOOR_TOP_Z + 0.02

# Debugging: set to False to bypass ERG and track IK joint targets directly.
USE_ERG_FILTERED_REFERENCE = True

def create_system_model(plant, scene_graph, robot_spec, contact_params):
    """
    Add the Panda arm model to the plant and configure contact properties.
    
    Args:
        plant: The MultibodyPlant object to which the Panda arm model will be added.
        scene_graph: The SceneGraph object for visualization.
    
    Returns:
        Tuple containing the updated plant and scene_graph.
    """
    urdf = "file://" + robot_spec.urdf_path
    arm = Parser(plant).AddModelsFromUrl(urdf)
    
    # Add the fixed box to the scene (acting as static ground)
    fixed_box_sdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/boxes/fixed_box.sdf"))
    fixed_box_sdf_url = "file://" + fixed_box_sdf_path
    fixed_box = Parser(plant).AddModelsFromUrl(fixed_box_sdf_url)
    
    # Get the fixed box model instance ID
    fixed_box_id = plant.GetModelInstanceByName("fixed_box")
    
    # The fixed box is already static due to <static>true</static> in the SDF file
    # No need to weld it manually - Drake handles this automatically
    
    # Add the movable box to the scene (on top of the fixed box)
    movable_box_sdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/boxes/movable_box.sdf"))
    movable_box_sdf_url = "file://" + movable_box_sdf_path
    Parser(plant).AddModelsFromUrl(movable_box_sdf_url)
    movable_box_id = plant.GetModelInstanceByName("movable_box")
    
    # Set the contact properties for interaction between the movable box and the ground
    plant.set_contact_surface_representation(contact_params.mesh_type)
    plant.set_contact_model(contact_params.contact_model)
    plant.set_discrete_contact_approximation(contact_params.discrete_solver)
    plant.Finalize()
    
    # Set the *default* initial pose of the movable box so it starts already resting on the fixed box
    # (and doesn't drop from a height when the simulator context is created).
    #
    # fixed_box.sdf: size z = 0.1, pose z = 0.0  => top surface at z = 0.05
    # movable_box.sdf: size z = 0.35           => half height = 0.175
    try:
        fixed_top_z = 0.05
        movable_half_z = 0.35 / 2.0
        z0 =  movable_half_z + 1e-3  # small epsilon to avoid initial penetration
        initial_box_pose = RigidTransform(p=[0.7, 0.0, z0])
        plant.SetDefaultFreeBodyPose(plant.GetBodyByName("box_link", movable_box_id), initial_box_pose)
    except Exception as e:
        print(f"[WARN] Failed to set default movable box pose: {e}")

    # Create a default context for downstream computations (note: this is NOT the simulator context)
    plant_context = plant.CreateDefaultContext()
    
    # Set default positions for the robot (not the box)
    panda_id = plant.GetModelInstanceByName("panda")
    plant.SetDefaultPositions(panda_id, trajInit_)
    
    return plant, scene_graph, plant_context
####################################
# Configurations parameters
####################################
CONTACT_PARAMS = ContactParams()
realtime_factor = 1  # Real-time factor for simulation speed
time_step = 0.001

meshcat_visualisation = True
simulate = True

# Create system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
plant, scene_graph, plant_context = create_system_model(plant, scene_graph, ROBOT_SPEC, CONTACT_PARAMS)
# Set the initial joint position of the robot otherwise it will correspond to zero positions
# plant_context = plant.CreateDefaultContext()  # This is now returned from create_system_model

# Get the panda model instance for specific connections
panda_id = plant.GetModelInstanceByName("panda")
num_positions = ROBOT_SPEC.num_positions
num_velocities = ROBOT_SPEC.num_velocities

# Get the movable box model instance for specific connections
movable_box_id = plant.GetModelInstanceByName("movable_box")

######################################################################################################
#              ##################Relaxed IK System for Box Tracking################
######################################################################################################
class RelaxedIKBoxTracker(LeafSystem):
    def __init__(self, plant, plant_context, robot_spec):
        super().__init__()
        
        # Store plant and context references
        self.plant = plant
        self.plant_context = plant_context
        self.nq = robot_spec.num_positions
        
        # Declare input port for box state (13 values: 7 pose + 6 velocities)
        self._box_state_port = self.DeclareVectorInputPort(name="box_state", size=13)
        
        state_index = self.DeclareDiscreteState(self.nq)
        self.DeclareStateOutputPort("ik_joint_targets", state_index)
        
        # Periodic update for IK solving
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # 10Hz IK updates
            offset_sec=0.0,
            update=self.solve_ik)
        
        # Initialize Relaxed IK
        try:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_rubber_settings.yaml"))
            self.rik = RelaxedIKRust(setting_file_path=config_path)
            print("Relaxed IK initialized successfully for box tracking!")
            self.ik_available = True
        except Exception as e:
            print(f"Failed to initialize Relaxed IK: {e}")
            self.ik_available = False
        
        # Tolerance for IK solving
        self.tolerance = [0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01]  # x,y,z and rx,ry,rz tolerances
        
        # Default orientation (quaternion xyzw)
        # +90° about X: q = [x, y, z, w] = [sin(pi/4), 0, 0, cos(pi/4)]
        _s = float(np.sqrt(0.5))
        self.default_orientation = [_s, 0.0, 0.0, _s]
        
        # Fallback joint configuration sized to robot (use trajInit_)
        base_fallback = list(trajInit_)
        if len(base_fallback) >= self.nq:
            self.fallback_joints = base_fallback[: self.nq]
        else:
            self.fallback_joints = base_fallback + [0.0] * (self.nq - len(base_fallback))

    def solve_ik(self, context, discrete_state):
        if not self.ik_available:
            # Use fallback if IK is not available
            discrete_state.get_mutable_vector().SetFromVector(self.fallback_joints)
            return
        
        try:
            # Target selection:
            # - box target (needs connected box_state) OR
            # - fixed point in space (requested)
            if USE_BOX_TARGET:
                box_state = self._box_state_port.Eval(context)
                box_pos = box_state[4:7]
                target_position_np = np.array(box_pos, dtype=float) + BOX_TARGET_OFFSET_W
                # target_position_np[2] = max(float(target_position_np[2]), float(MIN_IK_TARGET_Z))
            else:
                target_position_np = np.array(FIXED_TARGET_W, dtype=float)
                # target_position_np[2] = max(float(target_position_np[2]), float(MIN_IK_TARGET_Z))
            target_position = target_position_np.tolist()
            
            try:
                # Solve IK with default orientation
                joint_solution = self.rik.solve_position(target_position, self.default_orientation, self.tolerance)
                    
            except Exception as e:
                print(f"IK solving failed: {e}")
                discrete_state.get_mutable_vector().SetFromVector(self.fallback_joints)
                return
            
            # Size to nq (pad if IK gives 7, trim if longer)
            full_solution = list(joint_solution)
            if len(full_solution) < self.nq:
                full_solution += [0.0] * (self.nq - len(full_solution))
            elif len(full_solution) > self.nq:
                full_solution = full_solution[: self.nq]
            
            # Update the discrete state - use all 9 joints
            discrete_state.get_mutable_vector().SetFromVector(full_solution)
            
        except Exception as e:
            print(f"IK solving failed: {e}")
            # Use fallback configuration
            discrete_state.get_mutable_vector().SetFromVector(self.fallback_joints)

######################################################################################################
#                                  ########Box Target Extractor (for logging)#######
######################################################################################################
class BoxTargetExtractor(LeafSystem):
    """
    Takes the movable box 13D state and outputs the *actual* target xyz used for IK (after offset+clamp).
    """
    def __init__(self):
        super().__init__()
        self._box_state_port = self.DeclareVectorInputPort(name="box_state", size=13)
        self.DeclareVectorOutputPort("target_xyz", size=3, calc=self.calc_output)

    def calc_output(self, context, output):
        if USE_BOX_TARGET:
            box_state = self._box_state_port.Eval(context)
            box_pos = np.array(box_state[4:7], dtype=float)
            target = box_pos + BOX_TARGET_OFFSET_W
        else:
            target = np.array(FIXED_TARGET_W, dtype=float)
        target[2] = max(float(target[2]), float(MIN_IK_TARGET_Z))
        output.SetFromVector(target.tolist())

######################################################################################################
#                                  ########IK FK Debug: q -> panda_hand xyz#######
######################################################################################################
class FKBodyPositionFromQ(LeafSystem):
    """
    Debug helper:
      Input: joint positions q (size nq) for the panda model instance
      Output: world-frame xyz of the requested body (e.g., panda_hand)
    This lets us turn IK joint targets (q_r) into an end-effector position for plotting.
    """
    def __init__(self, plant, model_instance, body_name: str, nq: int):
        super().__init__()
        self.plant = plant
        self.model_instance = model_instance
        self.body = plant.GetBodyByName(body_name, model_instance)
        self.nq = int(nq)
        self.DeclareVectorInputPort("q", size=self.nq)
        self.DeclareVectorOutputPort("body_world_positions", size=3, calc=self.calc_output)
        self._context = plant.CreateDefaultContext()

    def calc_output(self, context, output):
        q = np.array(self.GetInputPort("q").Eval(context), dtype=float)
        # plant expects positions sized to model_instance
        self.plant.SetPositions(self._context, self.model_instance, q[: self.nq])
        pose = self.plant.EvalBodyPoseInWorld(self._context, self.body)
        p = pose.translation()
        output.SetFromVector([p[0], p[1], p[2]])

######################################################################################################
#              ##################Planner Trapezoidal motion profile ################
######################################################################################################
# DELETED - Not using motion profile, using RelaxedIK instead

######################################################################################################
#                       ################## Trajectory-Based ERG ################
######################################################################################################
class ERG(LeafSystem):
    def __init__(self, robot_spec, nq, nv):
        super().__init__()  # Don't forget to initialize the base class.

        self.nq = nq
        self.nv = nv
        self.robot_spec = robot_spec

        self._state_port = self.DeclareVectorInputPort(name="state", size=self.nq + self.nv)
        self._tau_port = self.DeclareVectorInputPort(name="tau", size=self.nq)
        self._qr_port = self.DeclareVectorInputPort(name="q_r", size=self.nq)
        self._box_state_port = self.DeclareVectorInputPort(name="box_state", size=13)

        state_index = self.DeclareDiscreteState(self.nq)
        self.DeclareStateOutputPort("q_v_filtered", state_index)
        
        # Add output port for calculated energy
        self.DeclareVectorOutputPort("calculated_energy", size=1, calc=self.output_energy)

        # Add output port for DSM (dynamic safety margin)
        self.DeclareVectorOutputPort("dsm", size=1, calc=self.output_dsm)
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.refrence) # Call the Update method defined below.

        erg_params = ErgParams()
        # Reuse the shared contact parameters so ERG matches plant setup
        contact_params = CONTACT_PARAMS
        self.erg = CompliantERG(
            robot_spec=self.robot_spec,
            erg_params=erg_params,
            contact_params=contact_params)

        # Initialize a flag to check if it's the first update
        self.first_update = True
        # Seed with q0 sized to nq; pad/trim if needed
        q0 = trajInit_
        if len(q0) != self.nq:
            if len(q0) > self.nq:
                q0 = q0[: self.nq]
            else:
                q0 = np.pad(q0, (0, self.nq - len(q0)), mode="constant")
        self.q_v_ = q0.copy()
        self.last_dsm = 0.0

    def refrence(self, context, discrete_state):
        # Evaluate the input ports
        state = self._state_port.Eval(context)
        q = state[: self.nq]
        dq = state[self.nq :]
        tau = self._tau_port.Eval(context)
        q_r = self._qr_port.Eval(context)
        box_state = self._box_state_port.Eval(context)
        # Box position (world). Keep consistent with the IK target definition.
        box_position = np.array(box_state[4:7], dtype=float) + BOX_TARGET_OFFSET_W

        self.q_v = context.get_discrete_state_vector().CopyToVector()              
        # Initialize q_v_ only at the first callback
        if self.first_update:
            self.q_v_ = self.q_v_  # already sized to nq
            self.first_update = False
        
        # Update box position in ERG
        self.q_v_ = self.erg.get_qv(q, dq, tau, q_r, self.q_v_, box_position)

        # Capture DSM for logging (stored inside CompliantERG during get_qv)
        dsm_val = getattr(self.erg, "last_dsm", None)
        try:
            self.last_dsm = float(dsm_val) if dsm_val is not None else 0.0
        except Exception:
            self.last_dsm = 0.0
        
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

    def output_dsm(self, context, output):
        # Return latest DSM from the ERG system
        try:
            output.SetAtIndex(0, float(getattr(self, "last_dsm", 0.0)))
        except Exception:
            output.SetAtIndex(0, 0.0)


######################################################################################################
#                                  ########PD+G controller#######   Check input output
######################################################################################################
class PD_gravity_local(LeafSystem):
    def __init__(self, plant, plant_context, panda_id, robot_spec):
        super().__init__()

        self.plant = plant
        self.plant_context = plant_context
        self.panda_id = panda_id

        # Size according to the robot in the plant
        self.nq = plant.num_positions(panda_id)
        self.nv = plant.num_velocities(panda_id)
        self.nx = self.nq + self.nv

        # Gains sized to the robot spec
        self.Kp_ = np.array(robot_spec.kp, dtype=float)[: self.nq]
        self.Kd_ = np.array(robot_spec.kd, dtype=float)[: self.nq]

        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=self.nq)
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=self.nx)

        state_index = self.DeclareDiscreteState(self.nq)
        self.DeclareStateOutputPort("tau_u", state_index)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1 / 1000,
            offset_sec=0.0,
            update=self.compute_tau_u,
        )
        
    def compute_tau_u(self, context, discrete_state):
        # Evaluate the input ports
        q_d = self._desired_state_port.Eval(context)
        state = self._current_state_port.Eval(context)
        q = state[: self.nq]
        dq = state[self.nq :]
        
        # Gravity compensation (safely sized to nq)
        gravity_full = -self.plant.CalcGravityGeneralizedForces(self.plant_context)
        gravity_full_np = np.array(gravity_full).flatten()
        if gravity_full_np.shape[0] >= self.nq:
            gravity = gravity_full_np[: self.nq]
        else:
            gravity = np.zeros(self.nq)
        
        tau = self.Kp_ * (q_d - q) - self.Kd_ * dq
        tau = tau + gravity
        
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
            print("No contacts detected")
            return
        
        # Get number of contacts
        num_contacts = contact_results.num_point_pair_contacts()
        print(f"Total contacts detected: {num_contacts}")
        
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
            
            print(f"Contact {i}: {bodyA_name} <-> {bodyB_name}")
            
            # Check if this contact is between any robot link and box
            if ((bodyA_name == object_name or bodyB_name == object_name) and
                (bodyA_name in robot_links or bodyB_name in robot_links)):
                
                robot_box_contacts += 1
                force = contact_info.contact_force()
                print(f"  Robot-Box contact detected! Force: [{force[0]:.3f}, {force[1]:.3f}, {force[2]:.3f}]")
                
                # Categorize contact type
                contact_body = bodyA_name if bodyA_name in robot_links else bodyB_name
                if "finger" in contact_body:
                    finger_contacts += 1
                    print(f"    -> Finger contact detected on {contact_body}")
                elif contact_body == "panda_hand":
                    hand_contacts += 1
                    print(f"    -> Hand contact detected on {contact_body}")
                else:
                    arm_contacts += 1
                    print(f"    -> Arm link contact detected on {contact_body}")
                
                # Extract contact force and add to total
                total_force[0] += force[0]
                total_force[1] += force[1]
                total_force[2] += force[2]
        
        print(f"Robot-Box contacts: {robot_box_contacts}")
        print(f"  - Arm link contacts: {arm_contacts}")
        print(f"  - Hand contacts: {hand_contacts}")
        print(f"  - Finger contacts: {finger_contacts}")
        print(f"Total force: [{total_force[0]:.3f}, {total_force[1]:.3f}, {total_force[2]:.3f}]")
        
        # Set the total contact force
        output.SetFromVector(total_force)

######################################################################################################
#                                  ##################################
######################################################################################################
######################################################################################################
#                                  ########Box Position Extractor#######
######################################################################################################
# Get the movable box model instance ID for state wiring/logging
box_id = movable_box_id

# Create systems
erg_system = builder.AddNamedSystem("Trajectory-based ERG", ERG(ROBOT_SPEC, num_positions, num_velocities))
# Use the same PD+gravity controller implementation as `simple_relaxed_ik_test.py`
pid_controller = builder.AddNamedSystem(
    "PD+G controller",
    SharedPDGravity(plant, panda_id, kp=ROBOT_SPEC.kp, kd=ROBOT_SPEC.kd),
)

# Add new systems for IK-based control
# Remove the custom box position extractor and use plant state output directly
ik_box_tracker = builder.AddNamedSystem("Relaxed IK Box Tracker", RelaxedIKBoxTracker(plant, plant_context, ROBOT_SPEC))

# Add contact force converter system
contact_force_converter = builder.AddNamedSystem("Contact Force Converter", ContactForceConverter(plant))

# Add box target extractor for logging/plotting
box_target_extractor = builder.AddNamedSystem("BoxTargetExtractor", BoxTargetExtractor())

#
# NOTE: We now spawn a real movable box, so we wire its true 13D free-body state.
#

# Add FK debug: IK joint targets -> panda_hand xyz
ik_fk_hand_extractor = builder.AddNamedSystem(
    "IKFKHandExtractor",
    FKBodyPositionFromQ(plant, panda_id, body_name="panda_hand", nq=ROBOT_SPEC.num_positions),
)

builder.Connect(
    plant.get_state_output_port(box_id),
    ik_box_tracker.GetInputPort("box_state"),
)

builder.Connect(
    plant.get_state_output_port(box_id),
    box_target_extractor.GetInputPort("box_state"),
)

builder.Connect(
    plant.get_state_output_port(box_id),
    erg_system.GetInputPort("box_state"),
)

# Connect contact results to contact force converter
builder.Connect(plant.get_contact_results_output_port(),
               contact_force_converter.GetInputPort("contact_results"))

# Connect IK system to ERG (replacing the trajectory)
builder.Connect(ik_box_tracker.GetOutputPort("ik_joint_targets"), 
               erg_system.GetInputPort("q_r"))

# Connect IK joint targets to FK extractor (debug)
builder.Connect(
    ik_box_tracker.GetOutputPort("ik_joint_targets"),
    ik_fk_hand_extractor.GetInputPort("q"),
)

# Keep the original connections for the rest of the system
builder.Connect(plant.get_state_output_port(panda_id), pid_controller.GetInputPort("Current_state"))
if USE_ERG_FILTERED_REFERENCE:
    builder.Connect(erg_system.GetOutputPort("q_v_filtered"), pid_controller.GetInputPort("Desired_state"))
else:
    builder.Connect(ik_box_tracker.GetOutputPort("ik_joint_targets"), pid_controller.GetInputPort("Desired_state"))
builder.Connect(pid_controller.GetOutputPort("tau_u"), plant.get_actuation_input_port(panda_id))
builder.Connect(plant.get_state_output_port(panda_id), erg_system.GetInputPort("state"))
builder.Connect(plant.GetOutputPort("panda_net_actuation"), erg_system.GetInputPort("tau"))


# Connect to visualizer
if meshcat_visualisation:
    meshcat = StartMeshcat()
    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    print(f"MeshCat visualization available at: {meshcat.web_url()}")

# Add Panda link7 pose extractor (generic, sized to the robot)
panda_link7_extractor = builder.AddNamedSystem(
    "PandaLink7PoseExtractor",
    BodyPoseExtractor(plant, panda_id, body_name="panda_link7", robot_spec=ROBOT_SPEC),
)

# Add Panda link6 pose extractor (world position)
panda_link6_extractor = builder.AddNamedSystem(
    "PandaLink6PoseExtractor",
    BodyPoseExtractor(plant, panda_id, body_name="panda_link6", robot_spec=ROBOT_SPEC),
)

# Add Panda hand pose extractor (world position)
panda_hand_extractor = builder.AddNamedSystem(
    "PandaHandPoseExtractor",
    BodyPoseExtractor(plant, panda_id, body_name="panda_hand", robot_spec=ROBOT_SPEC),
)

# Add Panda hand TCP pose extractor (world position) if present in this URDF
try:
    panda_hand_tcp_extractor = builder.AddNamedSystem(
        "PandaHandTcpPoseExtractor",
        BodyPoseExtractor(plant, panda_id, body_name="panda_hand_tcp", robot_spec=ROBOT_SPEC),
    )
except Exception:
    panda_hand_tcp_extractor = None

# Add rubber pad pose extractor (world position) if present in this URDF.
# Different URDF variants use different names, so try a small set.
_rubber_pad_name = None
for _cand in ["rubber_pad", "panda_rubber_pad", "gripper_pad"]:
    try:
        panda_rubber_pad_extractor = builder.AddNamedSystem(
            f"PandaRubberPadPoseExtractor_{_cand}",
            BodyPoseExtractor(plant, panda_id, body_name=_cand, robot_spec=ROBOT_SPEC),
        )
        _rubber_pad_name = _cand
        break
    except Exception:
        panda_rubber_pad_extractor = None

# Connect robot joint states to PandaLink7PoseExtractor
builder.Connect(
    plant.get_state_output_port(panda_id),
    panda_link7_extractor.GetInputPort("joint_positions"),
)

# Connect robot joint states to PandaLink6PoseExtractor
builder.Connect(
    plant.get_state_output_port(panda_id),
    panda_link6_extractor.GetInputPort("joint_positions"),
)

# Connect robot joint states to PandaHandPoseExtractor
builder.Connect(
    plant.get_state_output_port(panda_id),
    panda_hand_extractor.GetInputPort("joint_positions"),
)

if panda_hand_tcp_extractor is not None:
    builder.Connect(
        plant.get_state_output_port(panda_id),
        panda_hand_tcp_extractor.GetInputPort("joint_positions"),
    )

if panda_rubber_pad_extractor is not None:
    builder.Connect(
        plant.get_state_output_port(panda_id),
        panda_rubber_pad_extractor.GetInputPort("joint_positions"),
    )

logger_signals = [
    LogSignal("state", plant.get_state_output_port(panda_id)),
    LogSignal("tau", pid_controller.GetOutputPort("tau_u")),
    LogSignal("q_v", erg_system.GetOutputPort("q_v_filtered")),
    LogSignal("q_r", ik_box_tracker.GetOutputPort("ik_joint_targets")),
    # LogSignal("box", plant.get_state_output_port(box_id)),  # COMMENTED OUT: no movable box
    LogSignal("energy", erg_system.GetOutputPort("calculated_energy")),
    LogSignal("dsm", erg_system.GetOutputPort("dsm")),
    LogSignal("contact", contact_force_converter.GetOutputPort("contact_forces")),
    LogSignal("link_pos", panda_link7_extractor.GetOutputPort("body_world_positions")),
    LogSignal("link6_pos", panda_link6_extractor.GetOutputPort("body_world_positions")),
    LogSignal("hand_pos", panda_hand_extractor.GetOutputPort("body_world_positions")),
    LogSignal("box_target_pos", box_target_extractor.GetOutputPort("target_xyz")),
    LogSignal("ik_hand_pos", ik_fk_hand_extractor.GetOutputPort("body_world_positions")),
]

if panda_hand_tcp_extractor is not None:
    logger_signals.append(LogSignal("tcp_pos", panda_hand_tcp_extractor.GetOutputPort("body_world_positions")))

if panda_rubber_pad_extractor is not None:
    logger_signals.append(LogSignal("pad_pos", panda_rubber_pad_extractor.GetOutputPort("body_world_positions")))

mux_logger = add_mux_logger(builder, logger_signals, time_step, name="mux_logger")


# Finalize the diagram
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

# --- IMPORTANT init to avoid t=0 torque spike ---
# Drake discrete states default to zeros unless explicitly initialized.
# At t=0, the PD controller reads Desired_state from ERG output port `q_v_filtered`
# (a StateOutputPort backed by ERG discrete state). If that state is still zeros,
# the initial position error (q_d - q) can be huge => huge torque spike.
try:
    _erg_ctx = erg_system.GetMyMutableContextFromRoot(diagram_context)
    _erg_ctx.get_mutable_discrete_state_vector().SetFromVector(trajInit_.tolist())
except Exception as _e:
    print(f"[WARN] Failed to initialize ERG discrete state (q_v) at t=0: {_e}")

# Similarly initialize the IK tracker output state to a reasonable fallback (not zeros).
# try:
#     _ik_ctx = ik_box_tracker.GetMyMutableContextFromRoot(diagram_context)
#     _ik_ctx.get_mutable_discrete_state_vector().SetFromVector(trajInit_.tolist())
# except Exception as _e:
#     print(f"[WARN] Failed to initialize IK discrete state (q_r) at t=0: {_e}")

####################################
# Run Simple Simulation
####################################
if simulate:
    simulator = Simulator(diagram, diagram_context)
    simulator_context = simulator.get_mutable_context()
    plant_context_sim = plant.GetMyMutableContextFromRoot(simulator_context)
    plant.SetPositions(plant_context_sim, panda_id, trajInit_)
    simulator.set_target_realtime_rate(realtime_factor)
    simulator.set_publish_every_time_step(True)
    simulator.Initialize()

    # Debug: print the *actual* initial box pose from the simulator context.
    try:
        box_body = plant.GetBodyByName("box_link", movable_box_id)
        p_WB = plant.EvalBodyPoseInWorld(plant_context_sim, box_body).translation()
        print(f"INIT box_link p_WB = [{p_WB[0]:.3f}, {p_WB[1]:.3f}, {p_WB[2]:.3f}]")
    except Exception as _e:
        print(f"[WARN] Failed to read initial box pose from simulator context: {_e}")

    # Define the step size and simulation time
    kStep = 0.001
    # Allow quick debug runs without editing code:
    #   SIM_TIME=4 python Dual_arm_art/test_erg.py
    sim_time = float(os.environ.get("SIM_TIME", "4.0"))  # Fixed default since no trajectory duration
    print(f"Starting simulation for {sim_time} seconds...")
    print(f"Initial` positions: {trajInit_}")
    print("Using Relaxed IK to track box position!")

    # Run simulation
    while simulator_context.get_time() < sim_time:
         next_time = min(sim_time, simulator_context.get_time() + kStep)
         simulator.AdvanceTo(next_time)

    print("Simulation completed!")
    
    # Evaluate and print the final pose of the movable box

    
    # Record and publish MeshCat visualization
    if meshcat_visualisation:
        print("Recording simulation for MeshCat replay...")
        meshcat.StartRecording()
        simulator.AdvanceTo(sim_time)  # Adjust this time as needed
        meshcat.PublishRecording()
        print("MeshCat recording published!")
        
        # Save HTML recording
        html_path = os.path.join(os.path.dirname(__file__), "meshcat_recording_erg.html")
        html_data = meshcat.StaticHtml()
        print("Recording size (characters):", len(html_data))
        
        if len(html_data) > 0:
            with open(html_path, "w") as f:
                f.write(html_data)
            print(f"Recording saved to: {html_path}")
        else:
            print("MeshCat recording appears to be empty. Did any geometry move?")


state_raw, t_state = mux_logger.get(diagram_context, "state")
tau_raw, t_tau = mux_logger.get(diagram_context, "tau")
qv_raw, t_qv = mux_logger.get(diagram_context, "q_v")
qr_raw, t_qr = mux_logger.get(diagram_context, "q_r")
# box_raw, t_box = mux_logger.get(diagram_context, "box")  # COMMENTED OUT: no movable box
energy_raw, t_energy = mux_logger.get(diagram_context, "energy")
dsm_raw, t_dsm = mux_logger.get(diagram_context, "dsm")
contact_raw, t_contact = mux_logger.get(diagram_context, "contact")
link7_raw, t_link7 = mux_logger.get(diagram_context, "link_pos")
link6_raw, t_link6 = mux_logger.get(diagram_context, "link6_pos")
hand_raw, t_hand = mux_logger.get(diagram_context, "hand_pos")
box_target_raw, t_box_target = mux_logger.get(diagram_context, "box_target_pos")
ik_hand_raw, t_ik_hand = mux_logger.get(diagram_context, "ik_hand_pos")

tcp_raw = t_tcp = None
pad_raw = t_pad = None
if panda_hand_tcp_extractor is not None:
    tcp_raw, t_tcp = mux_logger.get(diagram_context, "tcp_pos")
if panda_rubber_pad_extractor is not None:
    pad_raw, t_pad = mux_logger.get(diagram_context, "pad_pos")

# Transpose to shape (samples, dim)
state_data = state_raw.T
tau_data = tau_raw.T
qv_data = qv_raw.T
qr_data = qr_raw.T
# box_data = box_raw.T  # COMMENTED OUT: no movable box
energy_data = energy_raw.T
dsm_data = dsm_raw.T
contact_data = contact_raw.T
link7_data = link7_raw.T
link6_data = link6_raw.T
hand_data = hand_raw.T
box_target_data = box_target_raw.T
ik_hand_data = ik_hand_raw.T
tcp_data = tcp_raw.T if tcp_raw is not None else None
pad_data = pad_raw.T if pad_raw is not None else None

# Print final joint positions if available (sized to robot)
if state_data.shape[0] > 0:
    final_q = state_data[-1, :ROBOT_SPEC.num_positions]
    print("\n=== Final Joint Positions (sized to robot) ===")
    print(final_q)

logs = {
    "state": (t_state, state_data),
    "tau": (t_tau, tau_data),
    "qv": (t_qv, qv_data),
    "qr": (t_qr, qr_data),
    # "box": (t_box, box_data),  # COMMENTED OUT: no movable box
    "energy": (t_energy, energy_data),
    "dsm": (t_dsm, dsm_data),
    "contact": (t_contact, contact_data),
    "link_pos": (t_link7, link7_data),
    "link6_pos": (t_link6, link6_data),
    "hand_pos": (t_hand, hand_data),
    "box_target_pos": (t_box_target, box_target_data),
    "ik_hand_pos": (t_ik_hand, ik_hand_data),
}

if tcp_data is not None and t_tcp is not None:
    logs["tcp_pos"] = (t_tcp, tcp_data)
if pad_data is not None and t_pad is not None:
    logs["pad_pos"] = (t_pad, pad_data)

plot_results(logs, ROBOT_SPEC, show=True)
