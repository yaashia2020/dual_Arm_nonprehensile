import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
import csv  
from pydrake.all import *
import pydot
from IPython.display import SVG, display
from trajectoryERG import ExplicitReferenceGovernor
import os
from pydrake.visualization import AddDefaultVisualization

# Add Relaxed IK wrapper import
import sys
wrapper_dir = "/home/yaashia/dual_arm_nonprehensile/relaxed_ik_core/wrappers"
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
    urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_fr3.urdf"))
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
    initial_box_position = RigidTransform(p=[0.7, 0.0, 0.2])  # Positioned just above the fixed box
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("box_link", movable_box_id), initial_box_position)
    
    # Set default positions for the robot (not the box)
    panda_id = plant.GetModelInstanceByName("panda")
    plant.SetDefaultPositions(panda_id, [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0])
    
    return plant, scene_graph, plant_context
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
plant, scene_graph, plant_context = create_system_model(plant, scene_graph)
# Set the initial joint position of the robot otherwise it will correspond to zero positions
# plant_context = plant.CreateDefaultContext()  # This is now returned from create_system_model

# Get the panda model instance for specific connections
panda_id = plant.GetModelInstanceByName("panda")
num_positions = plant.num_positions(panda_id)
num_velocities = plant.num_velocities(panda_id)

######################################################################################################
#              ##################Relaxed IK System for Box Tracking################
######################################################################################################
class RelaxedIKBoxTracker(LeafSystem):
    def __init__(self):
        super().__init__()
        
        # Declare input port for box position
        self._box_position_port = self.DeclareVectorInputPort(name="box_position", size=3)
        
        state_index = self.DeclareDiscreteState(9)  # 9 joint angles
        self.DeclareStateOutputPort("ik_joint_targets", state_index)
        
        # Periodic update for IK solving
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # 10Hz IK updates
            offset_sec=0.0,
            update=self.solve_ik)
        
        # Initialize Relaxed IK
        try:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_settings.yaml"))
            self.rik = RelaxedIKRust(setting_file_path=config_path)
            print("Relaxed IK initialized successfully for box tracking!")
            self.ik_available = True
        except Exception as e:
            print(f"Failed to initialize Relaxed IK: {e}")
            self.ik_available = False
        
        # Tolerance for IK solving
        self.tolerance = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]  # x,y,z and rx,ry,rz tolerances
        
        # Default orientation (quaternion xyzw)
        self.default_orientation = [0.0, 0.0, 0.0, 1.0]
        
        # Fallback joint configuration if IK fails
        self.fallback_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]

    def solve_ik(self, context, discrete_state):
        if not self.ik_available:
            # Use fallback if IK is not available
            discrete_state.get_mutable_vector().SetFromVector(self.fallback_joints)
            return
        
        try:
            # Get box position from input port
            box_pos = self._box_position_port.Eval(context)
            
            # Use box position directly as target (no offset)
            target_position = [box_pos[0], box_pos[1], box_pos[2]]
            
            try:
                # Solve IK with default orientation
                joint_solution = self.rik.solve_position(target_position, self.default_orientation, self.tolerance)
                    
            except Exception as e:
                print(f"IK solving failed: {e}")
                discrete_state.get_mutable_vector().SetFromVector(self.fallback_joints)
                return
            
            # Extend to 9 joints (add gripper joints)
            if len(joint_solution) == 7:
                full_solution = list(joint_solution) + [0.0, 0.0]  # Add gripper joints
            else:
                full_solution = joint_solution
            
            # Ensure we have exactly 9 joints
            if len(full_solution) != 9:
                full_solution = self.fallback_joints
            
            # Update the discrete state
            discrete_state.get_mutable_vector().SetFromVector(full_solution)
            
            print(f"Box position: {target_position}")
            print(f"IK solution: {full_solution}")
            
        except Exception as e:
            print(f"IK solving failed: {e}")
            # Use fallback configuration
            discrete_state.get_mutable_vector().SetFromVector(self.fallback_joints)

######################################################################################################
#              ##################Planner Trapezoidal motion profile ################
######################################################################################################
trajDuration_ = 3.5
accDuration_ = 2.5

trajInit_ = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]) # np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0])
trajEnd_ = np.array([3.0, -0.785, -0.50, -1.6, 0.7, 1.571, 0.785, 0.0, 0.0]) # np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0])
class TrajectoryPoint:
    def __init__(self):
        self.pos = np.zeros(9)
        self.vel = np.zeros(9)
        self.acc = np.zeros(9)

class motion_profile(LeafSystem):
    def __init__(self):
        super().__init__()  # Don't forget to initialize the base class.
        self._trajInit_port = self.DeclareVectorInputPort(name="trajInit_", size=9)
        self._trajEnd__port = self.DeclareVectorInputPort(name="trajEnd_", size=9)
        # self.DeclareVectorOutputPort(name="q_r", size=9, calc=self.compute_trajectory) 

        state_index = self.DeclareDiscreteState(9)  # One state variable.
        self.DeclareStateOutputPort("q_r", state_index)  # One output: y=x.
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.compute_trajectory) # Call the Update method defined below.
        
        self.traj = TrajectoryPoint()

    def compute_trajectory(self, context, discrete_state):
        # Evaluate the input ports
        self.ttime = context.get_time()
        self.trajInit_ =self._trajInit_port.Eval(context)
        self.trajEnd_ =self._trajEnd__port.Eval(context)

        ddot_traj_c = -1.0 / (accDuration_**2 - trajDuration_ * accDuration_) * (self.trajEnd_ - self.trajInit_)

        if self.ttime <= accDuration_:
            self.traj.pos = self.trajInit_ + 0.5 * ddot_traj_c * self.ttime**2
            self.traj.vel = ddot_traj_c * self.ttime
            self.traj.acc = ddot_traj_c
        elif self.ttime <= trajDuration_ - accDuration_:
            self.traj.pos = self.trajInit_ + ddot_traj_c * accDuration_ * (self.ttime - accDuration_ / 2)
            self.traj.vel = ddot_traj_c * accDuration_
            self.traj.acc = np.zeros(3)
        elif self.ttime <= trajDuration_:
            self.traj.pos = self.trajEnd_ - 0.5 * ddot_traj_c * (trajDuration_ - self.ttime)**2
            self.traj.vel = ddot_traj_c * (trajDuration_ - self.ttime)
            self.traj.acc = -ddot_traj_c
        else:
            # After trajDuration_, hold the final position
            self.traj.pos = trajEnd_
            self.traj.vel = np.zeros(9)
            self.traj.acc = np.zeros(9)
        
        # q_r = self.traj.pos
        q_r = trajEnd_
        # print(f"refrence = \n {q_r}")

        # Write into the output vector.
        discrete_state.get_mutable_vector().SetFromVector(q_r)


######################################################################################################
#                       ################## Trajectory-Based ERG ################
######################################################################################################
class ERG(LeafSystem):
    def __init__(self):
        super().__init__()  # Don't forget to initialize the base class.
        self._state_port = self.DeclareVectorInputPort(name="state", size=18)
        self._tau_port = self.DeclareVectorInputPort(name="tau", size=9)
        self._qr_port = self.DeclareVectorInputPort(name="q_r", size=9)

        state_index = self.DeclareDiscreteState(9)  # One state variable.
        self.DeclareStateOutputPort("q_v_filtered", state_index)  # One output: y=x.
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.refrence) # Call the Update method defined below.
        self.erg = ExplicitReferenceGovernor(
            robust_delta_tau_=0.1, kappa_tau_=1.0,
            robust_delta_q_=0.1, kappa_q_=15.0, robust_delta_dq_=0.1, kappa_dq_=7.0,
            robust_delta_dp_EE_=0.01, kappa_dp_EE_=7.0, kappa_terminal_energy_=7.5)

        # Initialize a flag to check if it's the first update
        self.first_update = True

    def refrence(self, context, discrete_state):
        # Evaluate the input ports
        state = self._state_port.Eval(context)
        q = state[:num_positions]
        dq = state[num_positions:]
        tau = self._tau_port.Eval(context)
        q_r = self._qr_port.Eval(context)

        # print(f"tau: \n {tau}")
        print(f"q: \n {q}")
        self.q_v = context.get_discrete_state_vector().CopyToVector()              
        # Initialize q_v_ only at the first callback
        if self.first_update:
            self.q_v_ = trajInit_  # or an appropriate initial value    
            self.first_update = False
        self.q_v_ = self.erg.get_qv(q, dq, tau, q_r, self.q_v_)

        # Write into the output vector.
        # print(f"filtered ref = \n {self.q_v_}")
        discrete_state.get_mutable_vector().SetFromVector(self.q_v_)


######################################################################################################
#                                  ########PD+G controller#######   Check input output
######################################################################################################
class PD_gravity(LeafSystem):
    def __init__(self):
        super().__init__()

        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=9)
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=18)
        self.Kp_ = [120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0, 120, 120]
        self.Kd_ = [8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0, 5, 5]

        state_index = self.DeclareDiscreteState(9)  # One state variable.
        self.DeclareStateOutputPort("tau_u", state_index)  # One output: y=x.
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/1000,  # One second time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.compute_tau_u) # Call the Update method defined below.
        
    def compute_tau_u(self, context, discrete_state):
        # Evaluate the input ports
        self.q_d =self._desired_state_port.Eval(context)
        self.q = self._current_state_port.Eval(context)
        # x = context.get_discrete_state_vector().GetAtIndex(0)
        
        # Get gravity for entire plant and extract panda portion
        gravity_full = -plant.CalcGravityGeneralizedForces(plant_context)
        # Convert to numpy array and extract first 9 elements for robot
        gravity_full_np = np.array(gravity_full).flatten()
        print(f"gravity_full shape: {gravity_full_np.shape}")
        gravity = gravity_full_np[:9]  # Force to 9 elements for robot
        print(f"gravity shape after slicing: {gravity.shape}")
        
        tau = self.Kp_ * (self.q_d - self.q[:num_positions]) - self.Kd_ * self.q[num_positions:]
        tau = tau + gravity
        
        # print(f"applied torque = \n {tau}")
        discrete_state.get_mutable_vector().SetFromVector(tau)

######################################################################################################
#                                  ##################################
######################################################################################################
######################################################################################################
#                                  ########Box Position Extractor#######
######################################################################################################
class BoxPositionExtractor(LeafSystem):
    def __init__(self, plant, plant_context, box_id):
        super().__init__()
        
        # Store references
        self.plant = plant
        self.plant_context = plant_context
        self.box_id = box_id
        
        # Output port for box position
        self.DeclareVectorOutputPort("box_position", size=3, calc=self.extract_box_position)
    
    def extract_box_position(self, context, output):
        try:
            # Get the box body
            box_body = self.plant.GetBodyByName("box_link", self.box_id)
            
            # Get the pose of the box body in world frame
            X_WB = self.plant.EvalBodyPoseInWorld(self.plant_context, box_body)
            
            # Extract position (translation)
            position = X_WB.translation()
            pos = [position[0], position[1], position[2]]
            
            output.SetFromVector(pos)
            
        except Exception as e:
            print(f"Error extracting box position: {e}")
            # Return default position if extraction fails
            output.SetFromVector([0.7, 0.0, 0.7])

######################################################################################################
#                                  ##################################
######################################################################################################
# Get the box model instance ID for the position extractor
box_id = plant.GetModelInstanceByName("movable_box")

# Create systems
init_pos = builder.AddNamedSystem("Initial position", ConstantVectorSource(trajInit_))
end_pos = builder.AddNamedSystem("End position", ConstantVectorSource(trajEnd_))
trajectory = builder.AddNamedSystem("Trajectory generator", motion_profile())
erg_system = builder.AddNamedSystem("Trajectory-based ERG", ERG())
pid_controller = builder.AddNamedSystem("PD+G controller", PD_gravity())

# Add new systems for IK-based control
box_position_extractor = builder.AddNamedSystem("Box Position Extractor", 
                                               BoxPositionExtractor(plant, plant_context, box_id))
ik_box_tracker = builder.AddNamedSystem("Relaxed IK Box Tracker", RelaxedIKBoxTracker())

# Connect box position extractor to IK system
builder.Connect(box_position_extractor.GetOutputPort("box_position"), 
               ik_box_tracker.GetInputPort("box_position"))

# Connect trajectory generator inputs
builder.Connect(init_pos.GetOutputPort("y0"), trajectory.GetInputPort("trajInit_"))
builder.Connect(end_pos.GetOutputPort("y0"), trajectory.GetInputPort("trajEnd_"))

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
    print(f"MeshCat visualization available at: {meshcat.web_url()}")

logger_x = LogVectorOutput(plant.get_state_output_port(), builder) #state
logger_tau = LogVectorOutput(pid_controller.GetOutputPort("tau_u"), builder) #tau_u
logger_qv = LogVectorOutput(erg_system.GetOutputPort("q_v_filtered"), builder) #q_v
logger_qr = LogVectorOutput(ik_box_tracker.GetOutputPort("ik_joint_targets"), builder) #ik targets
logger_box_pos = LogVectorOutput(box_position_extractor.GetOutputPort("box_position"), builder) #box position

# Finalize the diagram
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

######################################################################################################
#                                  ########Simple Box Tracker#######
######################################################################################################
def get_box_pose_in_world(plant, plant_context, box_id, collect_data=True, box_positions=None, box_orientations=None):
    """
    Simple function to get box position and orientation in world frame using EvalBodyPoseInWorld.
    
    Args:
        plant: The MultibodyPlant object
        plant_context: The plant context
        box_id: The box model instance ID
        collect_data: Whether to collect data for plotting
        box_positions: List to store position data
        box_orientations: List to store orientation data
        
    Returns:
        tuple: (position, orientation) where position is [x, y, z] and orientation is [roll, pitch, yaw]
    """
    # Get the box body
    box_body = plant.GetBodyByName("box_link", box_id)
    
    # Get the pose of the box body in world frame using Drake's EvalBodyPoseInWorld
    X_WB = plant.EvalBodyPoseInWorld(plant_context, box_body)
    
    # Extract position (translation)
    position = X_WB.translation()
    pos = [position[0], position[1], position[2]]
    
    # Extract orientation (rotation) as roll, pitch, yaw
    rotation = X_WB.rotation()
    roll_pitch_yaw = rotation.ToRollPitchYaw()
    rpy = roll_pitch_yaw.vector()
    orientation = [rpy[0], rpy[1], rpy[2]]
    
    # Collect data if requested
    if collect_data and box_positions is not None and box_orientations is not None:
        box_positions.append(pos)
        box_orientations.append(orientation)
    
    return pos, orientation

####################################
# Run Simple Simulation
####################################
if simulate:
    simulator = Simulator(diagram, diagram_context)
    plant.SetPositions(plant_context, panda_id, trajInit_)
    simulator.set_target_realtime_rate(realtime_factor)
    simulator.set_publish_every_time_step(True)
    simulator.Initialize()

    # Define the step size and simulation time
    kStep = 0.001
    sim_time = trajDuration_  # or whatever your total simulation time is
    simulator_context = simulator.get_mutable_context()

    # Lists to store box position data
    box_positions = []
    box_orientations = []

    print(f"Starting simulation for {sim_time} seconds...")
    print(f"Initial positions: {trajInit_}")
    print("Using Relaxed IK to track box position!")

    # Run simulation
    while simulator_context.get_time() < sim_time:
         next_time = min(sim_time, simulator_context.get_time() + kStep)
         simulator.AdvanceTo(next_time)
         
         # Track box position at each step using Drake's EvalBodyPoseInWorld
         current_time = simulator_context.get_time()
         pos, orientation = get_box_pose_in_world(plant, plant_context, box_id, collect_data=True, box_positions=box_positions, box_orientations=box_orientations)

    print("Simulation completed!")
    
    # Convert to numpy arrays for easier plotting
    box_positions = np.array(box_positions)
    box_orientations = np.array(box_orientations)

    # Print initial and final box positions
    print(f"Initial box position: {box_positions[0]}")
    print(f"Final box position: {box_positions[-1]}")
    print(f"Initial box orientation: {box_orientations[0]}")
    print(f"Final box orientation: {box_orientations[-1]}")
    
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

t_time = log_x.sample_times()

data_q = log_x.data().transpose()[:, 0:9]  # Selecting only the first 7 columns (joints)
data_qdot = log_x.data().transpose()[:, 9:18]  # Selecting only the first 7 columns (joints)
data_tau = log_tau.data().transpose()  # Selecting only the first 7 columns (joints)
data_qv = log_qv.data().transpose()  # Selecting only the first 7 columns (joints)
data_qr = log_qr.data().transpose()  # Selecting only the first 7 columns (joints)
data_box_pos = log_box_pos.data().transpose() # box position

# print(t_time)
# Identify modified joints (where values differ)
modified_indices = np.where(trajInit_ != trajEnd_)[0]

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

# Create a figure for box position plots
fig_box_pos, axs_box_pos = plt.subplots(1, 1, figsize=(12, 3))
fig_box_pos.suptitle('Box Position')

axs_box_pos.plot(t_time, data_box_pos[:, 0], label='Box X Position', linestyle='-')
axs_box_pos.plot(t_time, data_box_pos[:, 1], label='Box Y Position', linestyle='--')
axs_box_pos.plot(t_time, data_box_pos[:, 2], label='Box Z Position', linestyle=':')
axs_box_pos.set_ylabel('Position [m]')
axs_box_pos.grid(True)
axs_box_pos.set_xlim([t_time[1], t_time[-1]])
axs_box_pos.legend(loc='upper right')
axs_box_pos.set_xlabel('Time [s]')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create a figure for IK tracking performance (all joints)
fig_ik_tracking, axs_ik = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
fig_ik_tracking.suptitle('IK Tracking Performance - All Joints')

joint_names_short = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'G1', 'G2']

for i in range(9):
    row = i // 3
    col = i % 3
    
    axs_ik[row, col].plot(t_time, data_q[:, i], label='Actual', linestyle='-', linewidth=2)
    axs_ik[row, col].plot(t_time, data_qr[:, i], label='IK Target', linestyle='--', linewidth=2)
    axs_ik[row, col].plot(t_time, data_qv[:, i], label='ERG Filtered', linestyle=':', linewidth=1.5)
    
    # Calculate and plot tracking error
    tracking_error = data_q[:, i] - data_qr[:, i]
    axs_ik[row, col].plot(t_time, tracking_error, label='Error', linestyle='-.', alpha=0.7)
    
    axs_ik[row, col].set_ylabel('Position [rad]')
    axs_ik[row, col].set_title(f'{joint_names_short[i]}')
    axs_ik[row, col].grid(True)
    axs_ik[row, col].legend(loc='upper right', fontsize=8)
    axs_ik[row, col].set_xlim([t_time[1], t_time[-1]])

# Set common xlabel for the bottom row
for col in range(3):
    axs_ik[2, col].set_xlabel('Time [s]')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create a figure for tracking error analysis
fig_error, axs_error = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
fig_error.suptitle('IK Tracking Error Analysis')

for i in range(9):
    row = i // 3
    col = i % 3
    
    # Calculate tracking error
    tracking_error = data_q[:, i] - data_qr[:, i]
    
    # Plot error
    axs_error[row, col].plot(t_time, tracking_error, label='Tracking Error', color='red', linewidth=2)
    
    # Add zero reference line
    axs_error[row, col].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Calculate RMS error
    rms_error = np.sqrt(np.mean(tracking_error**2))
    axs_error[row, col].text(0.02, 0.98, f'RMS: {rms_error:.4f}', 
                            transform=axs_error[row, col].transAxes, 
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axs_error[row, col].set_ylabel('Error [rad]')
    axs_error[row, col].set_title(f'{joint_names_short[i]} Error')
    axs_error[row, col].grid(True)
    axs_error[row, col].set_xlim([t_time[1], t_time[-1]])

# Set common xlabel for the bottom row
for col in range(3):
    axs_error[2, col].set_xlabel('Time [s]')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Print summary statistics
print("\n=== IK Tracking Performance Summary ===")
print("Joint\t\tRMS Error [rad]\tMax Error [rad]")
print("-" * 50)
for i in range(9):
    tracking_error = data_q[:, i] - data_qr[:, i]
    rms_error = np.sqrt(np.mean(tracking_error**2))
    max_error = np.max(np.abs(tracking_error))
    print(f"{joint_names_short[i]}\t\t{rms_error:.4f}\t\t{max_error:.4f}")

# Calculate overall performance metrics
all_errors = data_q - data_qr
overall_rms = np.sqrt(np.mean(all_errors**2))
overall_max = np.max(np.abs(all_errors))
print("-" * 50)
print(f"Overall\t\t{overall_rms:.4f}\t\t{overall_max:.4f}")

# Block diagram generation removed - not needed
