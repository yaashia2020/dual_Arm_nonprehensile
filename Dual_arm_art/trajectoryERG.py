import time
from pydrake.all import *
import os
import numpy as np


####################################
#     Create system diagram
####################################
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
    contact_model = ContactModel.kHydroelasticWithFallback  # Options: Hydroelastic, Point, or HydroelasticWithFallback
    discrete_solver = DiscreteContactApproximation.kSap # Options:kTamsi, kSap, kLagged, kSimilar
    mesh_type = HydroelasticContactRepresentation.kTriangle  # Options: Triangle or Polygon    
    plant.set_contact_surface_representation(mesh_type)
    plant.set_contact_model(contact_model)
    plant.set_discrete_contact_approximation(discrete_solver)
    plant.Finalize()
    return plant, scene_graph

######################################################################################################
#                         #########  explicit_reference_governor  ##########                       #
######################################################################################################
class ExplicitReferenceGovernor:
    def __init__(self, robust_delta_tau_, kappa_tau_, 
                 robust_delta_q_, kappa_q_, robust_delta_dq_, kappa_dq_, 
                 robust_delta_dp_EE_, kappa_dp_EE_, kappa_terminal_energy_, FD_=1.0):
        """
        Initialize the Explicit Reference Governor (ERG) with given parameters.
        
        Args:
            robust_delta_tau_ (float): Robustness parameter for joint torques.
            kappa_tau_ (float): Scaling parameter for joint torques.
            robust_delta_q_ (float): Robustness parameter for joint positions.
            kappa_q_ (float): Scaling parameter for joint positions.
            robust_delta_dq_ (float): Robustness parameter for joint velocities.
            kappa_dq_ (float): Scaling parameter for joint velocities.
            robust_delta_dp_EE_ (float): Robustness parameter for end-effector velocities.
            kappa_dp_EE_ (float): Scaling parameter for end-effector velocities.
            kappa_terminal_energy_ (float): Scaling parameter for terminal energy.
            FD_ (float): Force damping parameter for soft navigation field.
        """
        # Plant Configuration parameters
        time_step = 0.01
        # PLant for simulation
        self.builder_pred = DiagramBuilder()
        self.plant_pred, scene_graph= AddMultibodyPlantSceneGraph(self.builder_pred, time_step)
        self.plant_pred, scene_graph = create_system_model(self.plant_pred, scene_graph) 

        # Finalize the diagram
        self.diagram = self.builder_pred.Build()                           
        self.diagram_context = self.diagram.CreateDefaultContext()    
        self.plant_context =  self.diagram.GetMutableSubsystemContext(self.plant_pred, self.diagram_context)
                
        self.eta_ = 0.005 
        self.zeta_q_ = 0.15 # range of influence for the repulsion field. 
        self.delta_q_ = 0.1 # threshold for when the repulsion effect starts to take place.
        self.dt_ = 0.01  # Sampling time for the refrence governor

        # Controller gains
        self.Kp_ = [120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0]
        self.Kd_ = [8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0]

        # Prediction parameters
        prediction_dt_ = time_step# 0.01  # Time step for predictions
        prediction_horizon_ = 0.2  # Total prediction horizon
        self.num_pred_samples_ = int(prediction_horizon_ / prediction_dt_)

        # Robustness and scaling parameters
        self.robust_delta_tau_ = robust_delta_tau_
        self.kappa_tau_ = kappa_tau_
        self.robust_delta_q_ = robust_delta_q_
        self.kappa_q_ = kappa_q_
        self.robust_delta_dq_ = robust_delta_dq_
        self.kappa_dq_ = kappa_dq_
        self.robust_delta_dp_EE_ = robust_delta_dp_EE_
        self.kappa_dp_EE_ = kappa_dp_EE_
        self.kappa_terminal_energy_ = kappa_terminal_energy_
        self.FD_ = FD_  # Force damping parameter

        self.num_positions =  self.plant_pred.num_positions()
        self.num_velocities = self.plant_pred.num_velocities()

        # Prediction lists for joint positions, velocities, and torques
        self.q_pred_list_ = np.zeros((self.num_positions, self.num_pred_samples_ + 1))
        self.dq_pred_list_ = np.zeros((self.num_velocities, self.num_pred_samples_ + 1))
        self.tau_pred_list_ = np.zeros((self.plant_pred.get_actuation_input_port().size(), self.num_pred_samples_ + 1))
        
        # Arrays to store body poses (translation part) for each prediction step
        # Store poses for 7 links (panda_link1..panda_link7) PLUS panda_hand for each prediction step
        # Shape: (8 tracked bodies, 3 coordinates (X,Y,Z), prediction steps)
        self.body_pose_list_ = np.zeros((8, 3, self.num_pred_samples_ + 1))
        
        # Limits for joint angles, velocities, and torques
        self.limit_q_min_ = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.limit_q_max_ = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.limit_tau_ = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        self.limit_dq_ = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
        self.limit_dp_EE_ = [1.7, 2.5]  # Translation and rotation limits for the end effector
        self.E_max_ = 3.0  # Maximum energy limit

    def get_qv(self, q, dq, tau, q_r, q_v, box_position=None):
        """
        Compute the new reference joint positions using the navigation field and DSM.
        
        Args:
            q (np.array): Current joint positions.
            dq (np.array): Current joint velocities.
            tau (np.array): Current joint torques.
            q_r (np.array): Desired reference joint positions.
            q_v (np.array): Current applied reference joint positions.
            box_position (np.array): Current box position
        
        Returns:
            np.array: Updated reference joint positions.
        """

        
        
        rho_ = self.navigationField(q_r, q_v, box_position)
        DSM_ = self.trajectoryBasedDSM(q, dq, tau, q_v, box_position)

        q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        
        # if DSM_ > 0:
        #   q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        # else:
        #   q_v_new = q_v + np.min([np.linalg.norm(DSM_ * rho_ * self.dt_), np.linalg.norm(q_r - q_v)]) * DSM_ * rho_ / max(np.linalg.norm(DSM_ * rho_), self.eta_)
        
        return q_v_new

    def get_energy(self, q, dq, tau, q_r, q_v, box_position=None):
        """
        Get the calculated energy from trajectory predictions.
        
        Args:
            q (np.array): Current joint positions.
            dq (np.array): Current joint velocities.
            tau (np.array): Current joint torques.
            q_r (np.array): Desired reference joint positions.
            q_v (np.array): Current applied reference joint positions.
            box_position (np.array): Current box position
        
        Returns:
            float: Calculated total energy.
        """
        # Get trajectory predictions and return the calculated energy
        total_energy = self.trajectoryPredictions(np.concatenate((q, dq)), tau, q_v)
        print(f"Total energy: {total_energy}")
        return total_energy

    def navigationField(self, q_r, q_v, box_position_constraint=None):
        """
        Compute the navigation field based on attraction and repulsion forces.
        """
        rho_att = np.zeros(self.num_positions)
        rho_rep_q = np.zeros(self.num_positions)
        rho = np.zeros(self.num_positions)

        # Attraction field
        rho_att = (q_r - q_v) / max(np.linalg.norm(q_r - q_v), self.eta_)
        # print(f"norm rho_att = {np.linalg.norm(rho_att)}")
        # print(f"rho_att = \n {rho_att}")

        # Joint angle repulsion field (q)
        for i in range(7):
            rho_rep_q[i] = max((self.zeta_q_ - abs(q_v[i] - self.limit_q_min_[i])) / (self.zeta_q_ - self.delta_q_), 0.0) - \
                           max((self.zeta_q_ - abs(q_v[i] - self.limit_q_max_[i])) / (self.zeta_q_ - self.delta_q_), 0.0)
        # print(f"norm rho_rep_q = {np.linalg.norm(rho_rep_q)}")
        # print(f"rho_rep_q = \n {rho_rep_q}")

        # Soft navigation field (if box position constraint is provided)
        rho_soft = np.zeros(self.num_positions)
        if box_position_constraint is not None:
            rho_soft = self.soft_navigation_field(box_position_constraint, q_v)

        # Total navigation field
        rho = rho_att + rho_rep_q + rho_soft
        return rho

    def soft_navigation_field(self, box_position_constraint, q_v):
        """
        Compute soft navigation field based on box position constraint.

        """
        # Define parameters (you can adjust these)
        delta_s = 0.1  # Safety distance
        eta_ = 0.005  # Small value to avoid division by zero
        
        # Define constraint vector
        c = np.array([-1.0, 0.0, 0.0])  # Vector c = [-1, 0, 0]
        
        # Initialize soft repulsion vector
        U = self.num_positions  # Number of joints
        soft_rep = np.zeros(U)
        
        # Set the plant context to use q_v (current reference) instead of current joint positions
        state = self.plant_context.get_mutable_state()
        discrete_values = state.get_mutable_discrete_state()
        xd = discrete_values.get_mutable_vector()
        # Set positions to q_v and velocities to zero
        xd.SetFromVector(np.concatenate([q_v, np.zeros(U)]))
        
        # Get link positions and calculate Jacobians using q_v
        link_positions = []
        jacobians = []
        link_names = []
        
        for j in range(1, 8):  # Links 1-7 (changed from 1-6)
            body_name = f"panda_link{j}"
            link_names.append(body_name)
            
            body = self.plant_pred.GetBodyByName(body_name)
            body_pose = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, body)
            link_position = body_pose.translation()
            link_positions.append(link_position)
            
            # Calculate Jacobian for this link
            frame = self.plant_pred.GetFrameByName(body_name)
            world_frame = self.plant_pred.world_frame()
            
            # Calculate translational velocity Jacobian
            Jq_V_WEp = self.plant_pred.CalcJacobianTranslationalVelocity(
                self.plant_context,
                JacobianWrtVariable.kQDot,
                frame,
                np.zeros(3),  # p_BoBp_B
                world_frame,
                world_frame
            )
            
            # Calculate pseudoinverse of Jacobian
            J_pinv = np.linalg.pinv(Jq_V_WEp)
            jacobians.append(J_pinv)
        
        # Also include panda_hand
        try:
            hand_body = self.plant_pred.GetBodyByName("rubber_pad")
            hand_pose = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, hand_body)
            link_names.append("rubber_pad")
            link_positions.append(hand_pose.translation())
            frame = self.plant_pred.GetFrameByName("rubber_pad")
            world_frame = self.plant_pred.world_frame()
            Jq_V_WEp = self.plant_pred.CalcJacobianTranslationalVelocity(
                self.plant_context,
                JacobianWrtVariable.kQDot,
                frame,
                np.zeros(3),
                world_frame,
                world_frame
            )
            J_pinv = np.linalg.pinv(Jq_V_WEp)
            jacobians.append(J_pinv)
        except Exception:
            pass

        # Calculate normalized q_dots for each link (7 + optional hand)
        normalized_q_dots = []
        for i in range(len(link_positions)-1):  # This will be 0-6 for 7 links
            qdot = jacobians[i] @ c
            norm_qdot = np.linalg.norm(qdot)
            normalized_q_dot = qdot / max(norm_qdot, eta_)
            normalized_q_dots.append(normalized_q_dot)
        
        # Calculate soft repulsion for each link
        for i in range(1, len(link_positions)):  # Start from 1 as in C++ code (includes hand if present)
            link_pos = link_positions[i]
            
            # Calculate scale factor using individual joint KP gains
            w = box_position_constraint[0]  # Use only the x-component of box position
            dot_product = np.dot(c, link_pos)
            
            # Debug: Check types and values

            
            # Ensure all components are scalars
            w_scalar = float(w)
            dot_product_scalar = float(dot_product)
            kp_scalar = float(self.Kp_[i-1])
            delta_s_scalar = float(delta_s)
            fd_scalar = float(self.FD_)
            
            scale_value = -kp_scalar * ((w_scalar + dot_product_scalar) / (delta_s_scalar * fd_scalar))
            scale = max(scale_value, 0.0)  # Ensure scalar result

            
            # Add to soft repulsion vector
            soft_rep[:7] += scale * normalized_q_dots[i-1][:7]  # Apply to first 7 joints
        
        return soft_rep

    def trajectoryBasedDSM(self, q, dq, tau, q_v, box_position):
      """
      Compute the Dynamic Safety Margin (DSM) based on trajectory predictions.
      """
      # Get trajectory predictions and save predicted q, dq, and tau in lists
      start_time = time.time()
      total_energy = self.trajectoryPredictions(np.concatenate((q, dq)), tau, q_v)

      # Compute DSMs
      DSM_tau_ = self.dsmTau()
      DSM_q_ = self.dsmQ()
      DSM_dq_ = self.dsmDq()
      DSM_dp_EE_ = self.dsmDpEE()
      DSM_s_ = self.dsmS(box_position)
      DSM_energy_ = self.dsmEnergy(total_energy, box_position, DSM_s_)

      # Find the minimum among the DSMs
      DSM = min(DSM_tau_,DSM_dq_)
      DSM = min(DSM,DSM_q_)
      DSM = min(DSM,DSM_dp_EE_)
      DSM = min(DSM,DSM_energy_)
      # DSM = min(DSM,DSM_terminal_energy_)

      DSM = max(DSM, 0)
      # DSM =1.0
    # Print DSMs
    #   print(f"DSM_tau_: {DSM_tau_}")
    #   print(f"DSM_q_: {DSM_q_}")
    #   print(f"DSM_dq_: {DSM_dq_}")
    #   print(f"DSM_dp_EE_: {DSM_dp_EE_}")
    #   print(f"DSM_final: {DSM}")
      
      return DSM

    def trajectoryPredictions(self, state, tau, q_v):
        """
        Predict joint positions, velocities, and torques over the prediction horizon.
        """
        q_pred, dq_pred = state[:self.num_positions], state[self.num_positions:]
        tau_pred = tau

        # Calculate energy at the beginning of trajectory prediction
        # Energy = InertiaMatrix*joint_velocity^2 + Kp_gains*(q_v-q)^2
        try:
            # Set the plant context to current joint positions (like in calc_dynamics)
            state = self.plant_context.get_mutable_state()
            discrete_values = state.get_mutable_discrete_state()
            xd = discrete_values.get_mutable_vector()
            xd.SetFromVector(np.concatenate([q_pred, dq_pred]))
            
            # Get the mass matrix (inertia matrix) for the current configuration
            mass_matrix = self.plant_pred.CalcMassMatrix(self.plant_context)
            
            # Kinetic energy: 0.5 * dq^T * M * dq
            kinetic_energy = 0.5 * dq_pred.T @ mass_matrix @ dq_pred
            
            # Potential energy: 0.5 * position_error^T * Kp_diagonal_matrix * position_error
            # Use only first 7 joints for Kp gains (matching the controller)
            position_error = q_v[:7] - q_pred[:7]
            Kp_diag_matrix = np.diag(self.Kp_)  # Create diagonal matrix from Kp gains
            potential_energy = 0.5 * position_error.T @ Kp_diag_matrix @ position_error
            
            # Total energy
            total_energy = kinetic_energy + potential_energy
            
        except Exception as e:
            print(f"Energy calculation failed: {e}")
            total_energy = 0.0

        # Initialize lists to store predicted states
        q_pred_traj = [q_pred.copy()]
        dq_pred_traj = [dq_pred.copy()]
        tau_pred_traj = [tau_pred.copy()]
        
        

        # Initialize lists to store predicted states
        self.q_pred_list_[:, 0] = q_pred
        self.dq_pred_list_[:, 0] = dq_pred
        self.tau_pred_list_[:, 0] = tau_pred
        
        # Calculate and store initial body pose
        try:
            # Set plant to initial q_pred
            self.plant_pred.SetPositions(self.plant_context, q_pred)  # Remove model_instance parameter
            
            # Define link range (similar to C++ code)
            init_link_id = 1  # Start from panda_link1
            final_link_id = 7  # End at panda_link7
            s = final_link_id - init_link_id + 2  # 7 links + panda_hand
            
            # Initialize plant_positions matrix (3 x s)
            plant_positions = np.zeros((3, s))
            
            # Calculate poses for all joints from init_link_id to final_link_id
            for j in range(init_link_id, final_link_id + 1):
                body_name = f"panda_link{j}"
                body = self.plant_pred.GetBodyByName(body_name)
                plant_joint_position = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, body).translation()
                plant_positions[:, j - init_link_id] = plant_joint_position

            # Also include panda_hand as the 8th tracked body
            try:
                hand_body = self.plant_pred.GetBodyByName("panda_hand")
                hand_pos = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, hand_body).translation()
                plant_positions[:, 7] = hand_pos
            except Exception:
                plant_positions[:, 7] = 0.0
            
            # Store all link poses (7 links + hand) in body_pose_list_
            for link_idx in range(8):
                self.body_pose_list_[link_idx, :, 0] = plant_positions[:, link_idx]
                
        except Exception as e:
            print(f"Initial body pose calculation failed: {e}")
            self.body_pose_list_[:, :, 0] = 0.0

        for k in range(self.num_pred_samples_):
        

            gravity_pred = - self.plant_pred.CalcGravityGeneralizedForces(self.plant_context) # Compute gravity_pred for the current state
            

            # print("Shapes inside trajectoryPredictions:")
            # print("q_v shape:", q_v.shape)
            # print("q_pred shape:", q_pred.shape) 
            # print("dq_pred shape:", dq_pred.shape)
            # print("gravity_pred shape:", gravity_pred.shape)
            # import numpy as np
            self.Kp_ = np.array(self.Kp_)
            self.Kd_ = np.array(self.Kd_)


            # Compute tau_pred
            tau_pred_partial = self.Kp_ * (q_v[:7] - q_pred[:7]) - self.Kd_ * dq_pred[:7] +gravity_pred[:7]
            tau_pred = np.zeros(7)
            tau_pred[:] = tau_pred_partial



            # Solve for x[k+1] using the computed tau_pred
            
            state_pred, body_pose_translation = self.calc_dynamics(np.concatenate((q_pred, dq_pred)), tau_pred, q_v)  # Adjust this based on your calculation method
            q_pred = state_pred[:self.num_positions]
            dq_pred = state_pred[self.num_positions:]

            # Store predicted states
            q_pred_traj.append(q_pred.copy())
            dq_pred_traj.append(dq_pred.copy())
            tau_pred_traj.append(tau_pred.copy())
            # print(q_v)

            # Add q, dq, and tau to prediction list
            self.q_pred_list_[:, k + 1] = q_pred
            self.dq_pred_list_[:, k + 1] = dq_pred
            self.tau_pred_list_[:, k + 1] = tau_pred
            
            # Store body pose from calc_dynamics return
            # body_pose_translation contains poses for all 7 links + panda_hand in column 7
            # Store in body_pose_list_ for this prediction step
            for link_idx in range(8):
                self.body_pose_list_[link_idx, :, k + 1] = body_pose_translation[:, link_idx]

        # Convert lists to arrays for plotting
        q_pred_traj = np.array(q_pred_traj)
        dq_pred_traj = np.array(dq_pred_traj)
        tau_pred_traj = np.array(tau_pred_traj)

        return total_energy

    # Calculate system dynamics
    '''
    TamsiSolver uses the Transition-Aware Modified Semi-Implicit (TAMSI) method, [Castro et al., 2019], 
    to solve the equations below for mechanical systems in contact with regularized friction:
                q̇ = N(q) v
    (1)  M(q) v̇ = τ + Jₙᵀ(q) fₙ(q, v) + Jₜᵀ(q) fₜ(q, v)

    where:
    - v ∈ ℝⁿᵛ: Vector of generalized velocities
    - M(q) ∈ ℝⁿᵛˣⁿᵛ: Mass matrix
    - Jₙ(q) ∈ ℝⁿᶜˣⁿᵛ : Jacobian of normal separation velocities
    - Jₜ(q) ∈ ℝ²ⁿᶜˣⁿᵛ: Jacobian of tangent velocities
    - fₙ ∈ ℝⁿᶜ: Vector of normal contact forces
    - fₜ ∈ ℝ²ⁿᶜ: Vector of tangent friction forces
    - τ ∈ ℝⁿᵛ: Vector of generalized forces containing all other applied forces (e.g., Coriolis, gyroscopic terms, actuator forces, etc.) but contact forces.

    This solver assumes a compliant law for the normal forces fₙ(q, v) and therefore the functional dependence of fₙ(q, v) with q and v is stated explicitly.

    Since TamsiSolver uses regularized friction, we explicitly emphasize the functional dependence of fₜ(q, v) with the generalized velocities. 
    The functional dependence of fₜ(q, v) with the generalized positions stems from its direct dependence with the normal forces fₙ(q, v).
    '''
    def calc_dynamics(self, x, u, qv):
    
        """
        Calculate the next state given the current state x and control input u.
        
        Args:
            x: Current state vector.
            u: Control input vector (torque).
            qv: Desired position vector.
        
        Returns:
            Tuple containing (next_state, body_pose_translation).
        """
        # Set the discrete state directly
        state = self.plant_context.get_mutable_state()
        discrete_values = state.get_mutable_discrete_state()
        xd = discrete_values.get_mutable_vector()
        xd.SetFromVector(x)

        # Calculate gravity, mass matrix, and bias term
        tau_g = self.plant_pred.CalcGravityGeneralizedForces(self.plant_context)  # gravity
        M = self.plant_pred.CalcMassMatrix(self.plant_context)                    # mass matrix
        C = self.plant_pred.CalcBiasTerm(self.plant_context)                      # bias term
        
        # Extract current state components
        q = x[:self.num_positions]  # positions
        dq = x[self.num_positions:]  # velocities
        
        # PD controller with gravity compensation
        kp = np.array(self.Kp_)
        kd = np.array(self.Kd_)
        U = 7  # Number of actual robot joints (excluding extra joints)
        
        # Control law: u = kp * (qv - q) + kd * dq - tau_g
        # Only apply control to the first 7 joints
        u_control = kp * (qv[:7] - q[:7]) - kd * dq[:7] - tau_g[:7]
        
        # Add 2 zeros at the end for the extra joints
        #u_control = np.concatenate([u_control, [0, 0]])
        
        # End effector control (commented out in original)
        # J_pseudo = J.completeOrthogonalDecomposition().pseudoInverse()
        # u_control = -kp * J_pseudo * p_diff - kd * dq[:U] - tau_g[:U]
        

        ddq = np.linalg.pinv(M) @ (u_control - C + tau_g)
        
        # Euler integration to get next state
        dt = 0.01  # time step
        dq_next = dq + ddq * dt
        q_next = q + dq_next * dt

        x_next = np.concatenate([q_next, dq_next])
        
        # Calculate body pose using current q_pred (q) for all joints
        try:
            # Set plant to current joint positions
            # self.plant_pred.SetPositions(self.plant_context, 0, q)  # Assuming model instance 0
            
            # Define link range (similar to C++ code)
            init_link_id = 1  # Start from panda_link1
            final_link_id = 7  # End at panda_link7
            s = final_link_id - init_link_id + 2  # 7 links + panda_hand
            
            # Initialize plant_positions matrix (3 x s)
            plant_positions = np.zeros((3, s))
            
            # Calculate poses for all joints from init_link_id to final_link_id
            for j in range(init_link_id, final_link_id + 1):
                body_name = f"panda_link{j}"
                body = self.plant_pred.GetBodyByName(body_name)
                plant_joint_position = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, body).translation()
                # Store in plant_positions matrix (similar to C++: plant_positions.col(j - init_link_id))
                plant_positions[:, j - init_link_id] = plant_joint_position

            # Also include panda_hand as column 7 (8th tracked body)
            try:
                hand_body = self.plant_pred.GetBodyByName("panda_hand")
                hand_pos = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, hand_body).translation()
                plant_positions[:, 7] = hand_pos
            except Exception:
                plant_positions[:, 7] = 0.0

            # Return all link poses (7 links + hand)
            body_pose_translation = plant_positions
            
        except Exception as e:
            print(f"Body pose calculation failed in calc_dynamics: {e}")
            body_pose_translation = np.zeros((3, 8))  # Return 3x8 matrix for 7 links + hand
        
        # Return both next state and body pose translation
        return x_next, body_pose_translation


    def dsmTau(self):
        """
        Compute the DSM for joint torques.
        """
        for k in range(self.tau_pred_list_.shape[1]):  # number of prediction samples + 1
            tau_pred = self.tau_pred_list_[:, k]
            DSM_tau_temp = self.distanceTau(tau_pred) - self.robust_delta_tau_
            if k == 0:
                DSM_tau = DSM_tau_temp
            else:
                DSM_tau = min(DSM_tau, DSM_tau_temp)

        DSM_tau = self.kappa_tau_ * DSM_tau
        return DSM_tau

    def dsmQ(self):
        """
        Compute the DSM for joint positions.
        """
        for k in range(self.q_pred_list_.shape[1]):  # number of prediction samples + 1
            q_pred = self.q_pred_list_[:, k]
            DSM_q_temp = self.distanceQ(q_pred) - self.robust_delta_q_
            if k == 0:
                DSM_q = DSM_q_temp
            else:
                DSM_q = min(DSM_q, DSM_q_temp)

        DSM_q = self.kappa_q_ * DSM_q
        return DSM_q

    def dsmDq(self):
        """
        Compute the DSM for joint velocities.
        """
        for k in range(self.dq_pred_list_.shape[1]):  # number of prediction samples + 1
            dotq_pred = self.dq_pred_list_[:, k]
            DSM_dotq_temp = self.distanceDq(dotq_pred) - self.robust_delta_dq_
            if k == 0:
                DSM_dotq = DSM_dotq_temp
            else:
                DSM_dotq = min(DSM_dotq, DSM_dotq_temp)

        DSM_dotq = self.kappa_dq_ * DSM_dotq
        return DSM_dotq

    def dsmDpEE(self):
        """
        Compute the DSM for end-effector velocities.
        """
        for k in range(self.q_pred_list_.shape[1]):  # number of prediction samples + 1
            q_pred = self.q_pred_list_[:, k]
            dotq_pred = self.dq_pred_list_[:, k]
            DSM_dotp_EE_temp = self.distanceDpEE(q_pred, dotq_pred) - self.robust_delta_dp_EE_            
            if k == 0:
                DSM_dotp_EE = DSM_dotp_EE_temp
            else:
                DSM_dotp_EE = min(DSM_dotp_EE, DSM_dotp_EE_temp)

        DSM_dotp_EE = self.kappa_dp_EE_ * DSM_dotp_EE
        return DSM_dotp_EE

    def dsmS(self, box_position):
        """
        Compute the DSM for distance-based safety margin.
        """
        for k in range(self.q_pred_list_.shape[1]):  # number of prediction samples + 1
            q_pred = self.q_pred_list_[:, k]
            DSM_s_temp = self.calculateDsmS(box_position, k)  # Use robust_delta_q_ for distance safety margin
            if k == 0:
                DSM_s = DSM_s_temp
            else:
                DSM_s = min(DSM_s, DSM_s_temp)

        DSM_s = self.kappa_q_ * DSM_s  # Use kappa_q_ as scaling factor
        return DSM_s

    def dsmEnergy(self, total_energy,box_position, dsm_s):

        # DSM energy: max(kappaS * dsm_s, kappaE * (E_max - E_current))
        kappaS = 1.0  # You can adjust this parameter
        kappaE = self.kappa_terminal_energy_
        
        DSM_energy = max(kappaS * dsm_s, kappaE * (self.E_max_ - total_energy))
        return DSM_energy

    def calculateDsmS(self, box_position, k):
        """
        Calculate DSM_s based on distance between robot links and tracked object.

        """
        if box_position is None:
            return float('inf')
        

        if self.body_pose_list_.shape[2] > 0:  # Check if we have stored positions
            # Use the first prediction step (index 0) for current positions
            plant_positions = self.body_pose_list_[:, :, k]  # Shape: (7, 3) - 7 links, 3 coordinates
            plant_positions = plant_positions.T  # Transpose to get (3, 7) format
        else:
            # Fallback: return infinite if no stored positions
            return float('inf')
        
        # Define constraint vector
        c = np.array([-1.0, 0.0, 0.0])  # Vector c = [-1, 0, 0]
        
        # Box position constraint - use box's x value
        box_constraint = box_position[0]  # Use x coordinate of box position
        
        # Calculate minimum distance using the C++ logic pattern
        wall = float('inf')
        for k in range(plant_positions.shape[1]):
            link_pos = plant_positions[:, k]
            # Calculate: (box_constraint - c.dot(link_pos))
            distance = box_constraint - np.dot(c, link_pos)
            wall = min(distance, wall)
        
        # Return the minimum wall value
        return wall

    def distanceTau(self, tau_pred):
        for i in range(7):
            tau_lowerlimit = tau_pred[i] - (-self.limit_tau_[i])
            tau_upperlimit = self.limit_tau_[i] - tau_pred[i]
            tau_distance_temp = min(tau_lowerlimit, tau_upperlimit)
            if i == 0:
                tau_distance = tau_distance_temp
            else:
                tau_distance = min(tau_distance, tau_distance_temp)
        return tau_distance

    def distanceQ(self, q_pred):
        for i in range(7):  # include all joints
            q_lowerlimit = q_pred[i] - self.limit_q_min_[i]
            q_upperlimit = self.limit_q_max_[i] - q_pred[i]
            q_distance_temp = min(q_lowerlimit, q_upperlimit)
            if i == 0:
                q_distance = q_distance_temp
            else:
                q_distance = min(q_distance, q_distance_temp)
        # print(f"q_distance: {q_distance}")
        return q_distance

    def distanceDq(self, dotq_pred):
        for i in range(7):
            distance_dotq_lowerlimit = dotq_pred[i] - (-self.limit_dq_[i])
            distance_dotq_upperlimit = self.limit_dq_[i] - dotq_pred[i]
            distance_dotq_temp = min(distance_dotq_lowerlimit, distance_dotq_upperlimit)
            if i == 0:
                distance_dotq = distance_dotq_temp
            else:
                distance_dotq = min(distance_dotq, distance_dotq_temp)
        return distance_dotq
      
    def distanceDpEE(self, q_pred, dotq_pred): ### Fix
        endeffector_jacobian = np.zeros((6, self.num_positions))  # Example initialization
        dotp_EE = endeffector_jacobian @ dotq_pred

        for i in range(6):
            if i < 3:  # Translation
                distance_dotp_EE_lowerlimit = dotp_EE[i] - (-self.limit_dp_EE_[0])
                distance_dotp_EE_upperlimit = self.limit_dp_EE_[0] - dotp_EE[i]
            else:  # Rotation
                distance_dotp_EE_lowerlimit = dotp_EE[i] - (-self.limit_dp_EE_[1])
                distance_dotp_EE_upperlimit = self.limit_dp_EE_[1] - dotp_EE[i]
            distance_dotp_EE_temp = min(distance_dotp_EE_lowerlimit, distance_dotp_EE_upperlimit)
            if i == 0:
                distance_dotp_EE = distance_dotp_EE_temp
            else:
                distance_dotp_EE = min(distance_dotp_EE, distance_dotp_EE_temp)
        return distance_dotp_EE
