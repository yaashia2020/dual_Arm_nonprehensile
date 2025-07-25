
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
    urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_fr3.urdf"))
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
                 robust_delta_dp_EE_, kappa_dp_EE_, kappa_terminal_energy_):
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

        self.num_positions =  self.plant_pred.num_positions()
        self.num_velocities = self.plant_pred.num_velocities()

        # Prediction lists for joint positions, velocities, and torques
        self.q_pred_list_ = np.zeros((self.num_positions, self.num_pred_samples_ + 1))
        self.dq_pred_list_ = np.zeros((self.num_velocities, self.num_pred_samples_ + 1))
        self.tau_pred_list_ = np.zeros((self.plant_pred.get_actuation_input_port().size(), self.num_pred_samples_ + 1))
        
        # Limits for joint angles, velocities, and torques
        self.limit_q_min_ = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.limit_q_max_ = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.limit_tau_ = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        self.limit_dq_ = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
        self.limit_dp_EE_ = [1.7, 2.5]  # Translation and rotation limits for the end effector


    def get_qv(self, q, dq, tau, q_r, q_v):
        """
        Compute the new reference joint positions using the navigation field and DSM.
        
        Args:
            q (np.array): Current joint positions.
            dq (np.array): Current joint velocities.
            tau (np.array): Current joint torques.
            q_r (np.array): Desired reference joint positions.
            q_v (np.array): Current applied reference joint positions.
        
        Returns:
            np.array: Updated reference joint positions.
        """

        
        
        rho_ = self.navigationField(q_r, q_v)
        DSM_ = self.trajectoryBasedDSM(q, dq, tau, q_v)

        q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        
        # if DSM_ > 0:
        #   q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        # else:
        #   q_v_new = q_v + np.min([np.linalg.norm(DSM_ * rho_ * self.dt_), np.linalg.norm(q_r - q_v)]) * DSM_ * rho_ / max(np.linalg.norm(DSM_ * rho_), self.eta_)
        
        return q_v_new

    def navigationField(self, q_r, q_v):
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

        # Total navigation field
        rho = rho_att + rho_rep_q
        return rho

    def trajectoryBasedDSM(self, q, dq, tau, q_v):
      """
      Compute the Dynamic Safety Margin (DSM) based on trajectory predictions.
      """
      # Get trajectory predictions and save predicted q, dq, and tau in lists
      start_time = time.time()
      self.trajectoryPredictions(np.concatenate((q, dq)), tau, q_v)
      print(f"TIme taken to predict  x= {(time.time() - start_time)*1000} ms")

      # Compute DSMs
      DSM_tau_ = self.dsmTau()
      DSM_q_ = self.dsmQ()
      DSM_dq_ = self.dsmDq()
      DSM_dp_EE_ = self.dsmDpEE()

      # Find the minimum among the DSMs
      print(f"DSM_tau_: {DSM_dq_}")
      DSM = min(DSM_tau_,DSM_dq_)
      DSM = min(DSM,DSM_q_)
      DSM = min(DSM,DSM_dp_EE_)
      # DSM = min(DSM,DSM_terminal_energy_)

      DSM = max(DSM, 0)
      print(f"DSM: {DSM}")
    #   DSM =1.0
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

        # Initialize lists to store predicted states
        q_pred_traj = [q_pred.copy()]
        dq_pred_traj = [dq_pred.copy()]
        tau_pred_traj = [tau_pred.copy()]
        
        

        # Initialize lists to store predicted states
        self.q_pred_list_[:, 0] = q_pred
        self.dq_pred_list_[:, 0] = dq_pred
        self.tau_pred_list_[:, 0] = tau_pred
        
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
            tau_pred = np.zeros(9)
            tau_pred[:7] = tau_pred_partial
            tau_pred[7:] = 0  # or some other feedforward/zero torque for extra joints



            # Solve for x[k+1] using the computed tau_pred
            
            state_pred = self.calc_dynamics(np.concatenate((q_pred, dq_pred)), tau_pred, q_v)  # Adjust this based on your calculation method
            print(f"state_pred: \n {state_pred}")
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
            # print(f"TIme taken to predict one sample = {(time.time() - start_time)*1000} ms")

        # Convert lists to arrays for plotting
        q_pred_traj = np.array(q_pred_traj)
        dq_pred_traj = np.array(dq_pred_traj)
        tau_pred_traj = np.array(tau_pred_traj)

        # # Define total prediction time T and time step size dt
        # dt = 0.2 / self.num_pred_samples_  # Time step size
        # time_steps = np.linspace(0, 0.2, self.num_pred_samples_ + 1)  # Time steps array

        # Assuming time_steps, q_pred_traj, dq_pred_traj, tau_pred_traj, q_desired_traj, dq_desired_traj, tau_desired_traj are defined

        # plt.figure()

        # Plot predicted states
        # plt.plot(time_steps, q_pred_traj, label='Predicted q', linestyle='-', color='b')
        # plt.plot(time_steps, dq_pred_traj, label='Predicted dq', linestyle='--', color='b')
        # plt.plot(time_steps, tau_pred_traj, label='Predicted tau', linestyle=':', color='b')

        # plt.xlabel('Time')
        # plt.ylabel('Angle Position / Velocity / Torque')
        # plt.legend()
        # plt.title('Predicted vs Desired States')
        # plt.show()

        # Plot predicted states vs desired states for each joint
        # num_joints = self.num_positions
        # fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints))

        # for i in range(num_joints):
        #     axs[i].plot(time_steps, q_pred_traj[:, i], label='Predicted q', linestyle='-', color='b')
        #     axs[i].plot(time_steps, q_v[i] * np.ones_like(time_steps), label='Desired q', linestyle='--', color='r')
        #     axs[i].set_xlabel('Time')
        #     axs[i].set_ylabel(f'Joint {i+1} Position')
        #     axs[i].legend()
        #     axs[i].set_title(f'Predicted vs Desired Position for Joint {i+1}')

        # plt.tight_layout()
        # plt.show()

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
        # print("IsDifferenceEquationSystem:", self.diagram.IsDifferenceEquationSystem())
        # assert self.diagram.IsDifferenceEquationSystem()[0], "must be a discrete-time system"
        """
        Calculate the next state given the current state x and control input u.
        
        Args:
            x: Current state vector.
            u: Control input vector (torque).
            qv: Desired position vector.
        
        Returns:
            The next state vector.
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
        u_control = np.concatenate([u_control, [0, 0]])
        
        # End effector control (commented out in original)
        # J_pseudo = J.completeOrthogonalDecomposition().pseudoInverse()
        # u_control = -kp * J_pseudo * p_diff - kd * dq[:U] - tau_g[:U]
        
        # Compute acceleration using inverse dynamics
        # Use only the first 7 elements for the dynamics calculation
        # u_control_7 = u_control[:U]  # Take only first 7 elements
        ddq = np.linalg.pinv(M) @ (u_control - C + tau_g)
        
        # Euler integration to get next state
        dt = 0.01  # time step
        dq_next = dq + ddq * dt
        q_next = q + dq_next * dt
        
        # Handle extra joints (keep them unchanged or set to zero)
        # if len(q) > U:
        #     q_next = np.concatenate([q_next, q[U:]])  # Keep extra joints unchanged
        #     dq_next = np.concatenate([dq_next, dq[U:]])  # Keep extra joints unchanged
        
        # Combine into state vector
        x_next = np.concatenate([q_next, dq_next])
        
        # print(x_next)
        print(f"x - x_next = {x - x_next}")
        return x_next


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
