import time
from pydrake.all import *
import os
import numpy as np
from CERG_Setup import (
    PredictionParams,
    ErgParams,
    ContactParams,
    RobotSpec,
)

# Import FCL for collision detection in predictions
try:
    from fcl_test import FCLLinkDistanceSystem
    FCL_AVAILABLE = True
except ImportError:
    print("Warning: FCL not available. Collision detection in predictions will be disabled.")
    FCL_AVAILABLE = False



####################################
#     Create system diagram
####################################
def create_system_model(plant, scene_graph, robot_spec: RobotSpec, contact_params: ContactParams = None):
    """
    Add the Panda arm model and movable box to the plant and configure contact properties.
    
    Args:
        plant: The MultibodyPlant object to which the Panda arm model will be added.
        scene_graph: The SceneGraph object for visualization.
        robot_spec: RobotSpec describing the robot URDF and limits (required).
        contact_params: Optional ContactParams; if None, uses Drake hydroelastic defaults used elsewhere.
    
    Returns:
        Tuple containing the updated plant and scene_graph.
    """
    if contact_params is None:
        contact_params = ContactParams()

    urdf = "file://" + robot_spec.urdf_path
    Parser(plant).AddModelsFromUrl(urdf)
    plant.set_contact_surface_representation(contact_params.mesh_type)
    plant.set_contact_model(contact_params.contact_model)
    plant.set_discrete_contact_approximation(contact_params.discrete_solver)
    plant.Finalize()
    return plant, scene_graph

######################################################################################################
#                         #########  explicit_reference_governor  ##########                       #
######################################################################################################
class CompliantERG:
    def __init__(self, 
                 robot_spec: RobotSpec,
                 prediction_params: PredictionParams = PredictionParams(),
                 erg_params: ErgParams = ErgParams(),
                 contact_params: ContactParams = ContactParams()):
        """
        Initialize the Explicit Reference Governor (ERG) with given parameters.
        
        Args:
            robot_spec (RobotSpec): Robot specification (URDF path, gains, limits).
            prediction_params (PredictionParams): Prediction timing parameters.
            erg_params (ErgParams): ERG/DSM parameters.
            contact_params (ContactParams): Drake contact settings.
        """
        # Parameter containers
        self.erg_params = erg_params
        self.prediction_params = prediction_params
        self.contact_params = contact_params

        self.robot_spec = robot_spec
        self.num_joints = robot_spec.controlled_dofs
        self.controlled_indices = robot_spec.controlled_indices

        # Plant Configuration parameters
        time_step = self.prediction_params.dt
        self.builder_pred = DiagramBuilder()
        self.plant_pred, scene_graph= AddMultibodyPlantSceneGraph(self.builder_pred, time_step)
        
        # Use URDF from robot spec
        urdf = "file://" + robot_spec.urdf_path
        arm = Parser(self.plant_pred).AddModelsFromUrl(urdf)
        self.plant_pred.set_contact_surface_representation(self.contact_params.mesh_type)
        self.plant_pred.set_contact_model(self.contact_params.contact_model)
        self.plant_pred.set_discrete_contact_approximation(self.contact_params.discrete_solver)
        self.plant_pred.Finalize() 

        # Debug: Print plant information
        print(f"ERG Plant num_positions: {self.plant_pred.num_positions()}")
        print(f"ERG Plant num_velocities: {self.plant_pred.num_velocities()}")
        print(f"ERG num_joints: {self.num_joints}")
        
        # Finalize the diagram
        self.diagram = self.builder_pred.Build()                           
        self.diagram_context = self.diagram.CreateDefaultContext()    
        self.plant_context =  self.diagram.GetMutableSubsystemContext(self.plant_pred, self.diagram_context)
                
        self.eta_ = self.erg_params.eta 
        self.zeta_q_ = self.erg_params.zeta_q # range of influence for the repulsion field. 
        self.delta_q_ = self.erg_params.delta_q # threshold for when the repulsion effect starts to take place.
        self.dt_ = self.erg_params.dt  # Sampling time for the reference governor

        # Controller gains - derived from robot specification
        self.Kp_ = np.array(robot_spec.kp, dtype=float)
        self.Kd_ = np.array(robot_spec.kd, dtype=float)

        # Prediction parameters
        prediction_dt_ = self.prediction_params.dt
        prediction_horizon_ = self.prediction_params.horizon
        self.num_pred_samples_ = self.prediction_params.num_steps()

        self.num_positions =  self.plant_pred.num_positions()
        self.num_velocities = self.plant_pred.num_velocities()
        # Cache prediction integration step (driven by PredictionParams in CERG_Setup)
        self.prediction_dt_ = float(self.prediction_params.dt)
        # Cache control mapping (driven by RobotSpec in CERG_Setup)
        self.u_idx_ = np.array(self.controlled_indices, dtype=int).reshape(-1)
        self.nu_ = int(self.plant_pred.get_actuation_input_port().size())

        # Initialize FCL collision detection if available
        self.fcl_available = FCL_AVAILABLE
        if FCL_AVAILABLE:
            try:
                import fcl
                self.fcl = fcl
                print("FCL collision detection initialized successfully!")
                
                # Pre-build FCL collision objects for robot links
                self._build_fcl_collision_objects()
                
            except Exception as e:
                print(f"Failed to initialize FCL: {e}")
                self.fcl_available = False
        else:
            print("FCL not available - collision detection disabled")

        # Prediction lists for joint positions, velocities, and torques
        self.q_pred_list_ = np.zeros((self.num_positions, self.num_pred_samples_ + 1))
        self.dq_pred_list_ = np.zeros((self.num_velocities, self.num_pred_samples_ + 1))
        self.tau_pred_list_ = np.zeros((self.plant_pred.get_actuation_input_port().size(), self.num_pred_samples_ + 1))
        
        # Tracked bodies come from robot_spec; store poses for each prediction step
        self.tracked_bodies = list(robot_spec.tracked_bodies)
        B = len(self.tracked_bodies)
        self.body_pose_list_ = np.zeros((B, 3, self.num_pred_samples_ + 1))
        
        # Limits for joint angles, velocities, torques, EE velocities - driven by robot spec
        self.limit_q_min_ = np.array(robot_spec.q_min, dtype=float)
        self.limit_q_max_ = np.array(robot_spec.q_max, dtype=float)
        self.limit_tau_ = np.array(robot_spec.tau_max, dtype=float)
        self.limit_dq_ = np.array(robot_spec.dq_max, dtype=float)
        self.limit_dp_EE_ = [robot_spec.limit_dp_trans, robot_spec.limit_dp_rot]  # Translation and rotation limits for the end effector
        self.E_max_ = self.erg_params.E_max  # Maximum energy limit

        # Latest DSM (for logging/plotting)
        self.last_dsm = None

    def _build_fcl_collision_objects(self):
        """
        Pre-build FCL collision objects for robot links with modular finger detection.
        Uses fingers if found in URDF, otherwise uses rubber pad.
        """
        if not self.fcl_available:
            return
            
        # Define base links that should always exist
        base_link_names = ["panda_link7", "panda_hand"]
        
        # Try to detect fingers in the URDF
        finger_names = ["panda_leftfinger", "panda_rightfinger"]
        detected_fingers = []
        
        for finger_name in finger_names:
            try:
                body = self.plant_pred.GetBodyByName(finger_name)
                detected_fingers.append(finger_name)
                print(f"Detected finger: {finger_name}")
            except Exception:
                print(f"Finger {finger_name} not found in URDF")
        
        # If no fingers detected, use rubber pad
        if not detected_fingers:
            print("No fingers detected, using rubber pad")
            # Try different rubber pad names
            rubber_pad_names = ["rubber_pad", "panda_rubber_pad", "gripper_pad"]
            rubber_pad_found = False
            
            for pad_name in rubber_pad_names:
                try:
                    body = self.plant_pred.GetBodyByName(pad_name)
                    detected_fingers.append(pad_name)
                    rubber_pad_found = True
                    print(f"Using rubber pad: {pad_name}")
                    break
                except Exception:
                    continue
            
            if not rubber_pad_found:
                print("No rubber pad found, using panda_hand as fallback")
                detected_fingers.append("panda_hand")
        
        # Build collision objects for all detected links
        self.fcl_robot_links = []
        all_link_names = base_link_names + detected_fingers
        
        for link_name in all_link_names:
            try:
                body = self.plant_pred.GetBodyByName(link_name)
                
                # Different box sizes for different link types
                if "finger" in link_name.lower():
                    # Smaller box for fingers
                    geom = self.fcl.Box(0.05, 0.05, 0.05)  # 5cm cube for fingers
                elif "rubber" in link_name.lower() or "pad" in link_name.lower():
                    # Medium box for rubber pad
                    geom = self.fcl.Box(0.08, 0.08, 0.08)  # 8cm cube for rubber pad
                else:
                    # Standard box for other links
                    geom = self.fcl.Box(0.1, 0.1, 0.1)  # 10cm cube for other links
                
                collision_obj = self.fcl.CollisionObject(geom)
                self.fcl_robot_links.append((body.index(), link_name, np.eye(4), collision_obj))
                print(f"Built FCL collision object for: {link_name}")
                
            except Exception as e:
                print(f"Failed to build FCL object for {link_name}: {e}")
                # Add dummy object as fallback
                geom = self.fcl.Box(0.1, 0.1, 0.1)
                collision_obj = self.fcl.CollisionObject(geom)
                self.fcl_robot_links.append((self.plant_pred.world_body().index(), link_name, np.eye(4), collision_obj))
        
        print(f"Built {len(self.fcl_robot_links)} FCL collision objects")
        print(f"Link names: {[name for _, name, _, _ in self.fcl_robot_links]}")

    def _update_fcl_objects_from_context(self, context):
        """
        Update FCL collision object transforms based on current plant context.
        """
        if not self.fcl_available:
            return
            
        for bidx, _name, T_LC, obj in self.fcl_robot_links:
            if bidx == self.plant_pred.world_body().index():
                continue
            body = self.plant_pred.get_body(bidx)
            T_WL = self.plant_pred.EvalBodyPoseInWorld(context, body).GetAsMatrix4()
            T_WC = T_WL @ T_LC
            obj.setTransform(self.fcl.Transform(T_WC[:3, :3], T_WC[:3, 3]))

    def _tracked_body_positions_world(self) -> np.ndarray:
        """
        Return a (3, B) array of world-frame xyz positions for `self.tracked_bodies`.
        Uses `self.plant_context` / `self.plant_pred` (caller must have already set positions in context).
        """
        B = len(self.tracked_bodies)
        plant_positions = np.zeros((3, B), dtype=float)
        for idx, body_name in enumerate(self.tracked_bodies):
            try:
                body = self.plant_pred.GetBodyByName(body_name)
                plant_positions[:, idx] = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, body).translation()
            except Exception:
                plant_positions[:, idx] = 0.0
        return plant_positions

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
        # Store for downstream logging/plotting
        try:
            self.last_dsm = float(DSM_)
        except Exception:
            self.last_dsm = None

        q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        
     
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
        total_energy = self.trajectoryPredictions(np.concatenate((q, dq)), tau, q_v, box_position)
        # print(f"Total energy: {total_energy}")
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
        for i in range(self.num_joints):
            rho_rep_q[i] = max((self.zeta_q_ - abs(q_v[i] - self.limit_q_min_[i])) / (self.zeta_q_ - self.delta_q_), 0.0) - \
                           max((self.zeta_q_ - abs(q_v[i] - self.limit_q_max_[i])) / (self.zeta_q_ - self.delta_q_), 0.0)
        # print(f"norm rho_rep_q = {np.linalg.norm(rho_rep_q)}")
        # print(f"rho_rep_q = \n {rho_rep_q}")

        # Soft navigation field (if box position constraint is provided)
        rho_soft = np.zeros(self.num_positions)
        if box_position_constraint is not None:
            rho_soft = self.soft_navigation_field(box_position_constraint, q_v)

        # Total navigation field
        rho = rho_att 
        return rho

    def soft_navigation_field(self, box_position_constraint, q_v):
        """
        Compute soft navigation field based on box position constraint.

        """
        # Parameters from ERG settings
        delta_s = self.erg_params.soft_delta_s  # Safety distance
        eta_ = self.erg_params.soft_eta  # Small value to avoid division by zero
        
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
            hand_body = self.plant_pred.GetBodyByName("panda_hand")
            hand_pose = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, hand_body)
            link_names.append("panda_hand")
            link_positions.append(hand_pose.translation())
            frame = self.plant_pred.GetFrameByName("panda_hand")
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

        # Include panda fingers
        try:
            leftfinger_body = self.plant_pred.GetBodyByName("panda_leftfinger")
            leftfinger_pose = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, leftfinger_body)
            link_names.append("panda_leftfinger")
            link_positions.append(leftfinger_pose.translation())
            frame = self.plant_pred.GetFrameByName("panda_leftfinger")
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

        try:
            rightfinger_body = self.plant_pred.GetBodyByName("panda_rightfinger")
            rightfinger_pose = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, rightfinger_body)
            link_names.append("panda_rightfinger")
            link_positions.append(rightfinger_pose.translation())
            frame = self.plant_pred.GetFrameByName("panda_rightfinger")
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

        # Calculate normalized q_dots for each link (7 links + hand + 2 fingers)
        normalized_q_dots = []
        for i in range(len(link_positions)-1):  # This will be 0-6 for 7 links
            qdot = jacobians[i] @ c
            norm_qdot = np.linalg.norm(qdot)
            normalized_q_dot = qdot / max(norm_qdot, eta_)
            normalized_q_dots.append(normalized_q_dot)
        # print(type(min(self.num_joints + 1, len(link_positions))))
        # Calculate soft repulsion for each link (only for the main joints)
        for i in range(1, int(min(self.num_joints + 1, len(link_positions)))):  # Process links 1 to num_joints
            link_pos = link_positions[i]
            
            # Calculate scale factor using individual joint KP gains
            w = box_position_constraint[0]  # Use only the x-component of box position
            dot_product = np.dot(c, link_pos)
            
            # Ensure all components are scalars
            w_scalar = float(w)
            dot_product_scalar = float(dot_product)
            kp_scalar = float(self.Kp_[i-1])  # Safe to access since we only go up to num_joints
            delta_s_scalar = float(delta_s)
            fd_scalar = float(self.erg_params.FD)
            if w_scalar + dot_product_scalar < -1e-12:
                breakpoint()
            scale_value = -kp_scalar * ((w_scalar + dot_product_scalar) / (delta_s_scalar * fd_scalar))
            scale = max(scale_value, 0.0)  # Ensure scalar result
            
            # Add to soft repulsion vector
            soft_rep[:self.num_joints] += scale * normalized_q_dots[i-1][:self.num_joints]  # Apply to all joints
        
        # Debug: trigger a breakpoint if the soft repulsion activates (non-zero).
        # if np.any(np.abs(soft_rep) > 1e-12):
        #     breakpoint()

        return soft_rep

    def trajectoryBasedDSM(self, q, dq, tau, q_v, box_position):
      """
      Compute the Dynamic Safety Margin (DSM) based on trajectory predictions.
      """
      # Get trajectory predictions and save predicted q, dq, and tau in lists
      start_time = time.time()
      total_energy = self.trajectoryPredictions(np.concatenate((q, dq)), tau, q_v, box_position)

      # Compute DSMs
      DSM_tau_ = self.dsmTau()
      DSM_q_ = self.dsmQ()
      DSM_dq_ = self.dsmDq()
      DSM_dp_EE_ = self.dsmDpEE()
      DSM_s_ = self.dsmS(box_position)
      # Debug: trigger a breakpoint if the distance-based DSM goes negative.
      if DSM_s_ < -1e-12:
          breakpoint()
      DSM_energy_ = self.dsmEnergy(total_energy, box_position, DSM_s_)

      # Find the minimum among the DSMs
      DSM = min(DSM_tau_,DSM_dq_)
      DSM = min(DSM,DSM_q_)
      DSM = min(DSM,DSM_dp_EE_)
      DSM = min(DSM,DSM_energy_)
      # DSM = min(DSM,DSM_terminal_energy_)

      DSM = max(DSM, 0)
    #   DSM =1.0
    # Print DSMs
    #   print(f"DSM_tau_: {DSM_tau_}")
    #   print(f"DSM_q_: {DSM_q_}")
    #   print(f"DSM_dq_: {DSM_dq_}")
    #   print(f"DSM_dp_EE_: {DSM_dp_EE_}")
    #   print(f"DSM_final: {DSM}")
      
      return DSM

    def trajectoryPredictions(self, state, tau, q_v, box_position=None):
        """
        Predict joint positions, velocities, and torques over the prediction horizon.
        
        Args:
            state: Current state [q, dq]
            tau: Current torques
            q_v: Reference joint positions
            box_position: Box position [x, y, z] for collision detection
        """
        # Cache sizes once (avoid repeating self.num_positions everywhere).
        nq = self.num_positions
        nv = self.num_velocities
        u_idx = self.u_idx_

        # State is expected as [q (nq), dq (nv)].
        q_pred = np.asarray(state[:nq], dtype=float).copy()
        dq_pred = np.asarray(state[nq : nq + nv], dtype=float).copy()
        # tau is the *current* actuation coming from the plant; store it as the k=0 sample.
        tau_pred = np.asarray(tau, dtype=float).reshape(-1)
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
            # Potential energy term uses only the controlled joints (RobotSpec.controlled_indices)
            Kp_ = np.asarray(self.Kp_, dtype=float).reshape(-1)
            position_error = q_v[u_idx] - q_pred[u_idx]
            Kp_diag_matrix = np.diag(Kp_)  # Create diagonal matrix from Kp gains (controlled joints only)
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
        # Store initial torque sample in actuation space (best-effort truncate/pad).
        # Be defensive: some Drake vector types can lead to non-int dtypes in shape/min.
        tau0 = np.zeros(self.tau_pred_list_.shape[0], dtype=float)
        n0 = int(min(int(tau0.shape[0]), int(np.asarray(tau_pred).shape[0])))
        tau0[:n0] = np.asarray(tau_pred, dtype=float).reshape(-1)[:n0]
        self.tau_pred_list_[:, 0] = tau0
        # Calculate and store initial body pose
        try:
            # Set plant to initial q_pred
            self.plant_pred.SetPositions(self.plant_context, q_pred)
            plant_positions = self._tracked_body_positions_world()  # (3, B)
            # Store all tracked body poses in body_pose_list_ (shape: (B, 3, ...))
            for link_idx in range(len(self.tracked_bodies)):
                self.body_pose_list_[link_idx, :, 0] = plant_positions[:, link_idx]
        except Exception as e:
            print(f"Initial body pose calculation failed: {e}")
            self.body_pose_list_[:, :, 0] = 0.0

        # Cache arrays for rollout; avoid mutating self.* inside the loop.
        Kp_ = np.asarray(self.Kp_, dtype=float).reshape(-1)
        Kd_ = np.asarray(self.Kd_, dtype=float).reshape(-1)

        for k in range(self.num_pred_samples_):
        

            gravity_pred = -self.plant_pred.CalcGravityGeneralizedForces(self.plant_context)  # PD+G convention
            

            # print("Shapes inside trajectoryPredictions:")
            # print("q_v shape:", q_v.shape)
            # print("q_pred shape:", q_pred.shape) 
            # print("dq_pred shape:", dq_pred.shape)
            # print("gravity_pred shape:", gravity_pred.shape)
            # Compute torques for controlled joints only, then expand to full generalized dimension.
            tau_u = Kp_ * (q_v[u_idx] - q_pred[u_idx]) - Kd_ * dq_pred[u_idx] + np.asarray(gravity_pred, dtype=float).reshape(-1)[u_idx]
            tau_pred = np.zeros(nq, dtype=float)
            tau_pred[u_idx] = tau_u
            # Solve for x[k+1] using the computed tau_pred
            
            state_pred, body_pose_translation = self.calc_dynamics(
                np.concatenate((q_pred, dq_pred)),
                tau_pred,
                q_v,
            )
            q_pred = state_pred[:nq]
            dq_pred = state_pred[nq:]

            # Store predicted states
            q_pred_traj.append(q_pred.copy())
            dq_pred_traj.append(dq_pred.copy())
            tau_pred_traj.append(tau_pred.copy())
            # print(q_v)

            # Add q, dq, and tau to prediction list
            self.q_pred_list_[:, k + 1] = q_pred
            self.dq_pred_list_[:, k + 1] = dq_pred
            # Store in actuation space for DSM_tau (truncate/pad if needed).
            tmp = np.zeros(self.tau_pred_list_.shape[0], dtype=float)
            n = int(min(int(tmp.shape[0]), int(np.asarray(tau_u).shape[0])))
            tmp[:n] = np.asarray(tau_u, dtype=float).reshape(-1)[:n]
            self.tau_pred_list_[:, k + 1] = tmp
            
            # Store body pose from calc_dynamics return
            # body_pose_translation contains poses for tracked bodies; store for this step
            for link_idx in range(len(self.tracked_bodies)):
                self.body_pose_list_[link_idx, :, k + 1] = body_pose_translation[:, link_idx]
            
            # Get FCL distances if available
            if self.fcl_available and box_position is not None:
                try:
                    # Get FCL normal distances from robot links to box
                    fcl_distances = self.get_fcl_distances(q_pred, box_position)
                    if fcl_distances is not None:
                        # Dynamic print based on actual number of collision objects
                        link_names = [name for _, name, _, _ in self.fcl_robot_links]
                        distance_str = ", ".join([f"{name}={dist:.4f}" for name, dist in zip(link_names, fcl_distances)])
                        print(f"FCL normal distances in prediction step {k+1}: {distance_str}")
                    
                    # Get FCL X-axis distances from robot links to box
                    fcl_x_distances = self.get_fcl_x_distances(q_pred, box_position)
                    if fcl_x_distances is not None:
                        # Dynamic print based on actual number of collision objects
                        link_names = [name for _, name, _, _ in self.fcl_robot_links]
                        distance_str = ", ".join([f"{name}={dist:.4f}" for name, dist in zip(link_names, fcl_x_distances)])
                        print(f"FCL X-axis distances in prediction step {k+1}: {distance_str}")
                        
                except Exception as e:
                    print(f"FCL distance calculation failed in prediction: {e}")

        # Convert lists to arrays for plotting
        q_pred_traj = np.array(q_pred_traj)
        dq_pred_traj = np.array(dq_pred_traj)
        tau_pred_traj = np.array(tau_pred_traj)

        return total_energy

    def get_fcl_distances(self, q_pred, box_position=None):
        """
        Get FCL distances from robot links to box.
        Creates collision objects dynamically based on box position.
        
        Args:
            q_pred: Predicted joint positions
            box_position: Box position [x, y, z] (optional)
        
        Returns:
            List of distances [link7, hand, leftfinger, rightfinger] or None if not available
        """
        if not self.fcl_available or box_position is None:
            return None
            
        try:
            # Set plant to predicted configuration
            self.plant_pred.SetPositions(self.plant_context, q_pred)
            
            # Create FCL collision objects for robot links
            robot_links = []
            link_names = ["panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger"]
            
            for link_name in link_names:
                try:
                    body = self.plant_pred.GetBodyByName(link_name)
                    body_pose = self.plant_pred.EvalBodyPoseInWorld(self.plant_context, body)
                    link_position = body_pose.translation()
                    
                    # Create 10cm cube collision object for each link
                    geom = self.fcl.Box(0.1, 0.1, 0.1)
                    collision_obj = self.fcl.CollisionObject(geom)
                    collision_obj.setTransform(self.fcl.Transform(link_position))
                    robot_links.append(collision_obj)
                except Exception:
                    robot_links.append(None)
            
            # Create FCL collision object for box
            box_geom = self.fcl.Box(0.22, 0.30, 0.20)  # Box dimensions
            box_collision_obj = self.fcl.CollisionObject(box_geom)
            box_collision_obj.setTransform(self.fcl.Transform(box_position))
            
            # Calculate distances between robot links and box
            distances = []
            for i, robot_link in enumerate(robot_links):
                if robot_link is not None:
                    # Calculate distance between robot link and box
                    req = self.fcl.DistanceRequest(enable_signed_distance=True)
                    res = self.fcl.DistanceResult()
                    distance = self.fcl.distance(robot_link, box_collision_obj, req, res)
                    distances.append(distance)
                else:
                    distances.append(float('inf'))  # Link not found
            
            return distances
            
        except Exception as e:
            print(f"FCL distance calculation failed: {e}")
            return None

    def get_fcl_distances_axis(self, q_pred, box_position=None, axis='x'):
        """
        Get FCL distances from robot links to box in a specific axis using pre-built collision objects.
        
        Args:
            q_pred: Predicted joint positions
            box_position: Box position [x, y, z] (optional)
            axis: 'x', 'y', or 'z' axis to measure distance along (default: 'x')
        
        Returns:
            List of distances along specified axis [link7, hand, finger1, finger2] or None
        """
        if not self.fcl_available or box_position is None:
            return None
            
        try:
            # Set plant to predicted configuration
            self.plant_pred.SetPositions(self.plant_context, q_pred)
            
            # Update pre-built FCL collision objects with current poses
            self._update_fcl_objects_from_context(self.plant_context)
            
            # Create FCL collision object for box
            box_geom = self.fcl.Box(0.22, 0.30, 0.20)  # Box dimensions
            box_collision_obj = self.fcl.CollisionObject(box_geom)
            box_collision_obj.setTransform(self.fcl.Transform(box_position))
            
            # Calculate axis-specific distances between robot links and box
            distances = []
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
            
            for (bidx, link_name, T_LC, robot_obj) in self.fcl_robot_links:
                if bidx != self.plant_pred.world_body().index():  # Skip dummy objects
                    # Calculate distance between robot link and box
                    req = self.fcl.DistanceRequest(enable_signed_distance=True, enable_nearest_points=True)
                    res = self.fcl.DistanceResult()
                    distance = self.fcl.distance(robot_obj, box_collision_obj, req, res)
                    
                    # Get nearest points and calculate axis distance
                    try:
                        nearest_point_robot = np.array(res.nearest_points[0])
                        nearest_point_box = np.array(res.nearest_points[1])
                        
                        # Calculate distance along specific axis: box_axis - robot_axis
                        axis_distance = nearest_point_box[axis_idx] - nearest_point_robot[axis_idx]
                        distances.append(axis_distance)
                        
                        # Debug print for X-axis
                        if axis.lower() == 'x':
                            print(f"FCL {axis}-axis distance for {link_name}: {axis_distance:.4f} m")
                            print(f"  Robot point: {nearest_point_robot}")
                            print(f"  Box point: {nearest_point_box}")
                            print(f"  Distance vector: {nearest_point_box - nearest_point_robot}")
                        
                    except Exception as e:
                        print(f"Failed to get nearest points for {link_name}: {e}")
                        # Fallback to overall distance if nearest points not available
                        distances.append(distance)
                else:
                    distances.append(float('inf'))  # Dummy object
            
            return distances
            
        except Exception as e:
            print(f"FCL axis distance calculation failed: {e}")
            return None

    def get_fcl_x_distances(self, q_pred, box_position=None):
        """
        Get FCL X-axis distances from robot links to box.
        Convenience method that returns only X distances.
        
        Args:
            q_pred: Predicted joint positions
            box_position: Box position [x, y, z] (optional)
        
        Returns:
            List of X distances [link7, hand, finger1, finger2] or None
        """
        return self.get_fcl_distances_axis(q_pred, box_position, axis='x')
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
        
        # Cache sizes once
        nq = self.num_positions
        # Extract current state components
        q = x[:nq]  # positions
        dq = x[nq:]  # velocities
        
        # Use the provided generalized torques `u` (computed in trajectoryPredictions from RobotSpec/ErgParams).
        # Convention: u already contains gravity compensation term (-tau_g) for controlled joints, so dynamics uses
        # (u - C + tau_g) to cancel gravity consistently with the original implementation.
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.shape[0] != nq:
            u_full = np.zeros(nq, dtype=float)
            n = min(nq, u.shape[0])
            u_full[:n] = u[:n]
            u = u_full

        ddq = np.linalg.pinv(M) @ (u - C + tau_g)
        
        # Euler integration to get next state
        dt = self.prediction_dt_  # time step (from PredictionParams in CERG_Setup)
        dq_next = dq + ddq * dt
        q_next = q + dq_next * dt

        x_next = np.concatenate([q_next, dq_next])
        
        # Calculate body pose using current plant context
        try:
            body_pose_translation = self._tracked_body_positions_world()  # (3, B)
        except Exception as e:
            print(f"Body pose calculation failed in calc_dynamics: {e}")
            body_pose_translation = np.zeros((3, len(self.tracked_bodies)), dtype=float)  # (3, B)
        
        # Return both next state and body pose translation
        return x_next, body_pose_translation


    def dsmTau(self):
        """
        Compute the DSM for joint torques.
        """
        delta = self.erg_params.robust_delta_tau
        kappa = self.erg_params.kappa_tau
        for k in range(self.tau_pred_list_.shape[1]):  # number of prediction samples + 1
            tau_pred = self.tau_pred_list_[:, k]
            DSM_tau_temp = self.distanceTau(tau_pred) - delta
            if k == 0:
                DSM_tau = DSM_tau_temp
            else:
                DSM_tau = min(DSM_tau, DSM_tau_temp)

        DSM_tau = kappa * DSM_tau
        return DSM_tau

    def dsmQ(self):
        """
        Compute the DSM for joint positions.
        """
        delta = self.erg_params.robust_delta_q
        kappa = self.erg_params.kappa_q
        for k in range(self.q_pred_list_.shape[1]):  # number of prediction samples + 1
            q_pred = self.q_pred_list_[:, k]
            DSM_q_temp = self.distanceQ(q_pred) - delta
            if k == 0:
                DSM_q = DSM_q_temp
            else:
                DSM_q = min(DSM_q, DSM_q_temp)

        DSM_q = kappa * DSM_q
        return DSM_q

    def dsmDq(self):
        """
        Compute the DSM for joint velocities.
        """
        delta = self.erg_params.robust_delta_dq
        kappa = self.erg_params.kappa_dq
        for k in range(self.dq_pred_list_.shape[1]):  # number of prediction samples + 1
            dotq_pred = self.dq_pred_list_[:, k]
            DSM_dotq_temp = self.distanceDq(dotq_pred) - delta
            if k == 0:
                DSM_dotq = DSM_dotq_temp
            else:
                DSM_dotq = min(DSM_dotq, DSM_dotq_temp)

        DSM_dotq = kappa * DSM_dotq
        return DSM_dotq

    def dsmDpEE(self):
        """
        Compute the DSM for end-effector velocities.
        """
        delta = self.erg_params.robust_delta_dp_EE
        kappa = self.erg_params.kappa_dp_EE
        for k in range(self.q_pred_list_.shape[1]):  # number of prediction samples + 1
            q_pred = self.q_pred_list_[:, k]
            dotq_pred = self.dq_pred_list_[:, k]
            DSM_dotp_EE_temp = self.distanceDpEE(q_pred, dotq_pred) - delta
            if k == 0:
                DSM_dotp_EE = DSM_dotp_EE_temp
            else:
                DSM_dotp_EE = min(DSM_dotp_EE, DSM_dotp_EE_temp)

        DSM_dotp_EE = kappa * DSM_dotp_EE
        return DSM_dotp_EE

    def dsmS(self, box_position):
        """
        Compute the DSM for distance-based safety margin.
        """
        kappa_q = self.erg_params.kappa_q
        for k in range(self.q_pred_list_.shape[1]):  # number of prediction samples + 1
            q_pred = self.q_pred_list_[:, k]
            DSM_s_temp = self.calculateDsmS(box_position, k)  # Use robust_delta_q_ for distance safety margin
            if k == 0:
                DSM_s = DSM_s_temp
            else:
                DSM_s = min(DSM_s, DSM_s_temp)

        DSM_s = kappa_q * DSM_s  # Use kappa_q as scaling factor
        return DSM_s

    def dsmEnergy(self, total_energy,box_position, dsm_s):

        # DSM energy: max(kappaS * dsm_s, kappaE * (E_max - E_current))
        kappaS = 1.0  # You can adjust this parameter
        kappaE = self.erg_params.kappa_terminal_energy
        
        DSM_energy = max(kappaS * dsm_s, kappaE * (self.E_max_ - total_energy))
        return DSM_energy

    def calculateDsmS(self, box_position, k):
        """
        Calculate DSM_s based on distance between robot links and tracked object.

        """
        if box_position is None:
            return float('inf')
        

        if self.body_pose_list_.shape[2] > 0:  # Check if we have stored positions
            # Use the first prediction step (index k) for current positions
            plant_positions = self.body_pose_list_[:, :, k]  # Shape: (B, 3)
            plant_positions = plant_positions.T  # Transpose to get (3, B) format
        else:
            # Fallback: return infinite if no stored positions
            return float('inf')
        
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
            if wall < -1e-12:
                print(f"Wall is negative: {wall}")
        
        # Return the minimum wall value
        return wall

    def distanceTau(self, tau_pred):
        for i in range(self.num_joints):
            tau_lowerlimit = tau_pred[i] - (-self.limit_tau_[i])
            tau_upperlimit = self.limit_tau_[i] - tau_pred[i]
            tau_distance_temp = min(tau_lowerlimit, tau_upperlimit)
            if i == 0:
                tau_distance = tau_distance_temp
            else:
                tau_distance = min(tau_distance, tau_distance_temp)
        return tau_distance

    def distanceQ(self, q_pred):
        for i in range(self.num_joints):  # include all joints
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
        for i in range(self.num_joints):
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
