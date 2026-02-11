import time
from pydrake.all import *
import os
import numpy as np

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
def create_system_model(plant, scene_graph):
    """
    Add the Panda arm model and movable box to the plant and configure contact properties.
    
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
                 robust_delta_dp_EE_, kappa_dp_EE_, kappa_terminal_energy_, FD_=1.0, 
                 num_joints=None,
                 urdf_path=None,
                 name="erg",
                 tracked_body_names=None,
                 fcl_link_names=None,
                 ee_body_name=None,
                 limit_q_min=None,
                 limit_q_max=None,
                 limit_tau=None,
                 limit_dq=None,
                 limit_dp_EE=None,
                 E_max=5.0,
                 verbose=True,
                 **_ignored_kwargs):
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
            num_joints (int): Number of robot joints (default: 7 for Panda).
            urdf_path (str): Path to URDF file (optional, uses default Panda if None).
            name (str): Optional identifier for debug prints (e.g. robot1/robot2).
        """
        self.name_ = str(name) if name is not None else "erg"
        self.verbose_ = bool(verbose)
        # Latest DSM value (for external consumers / logging).
        self.last_dsm_ = 0.0
        
        # Plant Configuration parameters
        time_step = 0.01
        # PLant for simulation
        self.builder_pred = DiagramBuilder()
        self.plant_pred, scene_graph= AddMultibodyPlantSceneGraph(self.builder_pred, time_step)
        
        # Use provided URDF path or default Panda URDF
        if urdf_path is None:
            urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_drake.urdf"))
        
        urdf = "file://" + urdf_path
        arm = Parser(self.plant_pred).AddModelsFromUrl(urdf)
        contact_model = ContactModel.kHydroelasticWithFallback
        discrete_solver = DiscreteContactApproximation.kSap
        mesh_type = HydroelasticContactRepresentation.kTriangle
        self.plant_pred.set_contact_surface_representation(mesh_type)
        self.plant_pred.set_contact_model(contact_model)
        self.plant_pred.set_discrete_contact_approximation(discrete_solver)
        self.plant_pred.Finalize() 

        # Dimensions / joint count.
        self.num_positions = int(self.plant_pred.num_positions())
        self.num_velocities = int(self.plant_pred.num_velocities())
        self.num_actuated = int(self.plant_pred.get_actuation_input_port().size())
        # "Controlled joints" default to number of actuated DoFs unless overridden.
        self.num_joints = int(self.num_actuated if num_joints is None else num_joints)

        if self.verbose_:
            print(f"ERG Plant num_positions: {self.num_positions}")
            print(f"ERG Plant num_velocities: {self.num_velocities}")
            print(f"ERG num_joints (controlled): {self.num_joints}")
        
        # Finalize the diagram
        self.diagram = self.builder_pred.Build()                           
        self.diagram_context = self.diagram.CreateDefaultContext()    
        self.plant_context =  self.diagram.GetMutableSubsystemContext(self.plant_pred, self.diagram_context)
                
        self.eta_ = 0.005 
        self.zeta_q_ = 0.15 # range of influence for the repulsion field. 
        self.delta_q_ = 0.1 # threshold for when the repulsion effect starts to take place.
        self.dt_ = 0.01  # Sampling time for the refrence governor

        # Controller gains (defaults; override by editing/deriving if needed).
        if self.num_joints == 7:
            # Panda defaults (kept for backwards compatibility / good behavior).
            self.Kp_ = [120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0]
            self.Kd_ = [8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0]
        else:
            # Generic gains for other joint counts.
            self.Kp_ = [100.0] * self.num_joints
            self.Kd_ = [5.0] * self.num_joints

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

        # Initialize FCL collision detection if available
        self.fcl_available = FCL_AVAILABLE
        if FCL_AVAILABLE:
            try:
                import fcl
                self.fcl = fcl
                if self.verbose_:
                    print("FCL collision detection initialized successfully!")
                
                # Pre-build FCL collision objects for robot links
                self._build_fcl_collision_objects(fcl_link_names=fcl_link_names, ee_body_name=ee_body_name)
                
            except Exception as e:
                if self.verbose_:
                    print(f"Failed to initialize FCL: {e}")
                self.fcl_available = False
        else:
            if self.verbose_:
                print("FCL not available - collision detection disabled")

        # Prediction lists for joint positions, velocities, and torques
        self.q_pred_list_ = np.zeros((self.num_positions, self.num_pred_samples_ + 1))
        self.dq_pred_list_ = np.zeros((self.num_velocities, self.num_pred_samples_ + 1))
        self.tau_pred_list_ = np.zeros((self.plant_pred.get_actuation_input_port().size(), self.num_pred_samples_ + 1))

        # If FCL is enabled, store x-axis distances (nearest-point delta in X) for each FCL robot object
        # across the prediction horizon. Shape: (num_fcl_links, num_pred_samples_ + 1)
        self.fcl_x_dist_list_ = None
        if self.fcl_available and hasattr(self, "fcl_robot_links"):
            self.fcl_x_dist_list_ = np.full(
                (len(self.fcl_robot_links), self.num_pred_samples_ + 1),
                float("inf"),
                dtype=float,
            )
        
        # ---- Tracked bodies for DSM_s / debugging ----
        # We track body translations over the prediction horizon. This list can be provided explicitly
        # (tracked_body_names) or auto-detected from the URDF.
        self.tracked_body_names = self._build_tracked_body_list(tracked_body_names=tracked_body_names)
        self.num_tracked_bodies = len(self.tracked_body_names)

        # Arrays to store body poses (translation part) for each prediction step
        # Shape: (num_tracked_bodies, 3 coordinates (X,Y,Z), prediction steps)
        # Order: self.tracked_body_names
        self.body_pose_list_ = np.zeros((self.num_tracked_bodies, 3, self.num_pred_samples_ + 1))
        
        # Limits (defaults pulled from the plant when possible).
        # Drake provides q/v limits via plant getters; torque limits are not always present in the model.
        self.limit_q_min_ = np.asarray(
            self.plant_pred.GetPositionLowerLimits()[: self.num_joints] if limit_q_min is None else limit_q_min,
            dtype=float,
        ).reshape(-1)
        self.limit_q_max_ = np.asarray(
            self.plant_pred.GetPositionUpperLimits()[: self.num_joints] if limit_q_max is None else limit_q_max,
            dtype=float,
        ).reshape(-1)
        # Use symmetric speed limit around zero.
        dq_upper = np.asarray(self.plant_pred.GetVelocityUpperLimits()[: self.num_joints], dtype=float).reshape(-1)
        self.limit_dq_ = np.asarray(dq_upper if limit_dq is None else limit_dq, dtype=float).reshape(-1)
        self.limit_tau_ = np.asarray(
            ([50.0] * self.num_joints) if limit_tau is None else limit_tau,
            dtype=float,
        ).reshape(-1)
        self.limit_dp_EE_ = [1.7, 2.5] if limit_dp_EE is None else list(limit_dp_EE)
        self.E_max_ = float(E_max)

        # Shape checks (mismatched limits silently break DSMs).
        for _nm, _arr in [
            ("limit_q_min", self.limit_q_min_),
            ("limit_q_max", self.limit_q_max_),
            ("limit_dq", self.limit_dq_),
            ("limit_tau", self.limit_tau_),
        ]:
            if _arr.shape[0] != self.num_joints:
                raise ValueError(f"[{self.name_}] {_nm} must have shape ({self.num_joints},), got {_arr.shape}")

    def _build_tracked_body_list(self, tracked_body_names=None):
        """
        Build the list of body names whose translations we track in body_pose_list_.

        - If `tracked_body_names` is provided: use it (filtering out missing bodies).
        - Else: best-effort auto detection. Prefer the Panda chain if present; otherwise track all non-world bodies.
        """
        if tracked_body_names is not None:
            tracked = []
            for nm in tracked_body_names:
                try:
                    self.plant_pred.GetBodyByName(nm)
                    tracked.append(nm)
                except Exception:
                    continue
            if tracked:
                if self.verbose_:
                    print(f"Tracked bodies for DSM_s: {tracked}")
                return tracked

        # Auto-detect: prefer common Panda chain if present.
        tracked = []
        panda_chain = [f"panda_link{i}" for i in range(1, 8)] + ["panda_hand"]
        for nm in panda_chain:
            try:
                self.plant_pred.GetBodyByName(nm)
                tracked.append(nm)
            except Exception:
                tracked = []
                break

        # Generic fallback: all bodies except world.
        if not tracked:
            for i in range(self.plant_pred.num_bodies()):
                body = self.plant_pred.get_body(BodyIndex(i))
                if body.index() == self.plant_pred.world_body().index():
                    continue
                tracked.append(body.name())

        # Add fingers if present
        for name in ["panda_leftfinger", "panda_rightfinger"]:
            try:
                self.plant_pred.GetBodyByName(name)
                tracked.append(name)
            except Exception:
                pass

        # Also add rubber pad if present (some URDFs use pad instead of fingers)
        for pad_name in ["rubber_pad", "panda_rubber_pad", "gripper_pad"]:
            try:
                self.plant_pred.GetBodyByName(pad_name)
                if pad_name not in tracked:
                    tracked.append(pad_name)
                break
            except Exception:
                continue

        # De-duplicate while preserving order
        deduped = []
        seen = set()
        for name in tracked:
            if name not in seen:
                deduped.append(name)
                seen.add(name)

        if self.verbose_:
            print(f"Tracked bodies for DSM_s: {deduped}")
        return deduped

    def _calc_tracked_body_positions(self, context):
        """
        Returns a (3, num_tracked_bodies) array of body translations in world frame.
        Missing bodies are filled with zeros.
        """
        plant_positions = np.zeros((3, self.num_tracked_bodies))
        for i, body_name in enumerate(self.tracked_body_names):
            try:
                body = self.plant_pred.GetBodyByName(body_name)
                plant_positions[:, i] = self.plant_pred.EvalBodyPoseInWorld(context, body).translation()
            except Exception:
                plant_positions[:, i] = 0.0
        return plant_positions

    def _build_fcl_collision_objects(self, fcl_link_names=None, ee_body_name=None):
        """
        Pre-build FCL collision objects for robot links with modular finger detection.
        Uses fingers if found in URDF, otherwise uses rubber pad.
        """
        if not self.fcl_available:
            return

        # If caller provided explicit set, use it (filtering out missing bodies).
        if fcl_link_names is not None:
            explicit = []
            for nm in fcl_link_names:
                try:
                    self.plant_pred.GetBodyByName(nm)
                    explicit.append(nm)
                except Exception:
                    continue
            if explicit:
                base_link_names = explicit
            else:
                base_link_names = []
        else:
            # Define base links that should always exist (Panda defaults).
            base_link_names = ["panda_link7", "panda_hand"]

        # Filter base_link_names to bodies that exist; if none exist, fall back to a reasonable default.
        filtered_base = []
        for nm in base_link_names:
            try:
                self.plant_pred.GetBodyByName(nm)
                filtered_base.append(nm)
            except Exception:
                continue
        if not filtered_base:
            fallback = ee_body_name or "panda_hand"
            try:
                self.plant_pred.GetBodyByName(fallback)
                filtered_base = [fallback]
            except Exception:
                # Last resort: pick the last non-world body in the model.
                for i in range(self.plant_pred.num_bodies() - 1, -1, -1):
                    body = self.plant_pred.get_body(BodyIndex(i))
                    if body.index() != self.plant_pred.world_body().index():
                        filtered_base = [body.name()]
                        break
        base_link_names = filtered_base

        # Detect optional end-effector bodies (fingers and/or rubber pad).
        detected_effectors = []

        # Try to detect fingers in the URDF
        finger_names = ["panda_leftfinger", "panda_rightfinger"]
        for finger_name in finger_names:
            try:
                self.plant_pred.GetBodyByName(finger_name)
                detected_effectors.append(finger_name)
                if self.verbose_:
                    print(f"Detected finger: {finger_name}")
            except Exception:
                if self.verbose_:
                    print(f"Finger {finger_name} not found in URDF")

        # Also detect rubber pad if present (even if fingers exist)
        rubber_pad_names = ["rubber_pad", "panda_rubber_pad", "gripper_pad"]
        for pad_name in rubber_pad_names:
            try:
                self.plant_pred.GetBodyByName(pad_name)
                if pad_name not in detected_effectors:
                    detected_effectors.append(pad_name)
                    if self.verbose_:
                        print(f"Detected rubber pad: {pad_name}")
                break
            except Exception:
                continue

        # If neither fingers nor rubber pad exist, fall back to panda_hand
        if not detected_effectors:
            fallback = ee_body_name or "panda_hand"
            if self.verbose_:
                print(f"No fingers/rubber pad detected, using {fallback} as fallback end-effector")
            detected_effectors.append(fallback)

        # Build collision objects for all detected links
        self.fcl_robot_links = []
        all_link_names = base_link_names + detected_effectors
        
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
                if self.verbose_:
                    print(f"Built FCL collision object for: {link_name}")
                
            except Exception as e:
                if self.verbose_:
                    print(f"Failed to build FCL object for {link_name}: {e}")
                # Add dummy object as fallback
                geom = self.fcl.Box(0.1, 0.1, 0.1)
                collision_obj = self.fcl.CollisionObject(geom)
                self.fcl_robot_links.append((self.plant_pred.world_body().index(), link_name, np.eye(4), collision_obj))
        
        if self.verbose_:
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

    def get_qv(self, q, dq, tau, q_r, q_v, box_position=None, box_position_fcl=None):
        """
        Compute the new reference joint positions using the navigation field and DSM.
        
        Args:
            q (np.array): Current joint positions.
            dq (np.array): Current joint velocities.
            tau (np.array): Current joint torques.
            q_r (np.array): Desired reference joint positions.
            q_v (np.array): Current applied reference joint positions.
            box_position (np.array): Box position used for constraints / navigation (often a "face" point).
            box_position_fcl (np.array): Box position used for FCL collision box placement (true center, no shift).
        
        Returns:
            np.array: Updated reference joint positions.
        """

        
        
        # Backward compatible: if not provided, use the same box_position for FCL as well.
        if box_position_fcl is None:
            box_position_fcl = box_position

        rho_ = self.navigationField(q_r, q_v, box_position)
        DSM_ = self.trajectoryBasedDSM(q, dq, tau, q_v, box_position, box_position_fcl)
        self.last_dsm_ = float(DSM_)

        q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        
        # if DSM_ > 0:
        #   q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        # else:
        #   q_v_new = q_v + np.min([np.linalg.norm(DSM_ * rho_ * self.dt_), np.linalg.norm(q_r - q_v)]) * DSM_ * rho_ / max(np.linalg.norm(DSM_ * rho_), self.eta_)
        
        return q_v_new

    def get_last_dsm(self):
        """Backward-compatible accessor used by wrappers (e.g. dual_robot_test_erg.py)."""
        return float(getattr(self, "last_dsm_", 0.0))

    def get_energy(self, q, dq, tau, q_r, q_v, box_position=None, box_position_fcl=None):
        """
        Get the calculated energy from trajectory predictions.
        
        Args:
            q (np.array): Current joint positions.
            dq (np.array): Current joint velocities.
            tau (np.array): Current joint torques.
            q_r (np.array): Desired reference joint positions.
            q_v (np.array): Current applied reference joint positions.
            box_position (np.array): Box position used for constraints / navigation (often a "face" point).
            box_position_fcl (np.array): Box position used for FCL collision box placement (true center, no shift).
        
        Returns:
            float: Calculated total energy.
        """
        # Get trajectory predictions and return the calculated energy
        # Backward compatible: if not provided, use the same box_position for FCL as well.
        if box_position_fcl is None:
            box_position_fcl = box_position

        total_energy = self.trajectoryPredictions(
            np.concatenate((q, dq)), tau, q_v, box_position=box_position, box_position_fcl=box_position_fcl
        )
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
            fd_scalar = float(self.FD_)
            
            scale_value = -kp_scalar * ((w_scalar + dot_product_scalar) / (delta_s_scalar * fd_scalar))
            scale = max(scale_value, 0.0)  # Ensure scalar result
            
            # Add to soft repulsion vector
            soft_rep[:self.num_joints] += scale * normalized_q_dots[i-1][:self.num_joints]  # Apply to all joints
        
        return soft_rep

    def trajectoryBasedDSM(self, q, dq, tau, q_v, box_position, box_position_fcl=None):
      """
      Compute the Dynamic Safety Margin (DSM) based on trajectory predictions.
      """
      if box_position_fcl is None:
          box_position_fcl = box_position
      # Get trajectory predictions and save predicted q, dq, and tau in lists
      start_time = time.time()
      total_energy = self.trajectoryPredictions(
          np.concatenate((q, dq)), tau, q_v, box_position=box_position, box_position_fcl=box_position_fcl
      )

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
    #   DSM =1.0
    # Print DSMs
    #   print(f"DSM_tau_: {DSM_tau_}")
    #   print(f"DSM_q_: {DSM_q_}")
    #   print(f"DSM_dq_: {DSM_dq_}")
    #   print(f"DSM_dp_EE_: {DSM_dp_EE_}")
    #   print(f"DSM_final: {DSM}")
      
      return DSM

    def trajectoryPredictions(self, state, tau, q_v, box_position=None, box_position_fcl=None):
        """
        Predict joint positions, velocities, and torques over the prediction horizon.
        
        Args:
            state: Current state [q, dq]
            tau: Current torques
            q_v: Reference joint positions
            box_position: Box position [x, y, z] used for DSM/constraints (often a "face" point).
            box_position_fcl: Box position [x, y, z] used to place the FCL collision box (true center, no shift).
        if box_position_fcl is None:
            box_position_fcl = box_position
        """
        q_pred, dq_pred = state[:self.num_joints], state[self.num_joints:]
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
            mass_matrix = self.plant_pred.CalcMassMatrix(self.plant_context)\
            
            # Kinetic energy: 0.5 * dq^T * M * dq
            kinetic_energy = 0.5 * dq_pred.T @ mass_matrix @ dq_pred
            
            # Potential energy: 0.5 * position_error^T * Kp_diagonal_matrix * position_error
            # Use joints based on num_joints
            position_error = q_v[:self.num_joints] - q_pred[:self.num_joints]
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
        
        # Calculate and store initial body pose (for all tracked bodies)
        try:
            self.plant_pred.SetPositions(self.plant_context, q_pred)
            plant_positions = self._calc_tracked_body_positions(self.plant_context)  # (3, N)
            for link_idx in range(self.num_tracked_bodies):
                self.body_pose_list_[link_idx, :, 0] = plant_positions[:, link_idx]
        except Exception as e:
            print(f"Initial body pose calculation failed: {e}")
            self.body_pose_list_[:, :, 0] = 0.0

        # Store initial FCL x-distances (k=0) if enabled
        if self.fcl_available and box_position_fcl is not None and self.fcl_x_dist_list_ is not None:
            try:
                fcl_x0 = self.get_fcl_x_distances(q_pred, box_position_fcl, pred_step=0)
                if fcl_x0 is not None:
                    self.fcl_x_dist_list_[:, 0] = np.array(fcl_x0, dtype=float)
            except Exception:
                # Keep inf defaults
                pass

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
            tau_pred_partial = self.Kp_ * (q_v[:self.num_joints] - q_pred[:self.num_joints]) - self.Kd_ * dq_pred[:self.num_joints] + gravity_pred[:self.num_joints]
            tau_pred = np.zeros(self.num_positions)
            tau_pred[:self.num_joints] = tau_pred_partial
            tau_pred[self.num_joints:] = 0  # Zero torque for extra joints



            # Solve for x[k+1] using the computed tau_pred
            
            state_pred, body_pose_translation = self.calc_dynamics(
                np.concatenate((q_pred, dq_pred)), tau_pred, q_v
            )
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
            
            # Store body pose from calc_dynamics return for this prediction step
            # body_pose_translation is (3, num_tracked_bodies)
            for link_idx in range(self.num_tracked_bodies):
                self.body_pose_list_[link_idx, :, k + 1] = body_pose_translation[:, link_idx]
            
            # Get FCL distances if available
            if self.fcl_available and box_position_fcl is not None:
                try:
                    # Get FCL normal distances from robot links to box
                    fcl_distances = self.get_fcl_distances(q_pred, box_position_fcl, pred_step=k + 1)
                    if fcl_distances is not None:
                        # Dynamic print based on actual number of collision objects
                        link_names = [name for _, name, _, _ in self.fcl_robot_links]
                        distance_str = ", ".join([f"{name}={dist:.4f}" for name, dist in zip(link_names, fcl_distances)])
                        print(f"FCL normal distances in prediction step {k+1}: {distance_str}")
                    
                    # Get FCL X-axis distances from robot links to box
                    fcl_x_distances = self.get_fcl_x_distances(q_pred, box_position_fcl, pred_step=k + 1)
                    if fcl_x_distances is not None:
                        # Store x-distances for DSM_s usage (only for calculated FCL objects)
                        if self.fcl_x_dist_list_ is not None and len(fcl_x_distances) == self.fcl_x_dist_list_.shape[0]:
                            self.fcl_x_dist_list_[:, k + 1] = np.array(fcl_x_distances, dtype=float)

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

    def get_fcl_distances(self, q_pred, box_position=None, pred_step=None):
        """
        Get FCL distances from robot links to box.
        Uses pre-built collision objects (self.fcl_robot_links) and updates transforms from plant context.
        
        Args:
            q_pred: Predicted joint positions
            box_position: Box position [x, y, z] (optional)
        
        Returns:
            List of signed distances in the same order as self.fcl_robot_links (dummy entries return inf),
            or None if not available.
        """
        if not self.fcl_available or box_position is None:
            return None
            
        try:
            # Set plant to predicted configuration
            self.plant_pred.SetPositions(self.plant_context, q_pred)

            # Update pre-built FCL collision objects with current poses (includes rubber_pad if present)
            self._update_fcl_objects_from_context(self.plant_context)
            
            # Create FCL collision object for box
            box_geom = self.fcl.Box(0.22, 0.30, 0.20)  # Box dimensions
            box_collision_obj = self.fcl.CollisionObject(box_geom)
            box_collision_obj.setTransform(self.fcl.Transform(box_position))
            
            # Calculate distances between robot links and box
            distances = []
            for (bidx, link_name, _T_LC, robot_obj) in self.fcl_robot_links:
                if bidx != self.plant_pred.world_body().index():  # Skip dummy objects
                    req = self.fcl.DistanceRequest(enable_signed_distance=True)
                    res = self.fcl.DistanceResult()
                    distance = self.fcl.distance(robot_obj, box_collision_obj, req, res)
                    distances.append(distance)
                else:
                    distances.append(float('inf'))
            
            return distances
            
        except Exception as e:
            print(f"FCL distance calculation failed: {e}")
            return None

    def get_fcl_distances_axis(self, q_pred, box_position=None, axis='x', pred_step=None):
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

    def get_fcl_x_distances(self, q_pred, box_position=None, pred_step=None):
        """
        Get FCL X-axis distances from robot links to box.
        Convenience method that returns only X distances.
        
        Args:
            q_pred: Predicted joint positions
            box_position: Box position [x, y, z] (optional)
        
        Returns:
            List of X distances [link7, hand, finger1, finger2] or None
        """
        return self.get_fcl_distances_axis(q_pred, box_position, axis='x', pred_step=pred_step)
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
        U = self.num_joints  # Number of actual robot joints
        
        # Control law: u = kp * (qv - q) + kd * dq - tau_g
        # Apply control to the robot joints
        u_control = kp * (qv[:U] - q[:U]) - kd * dq[:U] - tau_g[:U]
        
        # Add zeros for any extra joints beyond num_joints
        if self.num_positions > U:
            u_control = np.concatenate([u_control, np.zeros(self.num_positions - U)])
        
        # End effector control (commented out in original)
        # J_pseudo = J.completeOrthogonalDecomposition().pseudoInverse()
        # u_control = -kp * J_pseudo * p_diff - kd * dq[:U] - tau_g[:U]
        

        ddq = np.linalg.pinv(M) @ (u_control - C + tau_g)
        
        # Euler integration to get next state
        dt = 0.01  # time step
        dq_next = dq + ddq * dt
        q_next = q + dq_next * dt

        x_next = np.concatenate([q_next, dq_next])

        # Calculate body pose for the NEXT state, for all tracked bodies (including rubber_pad if present)
        try:
            # Update context to next state so EvalBodyPoseInWorld matches q_next
            xd.SetFromVector(x_next)
            body_pose_translation = self._calc_tracked_body_positions(self.plant_context)  # (3, N)
        except Exception as e:
            print(f"Body pose calculation failed in calc_dynamics: {e}")
            body_pose_translation = np.zeros((3, self.num_tracked_bodies))
        
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
            plant_positions = self.body_pose_list_[:, :, k]  # (N, 3)
            plant_positions = plant_positions.T              # (3, N)
        else:
            # Fallback: return infinite if no stored positions
            return float('inf')
        
        # Define constraint vector
        c = np.array([-1.0, 0.0, 0.0])  # Vector c = [-1, 0, 0]
        
        # Box position constraint - use box's x value
        box_constraint = box_position[0]  # Use x coordinate of box position
        
        # --- Existing "wall" metric (keep as-is) ---
        wall = float('inf')
        for i in range(plant_positions.shape[1]):
            link_pos = plant_positions[:, i]
            distance = box_constraint - np.dot(c, link_pos)
            wall = min(distance, wall)

        # --- FCL-based x-axis signed distance (ONLY for calculated FCL objects) ---
        # If available, use the minimum X-axis nearest-point delta across real FCL objects at this step.
        # This does not replace the wall metric for non-FCL-tracked bodies; we take the min of both.
        fcl_x_min = float("inf")
        if self.fcl_available and self.fcl_x_dist_list_ is not None:
            try:
                if 0 <= k < self.fcl_x_dist_list_.shape[1]:
                    # Filter out dummy entries (inf) and take min
                    vals = self.fcl_x_dist_list_[:, k]
                    finite = vals[np.isfinite(vals)]
                    if finite.size > 0:
                        fcl_x_min = float(np.min(finite))
            except Exception:
                pass

        return min(wall, fcl_x_min)

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