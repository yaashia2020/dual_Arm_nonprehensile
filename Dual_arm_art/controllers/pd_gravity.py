import numpy as np
from pydrake.all import LeafSystem


class PD_gravity(LeafSystem):
    """
    PD + gravity compensation controller.

    - Input "Desired_state": desired joint positions (size = nu actuated joints)
    - Input "Current_state": full plant state (q,v) for the model instance (size = nq+nv)
    - Output "tau_u": commanded torques (size = nu) for plant actuation port

    Notes:
    - Gravity is computed from an internal plant context that is updated with the current q each call.
    - Gains are taken from the provided arrays (truncated/padded to nu).
    """

    def __init__(self, plant, model_instance, kp, kd):
        super().__init__()
        self.plant = plant
        self.model_instance = model_instance
        self.nq = plant.num_positions(model_instance)
        self.nv = plant.num_velocities(model_instance)
        self.nu = plant.num_actuators(model_instance)

        kp = np.asarray(kp, dtype=float).reshape(-1)
        kd = np.asarray(kd, dtype=float).reshape(-1)

        # Use actuator count for control dimension (Panda arm: 7)
        self.Kp_ = kp[: self.nu] if kp.size >= self.nu else np.pad(kp, (0, self.nu - kp.size))
        self.Kd_ = kd[: self.nu] if kd.size >= self.nu else np.pad(kd, (0, self.nu - kd.size))

        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=self.nu)
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=self.nq + self.nv)
        self.DeclareVectorOutputPort("tau_u", size=self.nu, calc=self._calc_tau)

        self._ctx = plant.CreateDefaultContext()

    def _calc_tau(self, context, output):
        q_d = np.asarray(self._desired_state_port.Eval(context), dtype=float).reshape(-1)
        x = np.asarray(self._current_state_port.Eval(context), dtype=float).reshape(-1)

        q = x[: self.nq]
        v = x[self.nq : self.nq + self.nv]

        # Update internal context for gravity
        self.plant.SetPositions(self._ctx, self.model_instance, q)
        g = -self.plant.CalcGravityGeneralizedForces(self._ctx)
        g = np.asarray(g, dtype=float).reshape(-1)

        tau = self.Kp_ * (q_d[: self.nu] - q[: self.nu]) - self.Kd_ * v[: self.nu]
        if g.size >= self.nu:
            tau = tau + g[: self.nu]

        output.SetFromVector(tau.tolist())


