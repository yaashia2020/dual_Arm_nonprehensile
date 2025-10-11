import numpy as np

class Controllerv2:
    def __init__(self, ro, m, L, kp, kd):
        self.ro = ro  # Wheel radius
        self.m = m    # Mass of the mobile robot
        self.L = L    # Distance between the wheels
        
        # Control gains
        self.kp1, self.kp2, self.kp3 = kp
        self.kd1, self.kd2, self.kd3 = kd
        
        # Desired state
        self.xdes = 0
        self.ydes = 0
        self.thdes = 0
        
        # Current state
        self.x = 0
        self.y = 0
        self.th = 0
        self.thetadot = 0
        self.xd = 0
        self.yd = 0
        
    def set_desired_state(self, xdes, ydes):
        self.xdes = xdes
        self.ydes = ydes
        self.thdes = np.arctan2(ydes, xdes)
        
    def update_current_state(self, x, y, th, thetadot, xd, yd):
        self.x = x
        self.y = y
        self.th = th
        self.thetadot = thetadot
        self.xd = xd
        self.yd = yd
        
    def calculate_errors(self):
        e_x = self.xdes - self.x
        e_y = self.ydes - self.y
        e_th = self.thdes - self.th
        return e_x, e_y, e_th
        
    def compute_torques(self):
        e_x, e_y, e_th = self.calculate_errors()
        
        Tl = (self.ro * (2 * e_th * self.kp3 - 2 * self.kd3 * self.thetadot + self.L * e_x * self.kp1 * np.cos(self.th) + self.L * e_y * self.kp2 * np.sin(self.th) - self.L * self.kd1 * self.xd * np.cos(self.th) - self.L * self.kd2 * self.yd * np.sin(self.th))) / (2 * self.L)
        Tr = -(self.ro * (2 * e_th * self.kp3 - 2 * self.kd3 * self.thetadot - self.L * e_x * self.kp1 * np.cos(self.th) - self.L * e_y * self.kp2 * np.sin(self.th) + self.L * self.kd1 * self.xd * np.cos(self.th) + self.L * self.kd2 * self.yd * np.sin(self.th))) / (2 * self.L)
        
        return Tl, Tr

# # Example usage
# if __name__ == "__main__":
#     # Constants
#     ro = 0.1  # Example value, replace with actual radius
#     m = 1.0   # Example value, total mass of the mobile robot
#     L = 0.5   # Example value, distance between the wheels

#     # Control gains
#     kp = [1, 1, 1]
#     kd = [2, 2, 0.5]

#     # Initialize controller
#     controller = Controllerv2(ro, m, L, kp, kd)

#     # Set desired state
#     controller.set_desired_state(0.9, 0)

#     # Update current state
#     controller.update_current_state(0, 0, 0, 0, 0, 0)

#     # Compute torques
#     Tl, Tr = controller.compute_torques()

#     print(f"Torques at the wheels: Tl = {Tl}, Tr = {Tr}")
