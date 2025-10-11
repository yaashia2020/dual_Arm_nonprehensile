import numpy as np
from pydrake.all import MathematicalProgram, Solve

# -------------------------------------------------------------
# Parameters
# -------------------------------------------------------------
K = 1000.0   # virtual stiffness [N/m]
z = 0.0      # current compression [m]
m = 2.0      # mass [kg]
g = 9.81     # gravity [m/s^2]
mu = 0.5     # friction coefficient

# Desired object acceleration [xdd, ydd, zdd]
xdd_des = np.array([0.0, 0.0, 0.0])  # no acceleration, just gravity compensation

# Build full 6D desired wrench (force + torque)
# [Fx, Fy, Fz, Mx, My, Mz]
h_c = np.concatenate([m * xdd_des + np.array([0.0, 0.0, -m * g]), np.zeros(3)])

# Geometry (contact points in world frame)
w = 0.2  # half-width of box [m]
p_L = np.array([-w/2, 0, 0])
p_R = np.array([ w/2, 0, 0])
p_box = np.array([0, 0, 0])

# -------------------------------------------------------------
# Grasp matrix (6×6): maps contact forces to net wrench
# -------------------------------------------------------------
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

W = np.zeros((6, 6))
W[0:3, 0:3] = np.eye(3)
W[3:6, 0:3] = skew(p_L - p_box)
W[0:3, 3:6] = np.eye(3)
W[3:6, 3:6] = skew(p_R - p_box)

# -------------------------------------------------------------
# Minimum-norm force solution
# -------------------------------------------------------------
Wdag = np.linalg.pinv(W)
h_min = Wdag @ h_c

# Internal-force subspace (equal and opposite squeeze along X)
V = np.array([[1, 0, 0, -1, 0, 0]]).T  # shape (6×1)

# -------------------------------------------------------------
# Optimization
# -------------------------------------------------------------
prog = MathematicalProgram()
z_d = prog.NewContinuousVariables(1, "z_d")[0]  # virtual displacement

# Internal force
f_int = K * (z_d - z)

# Total contact forces (6-vector)
h = h_min + V.flatten() * f_int
fLx, fLy, fLz, fRx, fRy, fRz = h

# Friction cone constraints (3D)
prog.AddConstraint((fLy**2 + fLz**2)**0.5 <= mu * fLx)   # left
prog.AddConstraint((fRy**2 + fRz**2)**0.5 <= mu * (-fRx))  # right (normal flips)

# Normal compressive force positivity
prog.AddConstraint(fLx >= 0)
prog.AddConstraint(fRx <= 0)  # right pushes negative X

# Objective: minimize squeeze displacement
prog.AddCost((z_d - z)**2)

# Initial guess
prog.SetInitialGuess(z_d, 0.01)

# -------------------------------------------------------------
# Solve
# -------------------------------------------------------------
result = Solve(prog)

if result.is_success():
    z_d_opt = result.GetSolution(z_d)
    f_int_opt = K * (z_d_opt - z)
    h_opt = h_min + V.flatten() * f_int_opt

    fL = h_opt[:3]
    fR = h_opt[3:]
    print("✅ Optimal 3D grasp equilibrium found!")
    print(f"Virtual squeeze displacement z_d = {z_d_opt:.6f} m")
    print(f"Internal force = {f_int_opt:.2f} N")
    print(f"Left contact  = {fL}")
    print(f"Right contact = {fR}")
    print(f"Net wrench residual: {np.round(W @ h_opt - h_c, 6)}")
else:
    print("❌ Optimization failed (infeasible).")
