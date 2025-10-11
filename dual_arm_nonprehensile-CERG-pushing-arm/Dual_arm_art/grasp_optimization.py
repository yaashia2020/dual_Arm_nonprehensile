import time
from pydrake.all import *
import os
import numpy as np

import numpy as np
from pydrake.all import MathematicalProgram, Solve

# Parameters
K = 1000.0     # stiffness [N/m]
z = 0.0        # current displacement [m]
m = 2.0        # object mass [kg]
g = 9.81       # gravity [m/s^2]
mu = 0.3       # friction coefficient

# Desired object acceleration (lift up + move in y)
xdd_des = np.array([0.0, 2.0, 1.0])  # [xdd, ydd, zdd]
h_c = m * xdd_des + np.array([0.0, 0.0, -m * g])

# Define grasp map W (3x6)
W = np.array([
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1]
])

# Minimum-norm wrench distribution
Wdag = np.linalg.pinv(W)
h_min = Wdag @ h_c

# Internal force basis
V = np.array([[1, 0, 0, 1, 0, 0]]).T  # shape (6, 1)

# Create optimization program
prog = MathematicalProgram()

# Decision variable: virtual displacement z_d
z_d = prog.NewContinuousVariables(1, "z_d")[0]

# Internal force
f_int = K * (z_d - z)

# Total contact wrench
h = h_min + V.flatten() * f_int  # shape (6,)

# Split left/right contact forces
fLx, fLy, fLz, fRx, fRy, fRz = h

# --- Friction cone constraints ---
prog.AddConstraint((fLy**2 + fLz**2)**0.5 <= mu * fLx)
prog.AddConstraint((fRy**2 + fRz**2)**0.5 <= mu * fRx)

# --- Nonnegativity constraints ---
prog.AddConstraint(fLx >= 0)
prog.AddConstraint(fRx >= 0)

# --- Objective: minimize squeeze displacement ---
prog.AddCost((z_d - z)**2)

# Initial guess
prog.SetInitialGuess(z_d, 0.01)

# Solve using IPOPT (if available)
result = Solve(prog)

# --- Extract results ---
if result.is_success():
    z_d_opt = result.GetSolution(z_d)
    f_int_opt = K * (z_d_opt - z)
    h_opt = h_min + V.flatten() * f_int_opt

    print(f"✅ Optimal squeeze z_d = {z_d_opt:.6f} m")
    print(f"Left contact = [{h_opt[0]:.2f}, {h_opt[1]:.2f}, {h_opt[2]:.2f}] N")
    print(f"Right contact = [{h_opt[3]:.2f}, {h_opt[4]:.2f}, {h_opt[5]:.2f}] N")
else:
    print("❌ Optimization failed.")
