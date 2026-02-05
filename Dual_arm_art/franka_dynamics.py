#!/usr/bin/env python3
"""
Franka Robot Dynamics Calculator

This script computes and prints the mass matrix, Coriolis vector, gravity vector,
and forward kinematics for the Franka robot given joint values.

Usage:
    python franka_dynamics.py [joint_values] [joint_velocities]
    
    If no joint values are provided, default values will be used.
    Joint values should be provided as space-separated floats (7 values for 7-DOF arm).
    Joint velocities are optional. If provided, must be 7 values (rad/s).
    
Examples:
    python franka_dynamics.py 0.0 -0.785 0.0 -2.356 0.0 1.571 0.785
    python franka_dynamics.py 0.0 -0.785 0.0 -2.356 0.0 1.571 0.785 0.1 0.2 0.0 0.0 0.0 0.0 0.0
"""

import numpy as np
import sys
import os
from pydrake.all import *

def compute_franka_dynamics(joint_values, joint_velocities=None):
    """
    Compute mass matrix, Coriolis vector, gravity vector, and forward kinematics
    for the Franka robot.
    
    Args:
        joint_values: Array-like of 7 joint values (radians)
        joint_velocities: Array-like of 7 joint velocities (rad/s), optional. 
                         If None, velocities are set to zero.
    
    Returns:
        Dictionary containing:
            - mass_matrix: 7x7 mass matrix
            - coriolis: 7x1 Coriolis vector
            - gravity: 7x1 gravity vector
            - fk_position: 3x1 end-effector position [x, y, z]
            - fk_quaternion: 4x1 end-effector quaternion [w, x, y, z]
            - fk_rotation_matrix: 3x3 rotation matrix
    """
    # Load the Franka robot model
    urdf_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../models/robots/panda_fr3/urdf/panda_drake.urdf")
    )
    urdf = "file://" + urdf_path
    
    # Create plant and scene graph
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    arm = parser.AddModelsFromUrl(urdf)
    plant.Finalize()
    
    # Get the panda model instance
    panda_id = plant.GetModelInstanceByName("panda")
    
    # Create context
    context = plant.CreateDefaultContext()
    
    # Set joint positions
    joint_values = np.array(joint_values)
    if len(joint_values) != 7:
        raise ValueError(f"Expected 7 joint values, got {len(joint_values)}")
    
    plant.SetPositions(context, panda_id, joint_values)
    
    # Set joint velocities (default to zero if not provided)
    if joint_velocities is None:
        joint_velocities = np.zeros(7)
    else:
        joint_velocities = np.array(joint_velocities)
        if len(joint_velocities) != 7:
            raise ValueError(f"Expected 7 joint velocities, got {len(joint_velocities)}")
    
    plant.SetVelocities(context, panda_id, joint_velocities)
    
    # Compute mass matrix
    mass_matrix = plant.CalcMassMatrix(context)
    mass_matrix_np = np.array(mass_matrix)
    
    # Compute gravity vector
    # CalcGravityGeneralizedForces returns the generalized forces needed to counteract gravity
    # For the dynamics equation M*qddot + C*qdot + g = tau, we want g(q)
    gravity_full = plant.CalcGravityGeneralizedForces(context)
    gravity_np = np.array(gravity_full).flatten()
    
    # Extract only the robot portion (first 7 elements)
    if len(gravity_np) > 7:
        gravity_np = gravity_np[:7]
    
    # Compute Coriolis vector
    # Bias term = C(q,v)*v + g(q)
    # So: Coriolis vector = C(q,v)*v = bias_term - g(q)
    bias_term = plant.CalcBiasTerm(context)
    bias_term_np = np.array(bias_term)
    
    # Extract only the robot portion (first 7 elements)
    if len(bias_term_np) > 7:
        bias_term_np = bias_term_np[:7]
    
    # Coriolis vector = bias_term - gravity
    coriolis_np = bias_term_np
    
    # Forward kinematics - get end-effector pose
    # Try to find the end-effector link (panda_link7 or panda_hand)
    ee_body = None
    ee_link_names = ["rubber_pad", "panda_hand", "panda_link7"]
    
    for link_name in ee_link_names:
        try:
            ee_body = plant.GetBodyByName(link_name, panda_id)
            break
        except:
            continue
    
    if ee_body is None:
        # Fallback: get the last body
        bodies = plant.GetBodiesKinematicallyAffectedTo(plant.world_frame())
        for body in bodies:
            if body.model_instance() == panda_id:
                ee_body = body
        if ee_body is None:
            raise RuntimeError("Could not find end-effector body")
    
    # Get end-effector pose
    ee_pose = plant.EvalBodyPoseInWorld(context, ee_body)
    ee_position = ee_pose.translation()
    ee_rotation = ee_pose.rotation()
    
    # Convert rotation to quaternion [w, x, y, z]
    ee_quat = ee_rotation.ToQuaternion().wxyz()
    
    # Get rotation matrix
    ee_rotation_matrix = ee_rotation.matrix()
    
    return {
        'mass_matrix': mass_matrix_np,
        'coriolis': coriolis_np,
        'gravity': gravity_np,
        'fk_position': np.array([ee_position[0], ee_position[1], ee_position[2]]),
        'fk_quaternion': np.array([ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]]),  # [w, x, y, z]
        'fk_rotation_matrix': ee_rotation_matrix,
        'ee_body_name': ee_body.name()
    }


def print_dynamics_results(results, joint_values, joint_velocities=None):
    """Print the computed dynamics results in a formatted way."""
    print("=" * 80)
    print("FRANKA ROBOT DYNAMICS COMPUTATION")
    print("=" * 80)
    print(f"\nInput Joint Values (rad):")
    for i, q in enumerate(joint_values):
        print(f"  Joint {i+1}: {q:8.4f} rad ({np.degrees(q):7.2f} deg)")
    
    if joint_velocities is not None:
        print(f"\nInput Joint Velocities (rad/s):")
        for i, dq in enumerate(joint_velocities):
            print(f"  Joint {i+1}: {dq:8.4f} rad/s ({np.degrees(dq):7.2f} deg/s)")
    else:
        print(f"\nInput Joint Velocities (rad/s): All zeros")
    
    print(f"\nEnd-Effector Body: {results['ee_body_name']}")
    
    print("\n" + "-" * 80)
    print("FORWARD KINEMATICS (End-Effector Pose)")
    print("-" * 80)
    print(f"\nPosition [m]:")
    print(f"  X: {results['fk_position'][0]:8.4f}")
    print(f"  Y: {results['fk_position'][1]:8.4f}")
    print(f"  Z: {results['fk_position'][2]:8.4f}")
    
    print(f"\nQuaternion [w, x, y, z]:")
    q = results['fk_quaternion']
    print(f"  w: {q[0]:8.4f}")
    print(f"  x: {q[1]:8.4f}")
    print(f"  y: {q[2]:8.4f}")
    print(f"  z: {q[3]:8.4f}")
    
    print(f"\nRotation Matrix:")
    R = results['fk_rotation_matrix']
    print(f"  [{R[0,0]:8.4f}  {R[0,1]:8.4f}  {R[0,2]:8.4f}]")
    print(f"  [{R[1,0]:8.4f}  {R[1,1]:8.4f}  {R[1,2]:8.4f}]")
    print(f"  [{R[2,0]:8.4f}  {R[2,1]:8.4f}  {R[2,2]:8.4f}]")
    
    print("\n" + "-" * 80)
    print("MASS MATRIX (7x7)")
    print("-" * 80)
    M = results['mass_matrix']
    print("\nMass Matrix [kg⋅m²]:")
    for i in range(7):
        row_str = "  "
        for j in range(7):
            row_str += f"{M[i,j]:10.4f}  "
        print(row_str)
    
    print("\n" + "-" * 80)
    print("CORIOLIS VECTOR (7x1)")
    print("-" * 80)
    C = results['coriolis']
    print("\nCoriolis Vector [N⋅m]:")
    for i in range(7):
        print(f"  Joint {i+1}: {C[i]:10.4f}")
    
    print("\n" + "-" * 80)
    print("GRAVITY VECTOR (7x1)")
    print("-" * 80)
    G = results['gravity']
    print("\nGravity Vector [N⋅m]:")
    for i in range(7):
        print(f"  Joint {i+1}: {G[i]:10.4f}")
    
    print("\n" + "=" * 80)


def main():
    """Main function to parse arguments and compute dynamics."""
    # Default joint values (in radians)
    default_joint_values = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    
    # Parse command line arguments
    joint_values = None
    joint_velocities = None
    
    if len(sys.argv) > 1:
        try:
            all_values = [float(x) for x in sys.argv[1:]]
            
            if len(all_values) == 7:
                # Only joint positions provided
                joint_values = all_values
            elif len(all_values) == 14:
                # Both joint positions and velocities provided
                joint_values = all_values[:7]
                joint_velocities = all_values[7:14]
            else:
                print(f"Error: Expected 7 joint values, or 7 joint values + 7 joint velocities")
                print(f"       Got {len(all_values)} values")
                print(f"Usage: python {sys.argv[0]} [q1 q2 q3 q4 q5 q6 q7] [dq1 dq2 dq3 dq4 dq5 dq6 dq7]")
                print(f"Example (positions only): python {sys.argv[0]} 0.0 -0.785 0.0 -2.356 0.0 1.571 0.785")
                print(f"Example (with velocities): python {sys.argv[0]} 0.0 -0.785 0.0 -2.356 0.0 1.571 0.785 0.1 0.2 0.0 0.0 0.0 0.0 0.0")
                sys.exit(1)
        except ValueError:
            print("Error: All values must be numbers")
            print(f"Usage: python {sys.argv[0]} [q1 q2 q3 q4 q5 q6 q7] [dq1 dq2 dq3 dq4 dq5 dq6 dq7]")
            sys.exit(1)
    else:
        joint_values = default_joint_values
        print("No joint values provided, using defaults:")
        print(f"  Positions: {joint_values}")
        print(f"  Velocities: All zeros")
        print()
    
    try:
        # Compute dynamics
        results = compute_franka_dynamics(joint_values, joint_velocities)
        
        # Print results
        print_dynamics_results(results, joint_values, joint_velocities)
        
    except Exception as e:
        print(f"Error computing dynamics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

