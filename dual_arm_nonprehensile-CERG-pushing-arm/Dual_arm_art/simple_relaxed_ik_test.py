#!/usr/bin/env python3
"""
Simple Relaxed IK Test Script
This script demonstrates basic Relaxed IK functionality without Drake integration.
"""

import sys
import os

wrapper_dir = "/home/yaashia/dual_arm_nonprehensile/relaxed_ik_core/wrappers"
sys.path.insert(0, wrapper_dir)

from python_wrapper import RelaxedIKRust
# Your ctypes wrapper class


def test_relaxed_ik_basic():
    """Test basic Relaxed IK functionality"""
    
    print("Initializing Relaxed IK...")
    
    # Use our custom settings file with custom initial pose
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_settings.yaml"))
    print(f"Using custom config path: {config_path}")
    
    try:
        rik = RelaxedIKRust(setting_file_path=config_path)
        print("Relaxed IK initialized successfully with custom settings!")

        # Single target pose components
        position = [0.5, 0.0, 0.5]
        orientation = [0.0, 0.0, 0.0, 1.0]  # quaternion xyzw
        tolerance = [0.0]*6  # x,y,z and rx,ry,rz tolerances
        
        joint_angles = rik.solve_position(position, orientation, tolerance)
        print(f"Solved joint angles: {joint_angles}")

        # Multiple targets example
        multiple_targets = [
            ([0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]),
            ([0.6, 0.1, 0.4], [0.0, 0.0, 0.0, 1.0]),
            ([0.4, -0.1, 0.6], [0.0, 0.0, 0.0, 1.0])
        ]

        print("Solving for multiple targets...")
        for i, (pos, ori) in enumerate(multiple_targets):
            joint_angles = rik.solve_position(pos, ori, tolerance)
            print(f"Target {i+1}: {pos} -> Joint angles: {joint_angles}")
    
    except Exception as e:
        print(f"Error initializing Relaxed IK: {e}")
        print("Check that the config YAML exists and wrapper is properly imported")


def test_dual_arm_setup():
    """Test dual-arm Relaxed IK setup"""
    
    print("\n=== Testing Dual-Arm Setup ===")
    
    # Use our custom settings file with custom initial pose
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_settings.yaml"))
    print(f"Using custom config path for dual-arm: {config_path}")
    
    try:
        left_arm_rik = RelaxedIKRust(setting_file_path=config_path)
        right_arm_rik = RelaxedIKRust(setting_file_path=config_path)
        print("Dual-arm Relaxed IK initialized with custom settings!")

        left_target_pos = [0.3, 0.2, 0.5]
        left_target_ori = [0.0, 0.0, 0.0, 1.0]
        right_target_pos = [0.3, -0.2, 0.5]
        right_target_ori = [0.0, 0.0, 0.0, 1.0]
        tolerance = [0.0]*6

        left_joints = left_arm_rik.solve_position(left_target_pos, left_target_ori, tolerance)
        right_joints = right_arm_rik.solve_position(right_target_pos, right_target_ori, tolerance)
        print(f"Left arm joints: {left_joints}")
        print(f"Right arm joints: {right_joints}")

        print("\n--- Alternative: Single solver approach ---")
        rik = RelaxedIKRust(setting_file_path=config_path)

        left_joints_alt = rik.solve_position(left_target_pos, left_target_ori, tolerance)
        right_joints_alt = rik.solve_position(right_target_pos, right_target_ori, tolerance)
        print(f"Left arm joints (single solver): {left_joints_alt}")
        print(f"Right arm joints (single solver): {right_joints_alt}")
        
    except Exception as e:
        print(f"Error with dual-arm setup: {e}")
        print("Check that the config YAML exists and wrapper is properly imported")


def create_sample_config():
    """Create a sample configuration file for the Panda FR3 with Relaxed IK"""
    
    sample_config = """# Custom Relaxed IK config for Panda FR3
urdf: panda.urdf
link_radius: 0.05 
base_links:
  - world
ee_links:
  - panda_hand
starting_config: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # Custom initial pose
obstacles:
"""
    
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_settings/panda_settings.yaml"))
    with open(config_file, 'w') as f:
        f.write(sample_config)
    
    print(f"Sample configuration saved to: {config_file}")
    print("Edit this file if you want to adjust optimization or collision settings.")


if __name__ == "__main__":
    print("=== Relaxed IK Simple Test ===")
    
    # Create sample configuration
    create_sample_config()
    
    # Test basic functionality
    test_relaxed_ik_basic()
    
    # Test dual-arm setup
    test_dual_arm_setup()
    
    print("\n=== Test Complete ===")
