# Ah, this is fantastic clarification! The 29 DOF list and the explanation about the G1's wrist joints are exactly what we need to make the dof_positions conversion robust.

# Understanding the G1 Joint Structure for dof_positions:

# Your joint_names list (29 entries) implies that each of these is a 1-DOF revolute joint. This is a common setup for robot arms and legs.

# dof_positions shape (N, 29) in the working NPZ: This strongly suggests that Isaac Sim expects a single scalar value (an angle) for each of these 29 joints.
# G1's "funny thing" with wrist joints: The fact that G1 breaks down a typical 3-axis wrist into three separate 1-DOF joints (_roll_joint, _pitch_joint, _yaw_joint) confirms that each of these is indeed expected to be represented by a single angle.

# The Problem with my previous placeholder dof_positions = dof_positions_euler[:, :, 2]:

# My previous placeholder was a simplification. It assumed that for every joint, the relevant angle could be extracted from the Z-axis of an Euler conversion of the local quaternion. This is unlikely to be universally true for all 29 joints, as different joints will have different primary rotation axes.

# The Correct Approach for dof_positions:

# We need to extract the single scalar angle for each of the 29 revolute joints from the poselib target_motion.local_rotation quaternions. This requires knowing the axis of rotation for each joint in its local frame.

# poselib's SkeletonTree should contain this information. Specifically, the SkeletonTree (which target_motion.skeleton_tree is an instance of) has a local_orientation attribute. This local_orientation is typically the quaternion representing the joint's default orientation relative to its parent. We need the axis of rotation for each joint.

# Let's assume for a moment that poselib's local_rotation quaternions directly represent the rotation around the joint's primary axis. If so, we can convert these quaternions to axis-angle representation and extract the angle.

# Revised Strategy for dof_positions:

# Identify the primary axis of rotation for each of the 29 joints. This information is usually in the URDF, or poselib might expose it.
# For each joint's local rotation quaternion, convert it to axis-angle.
# Extract the angle component.

# Let's look at poselib's SkeletonState and SkeletonMotion more closely.

# The SkeletonState and SkeletonMotion objects store local_rotation as quaternions. To get the scalar angle for a revolute joint, you typically need to:

# Know the joint's axis of rotation in its parent's frame (or its own local frame).
# Convert the quaternion to an axis-angle representation.
# Project the rotation axis from the quaternion onto the known joint axis.

# However, poselib might have a more direct way to get these. Let's check the poselib source or documentation (or make a reasonable assumption if not readily available).

# Assumption (most common for revolute joints): For a single revolute joint, its local rotation quaternion (w, x, y, z) represents a rotation around its axis. If the axis is, say, (0, 0, 1) (Z-axis), then the angle theta can often be derived from w and z (or w and x/y depending on the axis).

# A more robust way to get joint angles from poselib's SkeletonMotion might be to leverage its internal structure if it stores joint limits or axis information.

# Let's try to get the angles from poselib's local_rotation quaternions, assuming they represent rotations around a single axis for each of your 29 joints.

# The poselib SkeletonMotion object's local_rotation is (N, D, 4). If D is 29, and these are all 1-DOF joints, then each (w, x, y, z) quaternion for a specific joint represents its rotation. We need to extract the scalar angle from this.

# Proposed Update to dof_positions in motion_retargeting_and_npz_export:

# Instead of converting to Euler angles and picking one, we should try to extract the single angle.

# ... (inside motion_retargeting_and_npz_export) ...

# 1. dof_names: Already available from target_motion.skeleton_tree
dof_names = np.array(target_motion.skeleton_tree.node_names, dtype=np.str_)

# 2. dof_positions: Joint angles (scalar values for each of the 29 revolute joints)
# target_motion.local_rotation is (N, D, 4) in (w, x, y, z) order
dof_positions_quat = target_motion.local_rotation.cpu().numpy() # (N, 29, 4)

# This is the critical part: Extracting the scalar angle for each 1-DOF joint.
# This requires knowing the rotation axis for each joint.
# If poselib's SkeletonTree or Model doesn't directly expose the axis for each joint,
# we might have to infer it from the URDF or make an assumption.
# A common convention for revolute joints is that the quaternion directly encodes
# the rotation around its single axis.

# Let's assume for now that for each joint, the quaternion (w,x,y,z) represents
# a rotation around a single axis. We can convert this to axis-angle and take the angle.
# This might need refinement based on the actual URDF joint axes.

dof_positions = np.zeros((N, dof_names.shape[0]), dtype=np.float64) # (N, 29)
for i in range(N):
    for j in range(dof_names.shape[0]):
        quat_w_xyz = dof_positions_quat[i, j, :] # (w, x, y, z)
        
        # Convert quaternion to axis-angle (scipy Rotation expects (x,y,z,w))
        quat_xyzw = np.array([quat_w_xyz[1], quat_w_xyz[2], quat_w_xyz[3], quat_w_xyz[0]])
        rotation = R.from_quat(quat_xyzw)
        
        # Extract the angle. For a single revolute joint, this is its scalar angle.
        # The axis-angle representation is (axis_x, axis_y, axis_z, angle_radians)
        # We just need the angle_radians part.
        # Note: The axis part is important if the joint's axis is not canonical (e.g., not pure X, Y, or Z).
        # For simplicity, we'll assume the angle directly from the quaternion is the scalar value.
        # This is often `2 * atan2(norm(xyz), w)`.
        
        # More robust way to get scalar angle for a single revolute joint:
        # If the joint is truly 1-DOF, its quaternion will effectively be a rotation
        # around a fixed axis. We can extract the angle from this.
        # For a quaternion q = (w, x, y, z), the angle theta is 2 * arccos(w).
        # The sign of the angle depends on the axis.
        # Let's try extracting the angle directly, assuming the quaternion is aligned.
        
        # This is a common way to get the scalar angle from a quaternion
        # for a single revolute joint, assuming the quaternion represents
        # rotation around the joint's single axis.
        # It gives the magnitude of the rotation. The sign might need adjustment
        # if the joint's positive direction is not aligned with the quaternion's axis.
        angle_rad = 2 * np.arctan2(np.linalg.norm(quat_w_xyz[1:]), quat_w_xyz[0])
        
        # Handle potential wrap-around or sign issues if needed.
        # For example, if the joint is expected to rotate around Z,
        # and the quaternion is (cos(theta/2), 0, 0, sin(theta/2)),
        # then theta is 2 * asin(z) or 2 * atan2(z, w).
        # This depends heavily on the URDF's joint axis definition.
        
        # Given the (N, 29) shape, we need a single scalar.
        # A common simplification for 1-DOF joints is to take the angle
        # associated with the primary axis of rotation.
        # If the joint's axis is Z, the angle is 2 * atan2(q_z, q_w).
        # If the joint's axis is Y, the angle is 2 * atan2(q_y, q_w).
        # If the joint's axis is X, the angle is 2 * atan2(q_x, q_w).
        
        # Without explicit joint axis info from poselib, this is an educated guess.
        # Let's assume the joint's primary axis corresponds to the largest component
        # of the quaternion's vector part (x,y,z) or is implicitly handled by poselib.
        # A safer bet is to use the `as_euler` and select the appropriate axis.
        # Given your working NPZ has (N, 29), it implies each joint has a single scalar value.
        # Let's revert to a simplified Euler extraction for now, but acknowledge it's a potential point of failure.
        # The best way is to get the joint axis from target_motion.skeleton_tree if available.
        
        # For now, let's use the Euler conversion, and assume the primary angle is what's needed.
        # This is a common source of errors if the Euler order or axis doesn't match the URDF.
        # Reordering to (x,y,z,w) for scipy.spatial.transform.Rotation
        quat_xyzw_for_scipy = np.array([quat_w_xyz[1], quat_w_xyz[2], quat_w_xyz[3], quat_w_xyz[0]])
        euler_angles = R.from_quat(quat_xyzw_for_scipy).as_euler('xyz', degrees=False)
        
        # Which Euler angle corresponds to the single DOF? This is the core problem.
        # For example, for 'left_knee_joint', it's usually a pitch (Y-axis) rotation.
        # For 'left_hip_roll_joint', it's a roll (X-axis) rotation.
        # For 'left_hip_yaw_joint', it's a yaw (Z-axis) rotation.
        
        # This requires a mapping from joint name to its primary axis.
        # Let's create a simple mapping based on common conventions.
        # This is still an assumption, but more informed.
        joint_axis_map = {
            "left_hip_pitch_joint": 1,  # Y-axis
            "left_hip_roll_joint": 0,   # X-axis
            "left_hip_yaw_joint": 2,    # Z-axis
            "left_knee_joint": 1,       # Y-axis
            "left_ankle_pitch_joint": 1,# Y-axis
            "left_ankle_roll_joint": 0, # X-axis
            "right_hip_pitch_joint": 1, # Y-axis
            "right_hip_roll_joint": 0,  # X-axis
            "right_hip_yaw_joint": 2,   # Z-axis
            "right_knee_joint": 1,      # Y-axis
            "right_ankle_pitch_joint": 1,# Y-axis
            "right_ankle_roll_joint": 0, # X-axis
            "waist_yaw_joint": 2,       # Z-axis
            "waist_roll_joint": 0,      # X-axis
            "waist_pitch_joint": 1,     # Y-axis
            "left_shoulder_pitch_joint": 1, # Y-axis
            "left_shoulder_roll_joint": 0,  # X-axis
            "left_shoulder_yaw_joint": 2,   # Z-axis
            "left_elbow_joint": 1,      # Y-axis
            "left_wrist_roll_joint": 0, # X-axis
            "left_wrist_pitch_joint": 1,# Y-axis
            "left_wrist_yaw_joint": 2,  # Z-axis
            "right_shoulder_pitch_joint": 1, # Y-axis
            "right_shoulder_roll_joint": 0,  # X-axis
            "right_shoulder_yaw_joint": 2,   # Z-axis
            "right_elbow_joint": 1,      # Y-axis
            "right_wrist_roll_joint": 0, # X-axis
            "right_wrist_pitch_joint": 1,# Y-axis
            "right_wrist_yaw_joint": 2   # Z-axis
        }
        
        # Get the index for the primary Euler angle based on the joint name
        axis_idx = joint_axis_map.get(dof_names[j], 2) # Default to Z if not found
        dof_positions[i, j] = euler_angles[axis_idx]

# Ensure it's float64 as per working NPZ
dof_positions = dof_positions.astype(np.float64) 

# ... (rest of the script) ...


Important Considerations for the joint_axis_map:

This joint_axis_map is an educated guess based on typical humanoid robot joint configurations. The exact primary axis for each joint (and thus which Euler angle to extract) must be verified against the G1 URDF file. Look for the <axis> tag within each <joint> definition in the URDF. For example, <axis xyz="0 1 0"/> means it rotates around the Y-axis.
Euler Angle Order: The as_euler('xyz', degrees=False) assumes an XYZ Euler angle sequence. If your URDF implicitly uses a different sequence (e.g., ZYX), the extracted angles will be incorrect. This is another point to verify.

Revised xsens_fbx_to_g1_npz.py (with dof_positions fix and isaac_body_names verification):

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Humanoid Motion Data Converter (FBX to Isaac Lab NPZ)

This script converts humanoid motion data from FBX format (Xsens) to NPZ format
suitable for Isaac Lab's Adversarial Motion Prior (AMP) training.
It leverages the Rofunc poselib for FBX import and retargeting,
and then processes the motion data into the required NPZ structure.

USAGE:
python xsens_fbx_to_g1_npz.py --fbx_file <path_to_your_fbx_file.fbx>

REQUIREMENTS:
- rofunc (and its dependencies like poselib, torch)
- numpy
- scipy (for gaussian_filter1d)
"""

import multiprocessing
import os
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

import rofunc as rf
from rofunc.utils.datalab.poselib.poselib.core.rotation3d import *
from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from rofunc.utils.datalab.poselib.poselib.visualization.common import plot_skeleton_motion_interactive, \
plot_skeleton_state

# --- Helper Functions (from your data_convert.py, adapted for poselib output) ---
def quaternion_inverse(q):
"""Input q: (w, x, y, z), returns its inverse."""
w, x, y, z = q
norm_sq = w*w + x*x + y*y + z*z
if norm_sq < 1e-8:
    norm_sq = 1e-8
return np.array([w, -x, -y, -z], dtype=q.dtype) / norm_sq

def quaternion_multiply(q1, q2):
"""Input/output: (w, x, y, z)"""
w1, x1, y1, z1 = q1
w2, x2, y2, z2 = q2
w = w1*w2 - x1*x2 - y1*y2 - z1*z2
x = w1*x2 + x1*w2 + y1*z2 - z1*y2
y = w1*y2 - x1*z2 + y1*w2 + z1*x2
z = w1*z2 + x1*y2 - y1*x2 + z1*w2
return np.array([w, x, y, z], dtype=q1.dtype)

def compute_angular_velocity(q_prev, q_next, dt, eps=1e-8):
"""
Compute angular velocity from adjacent quaternions (w, x, y, z):
  - Relative rotation q_rel = inv(q_prev) * q_next
  - Extract rotation angle and axis from q_rel
  - Return (angle / dt) * axis
"""
q_inv = quaternion_inverse(q_prev)
q_rel = quaternion_multiply(q_inv, q_next)
norm_q_rel = np.linalg.norm(q_rel)
if norm_q_rel < eps:
    return np.zeros(3, dtype=np.float32)
q_rel /= norm_q_rel

w = np.clip(q_rel[0], -1.0, 1.0)
angle = 2.0 * np.arccos(w)
sin_half = np.sqrt(1.0 - w*w)
if sin_half < eps:
    return np.zeros(3, dtype=np.float32)
axis = q_rel[1:] / sin_half
return (angle / dt) * axis

def calculate_velocities(data_array, dt, sigma=1):
"""
Calculates velocities using central differences and Gaussian smoothing.
Handles 2D (N, D) or 3D (N, B, 3/4) arrays.
"""
velocities = np.zeros_like(data_array)
if data_array.shape[0] > 1:
    velocities[1:-1] = (data_array[2:] - data_array[:-2]) / (2 * dt)
    velocities[0] = (data_array[1] - data_array[0]) / dt
    velocities[-1] = (data_array[-1] - data_array[-2]) / dt
else: # Handle single frame case
    velocities[0] = np.zeros_like(data_array[0])
return gaussian_filter1d(velocities, sigma=sigma, axis=0)

# --- Main Functions ---

def motion_from_fbx(fbx_file_path, root_joint, fps=60, visualize=True):
# import fbx file - make sure to provide a valid joint name for root_joint
motion = SkeletonMotion.from_fbx(
    fbx_file_path=fbx_file_path,
    root_joint=root_joint,
    fps=fps
)
if visualize:
    rf.logger.beauty_print(f"Plot Source FBX Motion ({fbx_file_path})", type="module")
    plot_skeleton_motion_interactive(motion, verbose=False)
return motion

def motion_retargeting_and_npz_export(retarget_cfg, source_motion, output_npz_filepath, visualize=True):
# load and visualize t-pose files
source_tpose = SkeletonState.from_file(retarget_cfg["source_tpose"])
if visualize:
    rf.logger.beauty_print("Plot Xsens T-pose", type="module")
    plot_skeleton_state(source_tpose)

target_tpose = SkeletonState.from_file(retarget_cfg["target_tpose"])
if visualize:
    rf.logger.beauty_print("Plot G1 T-pose", type="module")
    plot_skeleton_state(target_tpose, verbose=True)

# parse data from retarget config
rotation_to_target_skeleton = torch.tensor(retarget_cfg["rotation"])

# run retargeting
target_motion = source_motion.retarget_to_by_tpose(
    joint_mapping=retarget_cfg["joint_mapping"],
    source_tpose=source_tpose,
    target_tpose=target_tpose,
    rotation_to_target_skeleton=rotation_to_target_skeleton,
    scale_to_target_skeleton=retarget_cfg["scale"]
)

# keep frames between [trim_frame_beg, trim_frame_end - 1]
frame_beg = retarget_cfg["trim_frame_beg"]
frame_end = retarget_cfg["trim_frame_end"]
if frame_beg == -1:
    frame_beg = 0
if frame_end == -1:
    frame_end = target_motion.local_rotation.shape[0]

local_rotation = target_motion.local_rotation[frame_beg:frame_end, ...]
root_translation = target_motion.root_translation[frame_beg:frame_end, ...]

# move human to origin (adjusting root_translation based on average)
avg_root_translation_xy = root_translation[:, :2].mean(axis=0)
root_translation[:, :2] -= avg_root_translation_xy # Move X,Y to origin

# Create a temporary SkeletonState to calculate global positions for min_h
temp_sk_state = SkeletonState.from_rotation_and_root_translation(
    target_motion.skeleton_tree, local_rotation, root_translation, is_local=True
)
temp_motion = SkeletonMotion.from_skeleton_state(temp_sk_state, fps=target_motion.fps)

# Adjust height so feet are on the ground
foot_joint_names = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
foot_indices = []
for foot_name in foot_joint_names:
    try:
        # node_indices maps joint names to their index in the flattened array
        foot_indices.append(target_motion.skeleton_tree.node_indices[foot_name])
    except KeyError:
        print(f"Warning: Foot joint '{foot_name}' not found in target skeleton. Skipping ground adjustment.")
        foot_indices = [] # Disable adjustment if any foot joint is missing
        break

if foot_indices:
    # Get global Z positions of foot joints
    # global_translation is (N, num_joints, 3)
    foot_global_z = temp_motion.global_translation[:, foot_indices, 2]
    min_h = torch.min(foot_global_z)
    root_translation[:, 2] += -min_h.item() # Use .item() for scalar tensor

# adjust the height of the root to avoid ground penetration
root_height_offset = retarget_cfg["root_height_offset"]
root_translation[:, 2] += root_height_offset

# Recreate target_motion with adjusted root translation
new_sk_state = SkeletonState.from_rotation_and_root_translation(
    target_motion.skeleton_tree, local_rotation, root_translation, is_local=True
)
target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

# --- NPZ Export Section (integrating data_convert.py logic) ---
N = target_motion.local_rotation.shape[0]
fps = target_motion.fps
dt = 1.0 / fps

# 1. dof_names: Already available from target_motion.skeleton_tree
dof_names = np.array(target_motion.skeleton_tree.node_names, dtype=np.str_)

# 2. dof_positions: Joint angles (scalar values for each of the 29 revolute joints)
# target_motion.local_rotation is (N, D, 4) in (w, x, y, z) order
dof_positions_quat = target_motion.local_rotation.cpu().numpy() # (N, 29, 4)

# Mapping from joint name to its primary rotation axis index (0=X, 1=Y, 2=Z)
# This MUST be verified against your G1 URDF for accuracy.
joint_axis_map = {
    "left_hip_pitch_joint": 1,  # Y-axis
    "left_hip_roll_joint": 0,   # X-axis
    "left_hip_yaw_joint": 2,    # Z-axis
    "left_knee_joint": 1,       # Y-axis
    "left_ankle_pitch_joint": 1,# Y-axis
    "left_ankle_roll_joint": 0, # X-axis
    "right_hip_pitch_joint": 1, # Y-axis
    "right_hip_roll_joint": 0,  # X-axis
    "right_hip_yaw_joint": 2,   # Z-axis
    "right_knee_joint": 1,      # Y-axis
    "right_ankle_pitch_joint": 1,# Y-axis
    "right_ankle_roll_joint": 0, # X-axis
    "waist_yaw_joint": 2,       # Z-axis
    "waist_roll_joint": 0,      # X-axis
    "waist_pitch_joint": 1,     # Y-axis
    "left_shoulder_pitch_joint": 1, # Y-axis
    "left_shoulder_roll_joint": 0,  # X-axis
    "left_shoulder_yaw_joint": 2,   # Z-axis
    "left_elbow_joint": 1,      # Y-axis
    "left_wrist_roll_joint": 0, # X-axis
    "left_wrist_pitch_joint": 1,# Y-axis
    "left_wrist_yaw_joint": 2,  # Z-axis
    "right_shoulder_pitch_joint": 1, # Y-axis
    "right_shoulder_roll_joint": 0,  # X-axis
    "right_shoulder_yaw_joint": 2,   # Z-axis
    "right_elbow_joint": 1,      # Y-axis
    "right_wrist_roll_joint": 0, # X-axis
    "right_wrist_pitch_joint": 1,# Y-axis
    "right_wrist_yaw_joint": 2   # Z-axis
}

dof_positions = np.zeros((N, dof_names.shape[0]), dtype=np.float64) # (N, 29)
for i in range(N):
    for j in range(dof_names.shape[0]):
        quat_w_xyz = dof_positions_quat[i, j, :] # (w, x, y, z)
        
        # Reorder to (x, y, z, w) for scipy.spatial.transform.Rotation
        quat_xyzw_for_scipy = np.array([quat_w_xyz[1], quat_w_xyz[2], quat_w_xyz[3], quat_w_xyz[0]])
        
        # Convert to Euler angles. 'xyz' order is a common convention.
        # If your URDF uses a different implicit order, this needs to change.
        euler_angles = R.from_quat(quat_xyzw_for_scipy).as_euler('xyz', degrees=False)
        
        # Extract the single scalar angle based on the joint's primary axis
        axis_idx = joint_axis_map.get(dof_names[j], -1) # Default to -1 to catch missing mappings
        if axis_idx == -1:
            print(f"Error: Joint '{dof_names[j]}' not found in joint_axis_map. Please update the map.")
            # Fallback: try to guess the most significant axis or raise error
            # For now, let's just use the Z-axis as a fallback, but this is risky.
            dof_positions[i, j] = euler_angles[2] 
        else:
            dof_positions[i, j] = euler_angles[axis_idx]

# Ensure it's float64 as per working NPZ
dof_positions = dof_positions.astype(np.float64) 

# 3. dof_velocities: Calculate from dof_positions
dof_velocities = calculate_velocities(dof_positions, dt, sigma=1).astype(np.float64)

# 4. body_names: Specific links for Isaac Sim.
# These should be a subset of target_motion.skeleton_tree.node_names
# Your provided list has 16 entries.
isaac_body_names = [
    "pelvis", 
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_elbow_link", "right_elbow_link",
    "right_hip_yaw_link", "left_hip_yaw_link",
    "right_rubber_hand", "left_rubber_hand", # These might not be in G1, check your URDF
    "right_ankle_roll_link", "left_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "torso_link",
    "right_knee_link", "left_knee_link"
]
isaac_body_names_array = np.array(isaac_body_names, dtype=np.str_)

# Get indices of these body names in the target_motion's skeleton_tree
body_indices = []
for name in isaac_body_names:
    try:
        body_indices.append(target_motion.skeleton_tree.node_indices[name])
    except KeyError:
        print(f"Error: Isaac body name '{name}' not found in target skeleton. Please check your isaac_body_names list against the G1 URDF.")
        return # Exit if a required body is missing

# 5. body_positions: Global positions of selected bodies
# target_motion.global_translation is (N, num_joints, 3)
body_positions = target_motion.global_translation.cpu().numpy()[:, body_indices, :].astype(np.float32)

# 6. body_rotations: Global rotations of selected bodies
# target_motion.global_rotation is (N, num_joints, 4) (w,x,y,z)
body_rotations = target_motion.global_rotation.cpu().numpy()[:, body_indices, :].astype(np.float32)

# 7. body_linear_velocities: Calculate from body_positions
body_linear_velocities = calculate_velocities(body_positions, dt, sigma=1).astype(np.float32)

# 8. body_angular_velocities: Calculate from body_rotations
body_angular_velocities = np.zeros((N, len(isaac_body_names), 3), dtype=np.float32)
for j in range(len(isaac_body_names)):
    quats = body_rotations[:, j, :] # (N, 4)
    angular_vels = np.zeros((N, 3), dtype=np.float32)
    if N > 1:
        # For angular velocity, we use the provided helper function
        angular_vels[0] = compute_angular_velocity(quats[0], quats[1], dt)
        angular_vels[-1] = compute_angular_velocity(quats[-2], quats[-1], dt)
    for k in range(1, N - 1):
        av1 = compute_angular_velocity(quats[k - 1], quats[k], dt)
        av2 = compute_angular_velocity(quats[k], quats[k + 1], dt)
        angular_vels[k] = 0.5 * (av1 + av2)
    # Smoothing
    body_angular_velocities[:, j, :] = gaussian_filter1d(angular_vels, sigma=1, axis=0)

# 9. Package and save to NPZ
data_dict = {
    "fps": np.array(fps, dtype=np.int32),
    "dof_names": dof_names,
    "body_names": isaac_body_names_array,
    "dof_positions": dof_positions,
    "dof_velocities": dof_velocities,
    "body_positions": body_positions,
    "body_rotations": body_rotations,
    "body_linear_velocities": body_linear_velocities,
    "body_angular_velocities": body_angular_velocities
}

output_dir = os.path.dirname(output_npz_filepath)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

np.savez(output_npz_filepath, **data_dict)

print(f"Conversion completed, data saved to {output_npz_filepath}")
print("fps:", fps)
print("dof_names:", dof_names.shape)
print("body_names:", isaac_body_names_array.shape)
print("dof_positions:", dof_positions.shape)
print("dof_velocities:", dof_velocities.shape)
print("body_positions:", body_positions.shape)
print("body_rotations:", body_rotations.shape)
print("body_linear_velocities:", body_linear_velocities.shape)
print("body_angular_velocities:", body_angular_velocities.shape)

if visualize:
    rf.logger.beauty_print("Plot Retargeted G1 Motion (NPZ output)", type="module")
    plot_skeleton_motion_interactive(target_motion, verbose=False)

def npy_from_fbx(fbx_file):
rofunc_path = rf.oslab.get_rofunc_path()
config = {
    "target_motion_path": fbx_file.replace('_xsens.fbx', '_xsens2g1.npy'), # This .npy file is still saved by poselib's to_file
    "source_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/source_xsens_wo_gloves_tpose.npy"),
    "target_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/target_g1_29dof_tpose.npy"),
    "joint_mapping": { # Left: Xsens, Right: MJCF (or G1 URDF joint names)
        "Hips": "pelvis",
        "LeftUpLeg": "left_hip_pitch_joint",
        "LeftLeg": "left_knee_joint",
        "LeftFoot": "left_ankle_pitch_joint",
        "RightUpLeg": "right_hip_pitch_joint",
        "RightLeg": "right_knee_joint",
        "RightFoot": "right_ankle_pitch_joint",
        "Spine3": "torso_link", # This is a link, not a joint. Poselib might handle this, but be aware.
        "LeftArm": "left_shoulder_pitch_joint",
        "LeftForeArm": "left_elbow_joint",
        "LeftHand": "left_wrist_pitch_joint", # This is a joint, but your body_names had 'left_hand' (link)
        "RightArm": "right_shoulder_pitch_joint",
        "RightForeArm": "right_elbow_joint",
        "RightHand": "right_wrist_pitch_joint" # This is a joint, but your body_names had 'right_hand' (link)
    },
    "rotation": [0.5, 0.5, 0.5, 0.5], # Example rotation, adjust as needed
    "scale": 0.01, # Example scale, adjust as needed
    "root_height_offset": 0.0,
    "trim_frame_beg": 0,
    "trim_frame_end": -1
}

source_motion = motion_from_fbx(fbx_file, root_joint="Hips", fps=60, visualize=False)

# Define the output NPZ path
output_npz_filepath = fbx_file.replace('_xsens.fbx', '_g1_motion.npz')

motion_retargeting_and_npz_export(config, source_motion, output_npz_filepath, visualize=False)

if __name__ == '__main__':
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fbx_file", type=str, default=None, help="Path to the input FBX file.")
args = parser.parse_args()

if args.fbx_file is None:
    print("Error: Please provide an FBX file using --fbx_file argument.")
    exit(1)

# Example usage: python your_script_name.py --fbx_file "path/to/your/xsens_motion.fbx"
npy_from_fbx(args.fbx_file)


Summary of Changes and Verification Points:

joint_axis_map for dof_positions:
I've added a joint_axis_map dictionary. This maps each of your 29 dof_names to an index (0 for X, 1 for Y, 2 for Z) representing its primary rotation axis when converting from quaternion to Euler angles using as_euler('xyz').
CRITICAL VERIFICATION: You must check your G1 URDF file for each of these 29 joints. Find the <joint> tag for each, and within it, look for the <axis> tag.
If <axis xyz="1 0 0"/>, the primary axis is X (index 0).
If <axis xyz="0 1 0"/>, the primary axis is Y (index 1).
If <axis xyz="0 0 1"/>, the primary axis is Z (index 2).
Adjust the joint_axis_map if my assumptions are incorrect for any joint.
The as_euler('xyz', degrees=False) conversion assumes an XYZ Euler sequence. If your G1 URDF (or the way poselib interprets it) implicitly uses a different Euler sequence, this will be incorrect. This is a common source of error in retargeting.
isaac_body_names List:
I've kept your list of 16 body_names.
CRITICAL VERIFICATION: You must ensure these 16 names are the exact names of the links/bodies in your G1 URDF that Isaac Sim expects.
Special Note on "hands": Your joint_mapping maps LeftHand to left_wrist_pitch_joint (a joint). Your isaac_body_names list contains left_rubber_hand (a link/body). These are different. Make sure you are consistently referring to either joints or links as appropriate for each context. If Isaac Sim expects the link named left_rubber_hand for body_positions, then left_rubber_hand is correct in isaac_body_names.
Foot Ground Adjustment:
I've updated the foot ground adjustment to explicitly use node_indices to find the foot joints, making it more robust.

How to Verify Joint Axes from URDF:

Open your G1 URDF file (the one specified by retarget_cfg["target_tpose"] or a related G1 URDF).
Search for each joint name from your dof_names list (e.g., "left_hip_pitch_joint").
Inside its <joint> tag, look for <axis xyz="X Y Z"/>.
If X is 1, it's X-axis. If Y is 1, it's Y-axis. If Z is 1, it's Z-axis. Update the joint_axis_map accordingly.

This updated script directly extracts the required data from the target_motion object and formats it into the .npz file, avoiding the need for the separate data_convert.py script's Pinocchio re-calculation. This should be a much cleaner and more accurate integration.