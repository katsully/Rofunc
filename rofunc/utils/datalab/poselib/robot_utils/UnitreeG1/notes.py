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
- pandas (though not directly used in this modified script, it's often part of the ecosystem)
- scipy (for gaussian_filter1d)
- pinocchio (if you were to use the original data_convert.py's FK, but we're avoiding that here)
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
# Note: This might need more sophisticated ground plane projection for robust results
avg_root_translation_xy = root_translation[:, :2].mean(axis=0)
root_translation[:, :2] -= avg_root_translation_xy # Move X,Y to origin

# Create a temporary SkeletonState to calculate global positions for min_h
temp_sk_state = SkeletonState.from_rotation_and_root_translation(
    target_motion.skeleton_tree, local_rotation, root_translation, is_local=True
)
temp_motion = SkeletonMotion.from_skeleton_state(temp_sk_state, fps=target_motion.fps)

# Adjust height so feet are on the ground
# This requires knowing which joints are feet, and getting their global Z position
# For G1, let's assume 'left_ankle_pitch_link' and 'right_ankle_pitch_link' are good indicators
# You might need to refine this based on your specific G1 model and desired ground contact points

# Find indices for relevant "foot" joints in the target_motion's skeleton_tree
foot_joint_names = ["left_ankle_pitch_link", "right_ankle_pitch_link"]
foot_indices = []
for foot_name in foot_joint_names:
    try:
        foot_indices.append(target_motion.skeleton_tree.node_indices[foot_name])
    except KeyError:
        print(f"Warning: Foot joint '{foot_name}' not found in target skeleton. Skipping ground adjustment.")
        foot_indices = [] # Disable adjustment if any foot joint is missing
        break

if foot_indices:
    # Get global Z positions of foot joints
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

# 2. dof_positions: Joint angles (local_rotation quaternions converted to Euler angles)
# Poselib's local_rotation is (N, D, 4) (w,x,y,z)
# For Pinocchio/Isaac Sim, dof_positions are typically joint angles (e.g., Euler angles for spherical, scalar for revolute)
# This is a critical point. Assuming 'xyz' Euler order for now, but this might need adjustment
# based on your specific G1 URDF and how Pinocchio interprets joint angles.
# A more robust approach would be to get joint angles directly from Pinocchio if you were using it.
# For now, let's convert quaternions to Euler angles (XYZ order, radians)
dof_positions_quat = target_motion.local_rotation.cpu().numpy() # (N, D, 4)
dof_positions_euler = np.zeros((N, dof_names.shape[0], 3), dtype=np.float64) # (N, D, 3)
for i in range(N):
    for j in range(dof_names.shape[0]):
        # R.from_quat expects (x, y, z, w) for as_euler, but poselib uses (w, x, y, z)
        # So, reorder to (x, y, z, w) for scipy.spatial.transform.Rotation
        quat_xyzw = np.array([dof_positions_quat[i, j, 1], dof_positions_quat[i, j, 2],
                              dof_positions_quat[i, j, 3], dof_positions_quat[i, j, 0]])
        dof_positions_euler[i, j, :] = R.from_quat(quat_xyzw).as_euler('xyz', degrees=False)

# Flatten dof_positions_euler to (N, D*3) if Isaac Sim expects a flat array of angles
# Your working NPZ had (499, 29) for dof_positions, implying 1 value per DOF.
# This is a major discrepancy. If your G1 has 29 revolute joints, then (N, 29) is correct.
# If it has spherical joints, (N, 29*3) would be more appropriate.
# Given your working NPZ, let's assume your 29 DOFs are 1D (revolute-like).
# This means you need to extract a single angle per joint from the quaternion.
# This is highly dependent on your G1 URDF joint definitions.
# For now, I'll use a placeholder, but this needs careful mapping.
# Let's assume for now that poselib's local_rotation for these 29 DOFs
# can be represented by a single angle (e.g., from the 'z' axis of Euler conversion).
# This is a simplification and might need refinement.
dof_positions = dof_positions_euler[:, :, 2] # Taking only the Z-axis Euler angle as a placeholder
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
        print(f"Error: Isaac body name '{name}' not found in target skeleton. Check your body_names list.")
        # Handle this error appropriately, e.g., by exiting or skipping.
        return

# 5. body_positions: Global positions of selected bodies
# target_motion.global_translation is (N, D, 3), where D is total DOFs (29)
body_positions = target_motion.global_translation.cpu().numpy()[:, body_indices, :].astype(np.float32)

# 6. body_rotations: Global rotations of selected bodies
# target_motion.global_rotation is (N, D, 4) (w,x,y,z)
body_rotations = target_motion.global_rotation.cpu().numpy()[:, body_indices, :].astype(np.float32)

# 7. body_linear_velocities: Calculate from body_positions
body_linear_velocities = calculate_velocities(body_positions, dt, sigma=1).astype(np.float32)

# 8. body_angular_velocities: Calculate from body_rotations
body_angular_velocities = np.zeros((N, len(isaac_body_names), 3), dtype=np.float32)
for j in range(len(isaac_body_names)):
    quats = body_rotations[:, j, :] # (N, 4)
    angular_vels = np.zeros((N, 3), dtype=np.float32)
    if N > 1:
        angular_vels[0] = compute_angular_velocity(quats[0], quats[1], dt)
        angular_vels[-1] = compute_angular_velocity(quats[-2], quats[-1], dt)
    for k in range(1, N - 1):
        av1 = compute_angular_velocity(quats[k - 1], quats[k], dt)
        av2 = compute_angular_velocity(quats[k], quats[k + 1], dt)
        angular_vels[k] = 0.5 * (av1 + av2)
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
    # To visualize the NPZ output, you'd need to load it back into a SkeletonMotion
    # This is for verification purposes.
    # Note: This visualization will use the full 29 DOFs, not just the 16 bodies.
    # You might need to adjust the SkeletonState/Motion creation for visualization
    # if you only want to see the 16 bodies.
    
    # Create a SkeletonMotion from the NPZ data for visualization
    # This requires reconstructing the SkeletonTree from dof_names and a T-pose.
    # For simplicity, let's just plot the original target_motion again.
    plot_skeleton_motion_interactive(target_motion, verbose=False)

def npy_from_fbx(fbx_file):
rofunc_path = rf.oslab.get_rofunc_path()
config = {
    "target_motion_path": fbx_file.replace('_xsens.fbx', '_xsens2g1.npy'), # This .npy file is still saved
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
