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


def motion_from_fbx(fbx_file_path, root_joint, fps=60, visualize=True):
	# import fbx file - make sure to provide a valid joint name for root_joint
	motion = SkeletonMotion.from_fbx(
		fbx_file_path=fbx_file_path,
		root_joint=root_joint,
		fps=fps
	)

	# todo: add visualize

	return motion

def motion_retargeting(retarget_cfg, source_motion, visualize=True):
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

	# state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, target_motion.rotation[0],
	#                                                          target_motion.root_translation[0], is_local=True)
	# plot_skeleton_state(state, verbose=True)
	# plot_skeleton_motion_interactive(target_motion)

	# keep frames between [trim_frame_beg, trim_frame_end - 1]
	frame_beg = retarget_cfg["trim_frame_beg"]
	frame_end = retarget_cfg["trim_frame_end"]
	if frame_beg == -1:
		frame_beg = 0

	if frame_end == -1:
		frame_end = target_motion.local_rotation.shape[0]

	local_rotation = target_motion.local_rotation
	root_translation = target_motion.root_translation
	local_rotation = local_rotation[frame_beg:frame_end, ...]
	root_translation = root_translation[frame_beg:frame_end, ...]
	# move human to origin
	avg_root_translation = root_translation.mean(axis=0)
	root_translation[1:] -= avg_root_translation

	new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation,
																	root_translation, is_local=True)
	target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

	# need to convert some joints from 3D to 1D (e.g. elbows and knees)
	# target_motion = _project_joints(target_motion)

	# move the root so that the feet are on the ground
	local_rotation = target_motion.local_rotation
	root_translation = target_motion.root_translation
	tar_global_pos = target_motion.global_translation
	min_h = torch.min(tar_global_pos[..., 2])
	root_translation[:, 2] += -min_h

	# adjust the height of the root to avoid ground penetration
	root_height_offset = retarget_cfg["root_height_offset"]
	root_translation[:, 2] += root_height_offset

	new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation,
																	root_translation, is_local=True)
	target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

	# NPZ export selection
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

	# save retargeted motion
	target_motion.to_file(retarget_cfg["target_motion_path"])

	if visualize:
		# visualize retargeted motion
		rf.logger.beauty_print("Plot G1 skeleton motion", type="module")
		plot_skeleton_motion_interactive(target_motion, verbose=False)

		# state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, target_motion.rotation[0],
		#                                                          target_motion.root_translation[0], is_local=True)
		# plot_skeleton_state(state, verbose=True)

def npy_from_fbx(fbx_file):
	"""
		This scripts shows how to retarget a motion clip from the source skeleton to a target skeleton.
		Data required for retargeting are stored in a retarget config dictionary as a json file. This file contains:
			- source_motion: a SkeletonMotion npy format representation of a motion sequence. The motion clip should use the same skeleton as the source T-Pose skeleton.
			- target_motion_path: path to save the retargeted motion to
			- source_tpose: a SkeletonState npy format representation of the source skeleton in it's T-Pose state
			- target_tpose: a SkeletonState npy format representation of the target skeleton in it's T-Pose state (pose should match source T-Pose)
			- joint_mapping: mapping of joint names from source to target
			- rotation: root rotation offset from source to target skeleton (for transforming across different orientation axes), represented as a quaternion in XYZW order.
			- scale: scale offset from source to target skeleton
		"""
	rofunc_path = rf.oslab.get_rofunc_path()
	config = {
		"target_motion_path": fbx_file.replace('_xsens.fbx', '_xsens2g1.npy'),
		"target_dof_states_path": fbx_file.replace('_xsens.fbx', '_xsens2g1_dof_states.npy'),
		"source_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/source_xsens_wo_gloves_tpose.npy"),
		"target_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/target_g1_29dof_tpose.npy"),
		"joint_mapping": { # Left: Xsens, Right: MJCF
			"Hips": "pelvis",
			"LeftUpLeg": "left_hip_pitch_link",
			"LeftLeg": "left_knee_link",
			"LeftFoot": "left_ankle_pitch_link",
			"RightUpLeg": "right_hip_pitch_link",
			"RightLeg": "right_knee_link",
			"RightFoot": "right_ankle_pitch_link",
			"Spine3": "torso_link",
			# "Neck":
			"LeftArm": "left_shoulder_pitch_link",
			"LeftForeArm": "left_elbow_link",
			"LeftHand": "left_wrist_pitch_link",
			"RightArm": "right_shoulder_pitch_link",
			"RightForeArm": "right_elbow_link",
			"RightHand": "right_wrist_pitch_link"
		},
		"rotation": [0.5, 0.5, 0.5, 0.5],
		"scale": 0.01,
		"root_height_offset": 0.0,
		"trim_frame_beg": 0,
		"trim_frame_end": -1
	}

	source_motion = motion_from_fbx(fbx_file, root_joint="Hips", fps=60, visualize=False)
	motion_retargeting(config, source_motion, visualize=False)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--fbx_file", type=str, default=None)
	args = parser.parse_args()

	fbx_file = args.fbx_file

	rofunc_path = rf.oslab.get_rofunc_path()

	npy_from_fbx(fbx_file)