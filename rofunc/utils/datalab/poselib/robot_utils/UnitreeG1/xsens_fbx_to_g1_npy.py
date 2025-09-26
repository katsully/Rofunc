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

from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R

# Add this function to your code
def ensure_quaternion_wxyz_format(quaternions):
	"""
	Ensures quaternions are in [w,x,y,z] format by checking the magnitude of components.
	If they appear to be in [x,y,z,w] format, swaps the components.
	"""
	# Check a sample of quaternions to determine format
	w_magnitude = np.mean(np.abs(quaternions[0:10, 0, 0]))
	last_magnitude = np.mean(np.abs(quaternions[0:10, 0, 3]))

	if last_magnitude > w_magnitude:
		print("Converting quaternions from [x,y,z,w] to [w,x,y,z] format")
		result = quaternions.copy()
		result[..., 0] = quaternions[..., 3]  # w
		result[..., 1:] = quaternions[..., :3]  # x,y,z
		return result
	else:
		print("Quaternions already in [w,x,y,z] format")
		return quaternions

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

# parses a urdf file and extracts the axis for each revolute joint
# returns a tuple - A] dictionary where keys are joint names and values are 
# numpy arrays representing the joint axis
# B] a list of revolute joint names 
def get_joint_axes_from_urdf(urdf_file_path):
	robot = URDF.from_xml_file(urdf_file_path)
	joint_axes = {}
	revolute_joint_names_ordered = []
	for joint in robot.joints:
		if joint.type == 'revolute':
			joint_axes[joint.name] = np.array(joint.axis)
			revolute_joint_names_ordered.append(joint.name)
	return joint_axes, revolute_joint_names_ordered

# function to convert quaternion to scalar angle
# converts a quaternion representing a revolute joint's rotation to a scalar angle
# around its primary axis
def quaternion_to_scalar_angle(quaternion_w_xyz, joint_axis):
	# ensure the joint_axis is normalized
	joint_axis = joint_axis / np.linalg.norm(joint_axis)

	# poselib's local rotation is w,x,y,z
	# scipy's rotation.from_quat expects x,y,z,w
	quat_xyzw = np.array([quaternion_w_xyz[1], quaternion_w_xyz[2], quaternion_w_xyz[3], quaternion_w_xyz[0]])
	rotation = R.from_quat(quat_xyzw)

	# convert the rotation to axis-angle representation
	rot_vec = rotation.as_rotvec()

	scalar_angle = np.dot(rot_vec, joint_axis)

	return scalar_angle

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


	# After loading T-poses
	print("Source T-pose root rotation:", source_tpose.local_rotation[0])
	print("Target T-pose root rotation:", target_tpose.local_rotation[0])
	# parse data from retarget config
	rotation_to_target_skeleton = torch.tensor(retarget_cfg["rotation"])
	# After loading T-poses
	print("after rotation_to_target_skeleton")
	print("Source T-pose root rotation:", source_tpose.local_rotation[0])
	print("Target T-pose root rotation:", target_tpose.local_rotation[0])

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

	total_frames = frame_end - frame_beg

	local_rotation = target_motion.local_rotation
	root_translation = target_motion.root_translation
	local_rotation = local_rotation[frame_beg:frame_end, ...]
	root_translation = root_translation[frame_beg:frame_end, ...]
	# move human to origin
	avg_root_translation = root_translation.mean(axis=0)
	root_translation[:] -= avg_root_translation

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

	print("after retargeting")
	print("First frame root position:", target_motion.root_translation[0])
	print("First frame root rotation:", target_motion.local_rotation[0,0])

	# NPZ export selection
	N = target_motion.local_rotation.shape[0]
	fps = target_motion.fps
	dt = 1.0 / fps

	# 1. dof_names
	path_name = os.path.join(rofunc_path, "simulator/assets/urdf/unitreeG1/urdf/g1_29dof.urdf")
	g1_joint_axes, g1_revolute_joint_names = get_joint_axes_from_urdf(path_name)

	dof_names = np.array(g1_revolute_joint_names)

	print("G1 Revolute Joints (ordered from URDF):", dof_names)
	print("Number of revolute joints:", len(dof_names))

	# 2. dof_positions: Joint angles (scalar values for each of the 29 revolute joints)
	# target_motion.local_rotation is (N, D, 4) in (w, x, y, z) order
	dof_positions_quat = target_motion.local_rotation.cpu().numpy() # (N, 29, 4)
	
	# initialize dof_positions with the correct shape
	dof_positions = np.zeros((total_frames, dof_names.shape[0]), dtype=np.float64)

	# loop through each frame and each DOF to extract scalar angles
	for i in range(total_frames):
		for j in range(dof_names.shape[0]):
			joint_name = dof_names[j]
			quaternion_w_xyz = dof_positions_quat[i,j,:] # w,x,y,z from PoseLib

			# get the specific joint axis from the URDF data
			if joint_name in g1_joint_axes:
				urdf_joint_axis = g1_joint_axes[joint_name]
				dof_positions[i,j] = quaternion_to_scalar_angle(quaternion_w_xyz, urdf_joint_axis)
			else:
				print(f"Warning: Joint '{joint_name}' not found in URDF ")
				dof_positions[i,j] = 0.0 # this might just need to throw an error instead

	# Ensure it's float64 as per working NPZ
	dof_positions = dof_positions.astype(np.float64) 

	dof_velocities = calculate_velocities(dof_positions, dt=dt, sigma=1)


	isaac_body_names = [
		"pelvis", 
		"left_shoulder_pitch_link", "right_shoulder_pitch_link",
		"left_elbow_link", "right_elbow_link",
		"right_hip_yaw_link", "left_hip_yaw_link",
		"right_wrist_pitch_link", "left_wrist_pitch_link", 
		"right_ankle_roll_link", "left_ankle_roll_link",
		"left_shoulder_yaw_link", "right_shoulder_yaw_link",
		"torso_link",
		"right_knee_link", "left_knee_link"
	]
	isaac_body_names_array = np.array(isaac_body_names, dtype=np.str_)

	# Get the indices of these body names in the target_motion's skeleton_tree
	body_indices = []
	for name in isaac_body_names:
		try: 
			body_indices.append(target_motion.skeleton_tree._node_indices[name])
		except KeyError:
			print(f"Error: Isaac body name '{name}' not found in target skeleton. Please check your isaac_body_names list against the G1 URDF")
			return

	# Get global positions and apply coordinate transformation
	global_positions = target_motion.global_translation.cpu().numpy()
	global_rotations = target_motion.global_rotation.cpu().numpy()

	# Ensure quaternions are in [w,x,y,z] format
	global_rotations = ensure_quaternion_wxyz_format(global_rotations)

	# Initialize arrays
	body_positions = np.zeros((N, len(body_indices), 3), dtype=np.float32)
	body_rotations = np.zeros((N, len(body_indices), 4), dtype=np.float32)

	# Instead of trying to combine rotations, let's set them directly
	for i in range(N):
		for j, idx in enumerate(body_indices):
			# Position transformation: Keep X, swap Y and Z
			x, y, z = global_positions[i, idx]
			body_positions[i, j] = np.array([x, z, y])
			
			# For the root (j=0), set a specific rotation that we know works
			if j == 0:
				# This is a rotation that should make the avatar stand upright and face forward
				body_rotations[i, j] = np.array([1, 0, 0, 0])  # Identity quaternion
			else:
				# For other joints, use the original rotation
				body_rotations[i, j] = global_rotations[i, idx]


	# Set the root height to match the working NPZ (~0.8m)
	body_positions[:, 0, 1] = 0.0  # Reset Y (left/right) to center
	body_positions[:, 0, 2] = 0.8  # Set Z (up) to 0.8m

	# Before saving NPZ
	print("NPZ first frame root position:", body_positions[0,0])
	print("NPZ first frame root rotation:", body_rotations[0,0])

	# After computing body_rotations, check and fix if needed
	w_component_magnitude = np.mean(np.abs(body_rotations[:, 0, 0]))
	last_component_magnitude = np.mean(np.abs(body_rotations[:, 0, 3]))
	if last_component_magnitude > w_component_magnitude:
		print("Fixing quaternion format from [x,y,z,w] to [w,x,y,z]")
		for i in range(N):
			for j in range(len(body_indices)):
				# Swap w and last component
				temp = body_rotations[i, j, 0].copy()
				body_rotations[i, j, 0] = body_rotations[i, j, 3]
				body_rotations[i, j, 3] = temp

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


	dof_names = np.array(g1_revolute_joint_names, dtype='U256')

	# 9. Package and save to NPZ
	data_dict = {
		"fps": fps,
		"dof_names": dof_names,
		"body_names": np.array(isaac_body_names_array, dtype='U256'),
		"dof_positions": dof_positions,
		"dof_velocities": dof_velocities,
		"body_positions": body_positions,
		"body_rotations": body_rotations,
		"body_linear_velocities": body_linear_velocities,
		"body_angular_velocities": body_angular_velocities
	}


	np.savez('my_motion_data.npz', **data_dict)
	# Create a test NPZ with just the root pose matching the working file
	test_data = data_dict.copy()

	# Set the root position and orientation to match the working file
	test_data["body_positions"][0, 0] = np.array([-0.0008, 0, 0.8])  # Similar to working file
	test_data["body_rotations"][0, 0] = np.array([0.9995, 0.0092, -0.0132, 0.0249])  # Similar to working file

	np.savez('test_root_only.npz', **test_data)
	# Immediately verify what was saved
	test_load = np.load('my_motion_data.npz', allow_pickle=True)
	print("\nVerifying saved NPZ file:")
	print("Keys in file:", list(test_load.keys()))
	for key in test_load.keys():
		print(f"{key}: type={type(test_load[key])}, shape={test_load[key].shape if hasattr(test_load[key], 'shape') else 'N/A'}")
	test_load.close()


	# save retargeted motion
	target_motion.to_file(retarget_cfg["target_motion_path"])

	if visualize:
		# visualize retargeted motion
		rf.logger.beauty_print("Plot G1 skeleton motion", type="module")
		plot_skeleton_motion_interactive(target_motion, verbose=False)


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
		"rotation": [0.7071, 0.7071, 0.0, 0.0],  # 90 degrees around X in [w,x,y,z] format		
		"scale": 0.01,
		"root_height_offset": 0.8,
		"trim_frame_beg": 0,
		"trim_frame_end": 300
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