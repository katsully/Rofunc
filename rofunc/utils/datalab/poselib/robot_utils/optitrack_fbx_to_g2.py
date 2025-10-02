#  Copyright (C) 2024, Junjia Liu
#
#  This file is part of Rofunc.
#
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
#
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
#
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

"""
Attention: Since the Autodesk FBX SDK just supports Python 3.7, this script should be run with Python 3.7.
"""

# import isaacgym
# import multiprocessing
import os
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import torch

def quaternion_to_euler(q, order='xyz'):
    """Convert quaternion (w,x,y,z) to Euler angles"""
    # Create Rotation object from quaternion (scipy expects x,y,z,w)
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_euler(order)

def compute_angular_velocity(q_prev, q_next, dt, eps=1e-8):
    """Compute angular velocity from adjacent quaternions"""
    # Convert to scipy format (x,y,z,w)
    q_prev_scipy = np.array([q_prev[1], q_prev[2], q_prev[3], q_prev[0]])
    q_next_scipy = np.array([q_next[1], q_next[2], q_next[3], q_next[0]])

    # Create rotation objects
    r_prev = R.from_quat(q_prev_scipy)
    r_next = R.from_quat(q_next_scipy)

    # Compute relative rotation
    r_rel = r_prev.inv() * r_next

    # Get rotation vector (axis * angle)
    rotvec = r_rel.as_rotvec()

    # Angular velocity is rotation vector divided by time
    return rotvec / dt

def retargeted_motion_to_npz(target_motion, retarget_cfg, urdf_path, mesh_dir):
    """Convert retargeted motion to NPZ format for IsaacSim"""

    # Extract motion data
    fps = int(target_motion.fps)
    dt = 1.0 / fps

    # Get dimensions
    num_frames = target_motion.local_rotation.shape[0]
    num_joints = target_motion.num_joints

    print(f"Processing {num_frames} frames at {fps} fps")
    print(f"Number of joints in skeleton: {num_joints}")

    # Define URDF joint names (from your second script)
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    ]

    # Get root motion (pelvis)
    root_translation = target_motion.root_translation.numpy()  # (N, 3)
    root_rotation = target_motion.global_rotation[:, 0].numpy()  # (N, 4) - quaternion of root

    # Get joint rotations - we need to convert from quaternions to joint angles
    # This is the tricky part - we need to extract the actual joint angles
    # For now, let's use the local rotations and convert them to Euler angles
    local_rotations = target_motion.local_rotation.numpy()  # (N, num_joints, 4)

    # Initialize joint positions array
    num_dof = len(joint_names)
    dof_positions = np.zeros((num_frames, num_dof), dtype=np.float32)

    # Map skeleton joints to URDF joints
    # This mapping depends on your skeleton structure - you'll need to adjust
    # For now, I'll create a basic mapping based on common joint order
    joint_mapping = {
        # Legs
        "left_hip_pitch_joint": 7,  # Adjust these indices based on your skeleton
        "left_hip_roll_joint": 8,
        "left_hip_yaw_joint": 9,
        "left_knee_joint": 10,
        "left_ankle_pitch_joint": 11,
        "left_ankle_roll_joint": 12,
        "right_hip_pitch_joint": 1,
        "right_hip_roll_joint": 2,
        "right_hip_yaw_joint": 3,
        "right_knee_joint": 4,
        "right_ankle_pitch_joint": 5,
        "right_ankle_roll_joint": 6,
        # Torso
        "waist_yaw_joint": 13,
        "waist_roll_joint": 14,
        "waist_pitch_joint": 15,
        # Arms
        "left_shoulder_pitch_joint": 19,
        "left_shoulder_roll_joint": 20,
        "left_shoulder_yaw_joint": 21,
        "left_elbow_joint": 22,
        "left_wrist_roll_joint": 23,
        "left_wrist_pitch_joint": 24,
        "left_wrist_yaw_joint": 25,
        "right_shoulder_pitch_joint": 16,
        "right_shoulder_roll_joint": 17,
        "right_shoulder_yaw_joint": 18,
        "right_elbow_joint": 19,
        "right_wrist_roll_joint": 26,
        "right_wrist_pitch_joint": 27,
        "right_wrist_yaw_joint": 28,
    }

    # Convert quaternions to joint angles
    # This is simplified - in reality you'd need proper FK/IK
    for i, joint_name in enumerate(joint_names):
        if joint_name in joint_mapping:
            joint_idx = joint_mapping[joint_name]
            if joint_idx < num_joints:
                # Extract rotation for this joint
                joint_quats = local_rotations[:, joint_idx, :]
                
                # Convert to Euler angles (simplified - assumes single-axis joints)
                for frame in range(num_frames):
                    euler = quaternion_to_euler(joint_quats[frame])
                    # For single-DOF joints, extract the appropriate angle
                    if 'pitch' in joint_name or 'elbow' in joint_name or 'knee' in joint_name:
                        dof_positions[frame, i] = euler[1]  # Y-axis
                    elif 'roll' in joint_name:
                        dof_positions[frame, i] = euler[0]  # X-axis
                    elif 'yaw' in joint_name:
                        dof_positions[frame, i] = euler[2]  # Z-axis

    # Calculate joint velocities
    dof_velocities = np.zeros_like(dof_positions)
    if num_frames > 1:
        dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
        dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
        dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt
    dof_velocities = gaussian_filter1d(dof_velocities, sigma=1, axis=0)

    # Body names for forward kinematics
    body_names = [
        "pelvis",
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link", 
        "left_elbow_link",
        "right_elbow_link",
        "right_hip_yaw_link",
        "left_hip_yaw_link",
        "right_rubber_hand",
        "left_rubber_hand",
        "right_ankle_roll_link",
        "left_ankle_roll_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "torso_link",
        "right_knee_link",
        "left_knee_link"
    ]

    # Build Pinocchio robot
    robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir, pin.JointModelFreeFlyer())
    model = robot.model
    data = robot.data

    # Initialize output arrays
    B = len(body_names)
    body_positions = np.zeros((num_frames, B, 3), dtype=np.float32)
    body_rotations = np.zeros((num_frames, B, 4), dtype=np.float32)

    # Perform forward kinematics for each frame
    q = pin.neutral(model)

    for frame_idx in range(num_frames):
        # Set floating base pose
        q[0:3] = root_translation[frame_idx]
        # Convert quaternion from (w,x,y,z) to (x,y,z,w) for Pinocchio
        root_q = root_rotation[frame_idx]
        q[3:7] = np.array([root_q[1], root_q[2], root_q[3], root_q[0]])
        
        # Set joint positions
        q[7:7+num_dof] = dof_positions[frame_idx]
        
        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Extract body poses
        for j, body_name in enumerate(body_names):
            try:
                frame_id = model.getFrameId(body_name)
                tf = data.oMf[frame_id]
                
                # Position
                body_positions[frame_idx, j] = tf.translation
                
                # Rotation (convert back to w,x,y,z)
                quat = pin.Quaternion(tf.rotation)
                body_rotations[frame_idx, j] = np.array([quat.w, quat.x, quat.y, quat.z])
            except:
                print(f"Warning: Could not find frame {body_name}")

    # Calculate body velocities
    body_linear_velocities = np.zeros_like(body_positions)
    if num_frames > 1:
        body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
        body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
        body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt
    body_linear_velocities = gaussian_filter1d(body_linear_velocities, sigma=1, axis=0)

    # Angular velocities
    body_angular_velocities = np.zeros((num_frames, B, 3), dtype=np.float32)
    for j in range(B):
        for k in range(1, num_frames - 1):
            body_angular_velocities[k, j] = compute_angular_velocity(
                body_rotations[k-1, j], body_rotations[k+1, j], 2*dt
            )
        if num_frames > 1:
            body_angular_velocities[0, j] = compute_angular_velocity(
                body_rotations[0, j], body_rotations[1, j], dt
            )
            body_angular_velocities[-1, j] = compute_angular_velocity(
                body_rotations[-2, j], body_rotations[-1, j], dt
            )
    body_angular_velocities = gaussian_filter1d(body_angular_velocities, sigma=1, axis=0)

    # Create output dictionary
    data_dict = {
        "fps": fps,
        "dof_names": np.array(joint_names, dtype=np.str_),
        "body_names": np.array(body_names, dtype=np.str_),
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
        "body_positions": body_positions,
        "body_rotations": body_rotations,
        "body_linear_velocities": body_linear_velocities,
        "body_angular_velocities": body_angular_velocities
    }

    return data_dict


def _run_sim(motion):
    from isaacgym import gymapi

    body_links = {"torso_link": gymapi.AXIS_ROTATION,
                  "right_elbow_link": gymapi.AXIS_ROTATION, "left_elbow_link": gymapi.AXIS_ROTATION,
                  "right_wrist_pitch_link": gymapi.AXIS_ALL, "left_wrist_pitch_link": gymapi.AXIS_ALL,
                  "pelvis": gymapi.AXIS_ROTATION,
                  "left_hip_pitch_link": gymapi.AXIS_ROTATION, "right_hip_pitch_link": gymapi.AXIS_ROTATION,
                  "left_shoulder_yaw_link": gymapi.AXIS_ROTATION, "right_shoulder_yaw_link": gymapi.AXIS_ROTATION,
                  "left_knee_link": gymapi.AXIS_ALL, "right_knee_link": gymapi.AXIS_ALL,
                  "left_ankle_pitch_link": gymapi.AXIS_ALL, "right_ankle_pitch_link": gymapi.AXIS_ALL}
    body_ids = [motion.skeleton_tree._node_indices[link] for link in body_links.keys()]
    # hand_links = ["right_qbhand_root_link", "left_qbhand_root_link",
    #               "left_qbhand_thumb_knuckle_link", "left_qbhand_thumb_proximal_link",
    #               "left_qbhand_thumb_distal_link", "left_qbhand_index_proximal_link",
    #               "left_qbhand_index_middle_link", "left_qbhand_index_distal_link",
    #               "left_qbhand_middle_proximal_link", "left_qbhand_middle_middle_link",
    #               "left_qbhand_middle_distal_link", "left_qbhand_ring_proximal_link",
    #               "left_qbhand_ring_middle_link", "left_qbhand_ring_distal_link",
    #               "left_qbhand_little_proximal_link", "left_qbhand_little_middle_link",
    #               "left_qbhand_little_distal_link",
    #               "right_qbhand_thumb_knuckle_link", "right_qbhand_thumb_proximal_link",
    #               "right_qbhand_thumb_distal_link", "right_qbhand_index_proximal_link",
    #               "right_qbhand_index_middle_link", "right_qbhand_index_distal_link",
    #               "right_qbhand_middle_proximal_link", "right_qbhand_middle_middle_link",
    #               "right_qbhand_middle_distal_link", "right_qbhand_ring_proximal_link",
    #               "right_qbhand_ring_middle_link", "right_qbhand_ring_distal_link",
    #               "right_qbhand_little_proximal_link", "right_qbhand_little_middle_link",
    #               "right_qbhand_little_distal_link"]
    # hand_ids = [motion.skeleton_tree._node_indices[link] for link in hand_links]

    # all_links = body_links + hand_links
    # all_ids = body_ids + hand_ids
    all_links = list(body_links.keys())
    all_ids = body_ids
    all_types = list(body_links.values())

    motion_rb_states_pos = motion.global_translation
    motion_rb_states_rot = motion.global_rotation

    # motion_rb_states_rot[:, hand_ids] = quat_mul(
    #     torch.tensor([0, 0.707, 0, 0.707]),
    #     motion_rb_states_rot[:, hand_ids]
    # )

    motion_rb_states_pos[:, :, 2] += 0.06
    motion_rb_states = torch.cat([motion_rb_states_pos, motion_rb_states_rot], dim=-1)

    motion_root_pos = motion_rb_states_pos[:, 0]
    motion_root_rot = motion_rb_states_rot[:, 0]
    motion_root_vel = motion.global_root_velocity
    motion_root_ang_vel = motion.global_root_angular_velocity
    motion_root_states = torch.cat([motion_root_pos, motion_root_rot, motion_root_vel, motion_root_ang_vel], dim=-1)

    args = rf.config.get_sim_config("UnitreeG1")
    UnitreeG1sim = rf.sim.RobotSim(args)
    dof_states = UnitreeG1sim.run_traj_multi_rigid_bodies(
        traj=[motion_rb_states[:, id] for id in all_ids],
        attr_rbs=all_links,
        update_freq=0.001,
        root_state=motion_root_states,
        attr_types=all_types,
        verbose=False
    )
    print(f"dof_states shape: {dof_states.shape}")
    # Expected: (num_frames, num_envs, 58)
    return dof_states


def motion_from_fbx(fbx_file_path, root_joint, fps=60, visualize=True):
    # import fbx file - make sure to provide a valid joint name for root_joint
    motion = SkeletonMotion.from_fbx(
        fbx_file_path=fbx_file_path,
        root_joint=root_joint,
        fps=fps
    )
    # visualize motion
    if visualize:
        rf.logger.beauty_print("Plot Optitrack skeleton motion", type="module")
        plot_skeleton_motion_interactive(motion)
    return motion


def motion_retargeting(retarget_cfg, source_motion, visualize=False):
    # load and visualize t-pose files
    source_tpose = SkeletonState.from_file(retarget_cfg["source_tpose"])
    if visualize:
        rf.logger.beauty_print("Plot Optitrack T-pose", type="module")
        plot_skeleton_state(source_tpose)

    target_tpose = SkeletonState.from_file(retarget_cfg["target_tpose"])
    if visualize:
        rf.logger.beauty_print("Plot H1 T-pose", type="module")
        plot_skeleton_state(target_tpose, verbose=True)

    # parse data from retarget config
    rotation_to_target_skeleton = torch.tensor(retarget_cfg["rotation"])

    # run retargeting
    # target_motion = source_motion.retarget_to_by_tpose(
    target_motion = source_motion.retarget_to_hotu_qbhand_by_tpose(
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

    # move the human to the origin
    # avg_root_translation = root_translation.mean(axis=0)
    # root_translation[1:] -= avg_root_translation

    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation,
                                                                    root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # need to convert some joints from 3D to 1D (e.g. elbows and knees)
    # target_motion = _project_joints(target_motion)

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation

    # Set the human foot on the ground
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h

    # adjust the height of the root to avoid ground penetration
    root_height_offset = retarget_cfg["root_height_offset"]
    root_translation[:, 2] += root_height_offset

    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation,
                                                                    root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # 1. Check skeleton tree structure
    print("=== Target Motion Skeleton Info ===")
    print("Number of joints:", target_motion.num_joints)
    print("Skeleton tree:", target_motion.skeleton_tree)

    # Try to access joint names if available
    if hasattr(target_motion.skeleton_tree, 'node_names'):
        print("\nJoint names:")
    for i, name in enumerate(target_motion.skeleton_tree.node_names):
        print(f"Index {i}: {name}")

    # 2. Check the shape of the motion data
    print("\n=== Motion Data Shapes ===")
    print("Local rotation shape:", target_motion.local_rotation.shape)
    print("Global translation shape:", target_motion.global_translation.shape)
    print("Root translation shape:", target_motion.root_translation.shape)
    print("FPS:", target_motion.fps)

    # 3. Print the joint mapping from the retargeting config
    print("\n=== Joint Mapping ===")
    for opti_name, g1_name in config["joint_mapping"].items():
        print(f"{opti_name} -> {g1_name}")


    # save retargeted motion
    target_motion.to_file(retarget_cfg["target_motion_path"])
    urdf_path = os.path.join(rofunc_path, "simulator/assets/urdf/unitreeG1/g1_29dof.urdf")
    mesh_dir = os.path.join(rofunc_path, "simulator/assets/urdf/unitreeG1/meshes/")
    data_dict = retargeted_motion_to_npz(target_motion, retarget_cfg, urdf_path, mesh_dir)

    # save as NPZ file
    npz_path = retarget_cfg["target_motion_path"].replace('.npy', '.npz')
    np.savez(npz_path, **data_dict)
    rf.logger.beauty_print(f"Saved G1 motion data to {npz_path}", type="module")

   
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

    rf.logger.beauty_print(f"Processing {fbx_file}", type="module")

    rofunc_path = rf.oslab.get_rofunc_path()
    config = {
        "target_motion_path": fbx_file.replace('_optitrack.fbx', '_optitrack2g1.npy'),
        "target_dof_states_path": fbx_file.replace('_optitrack.fbx', '_optitrack2g1_dof29.npy'),
        "source_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/source_optitrack_w_gloves_tpose.npy"),
        "target_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/target_g1_29dof_tpose.npy"),
        # "target_tpose": os.path.join(rofunc_path, args.target_tpose),
        "joint_mapping": {  # Left: Optitrack, Right: MJCF
            # hotu_humanoid.xml
            "Skeleton_Hips": "pelvis",
            # "Skeleton_LeftUpLeg": "left_hip_yaw_link",
            "Skeleton_LeftUpLeg": "left_hip_pitch_link",
            "Skeleton_LeftLeg": "left_knee_link",
            "Skeleton_LeftFoot": "left_ankle_link",
            # "Skeleton_RightUpLeg": "right_hip_yaw_link",
            "Skeleton_RightUpLeg": "right_hip_pitch_link",
            "Skeleton_RightLeg": "right_knee_link",
            "Skeleton_RightFoot": "right_ankle_link",
            "Skeleton_Spine1": "torso_link",
            # "Skeleton_Neck": "head",
            "Skeleton_LeftArm": "left_shoulder_pitch_link",
            # "Skeleton_LeftArm": "left_shoulder_yaw_link",
            "Skeleton_LeftForeArm": "left_elbow_link",
            "Skeleton_LeftHand": "left_wrist_pitch_link",
            "Skeleton_RightArm": "right_shoulder_pitch_link",
            # "Skeleton_RightArm": "right_shoulder_yaw_link",
            "Skeleton_RightForeArm": "right_elbow_link",
            "Skeleton_RightHand": "right_wrist_pitch_link",
            # extra mapping for hotu_humanoid_w_qbhand.xml
            # "Skeleton_LeftHandThumb1": "left_qbhand_thumb_knuckle_link",
            # "Skeleton_LeftHandThumb2": "left_qbhand_thumb_proximal_link",
            # "Skeleton_LeftHandThumb3": "left_qbhand_thumb_distal_link",
            # "Skeleton_LeftHandIndex1": "left_qbhand_index_proximal_link",
            # "Skeleton_LeftHandIndex2": "left_qbhand_index_middle_link",
            # "Skeleton_LeftHandIndex3": "left_qbhand_index_distal_link",
            # "Skeleton_LeftHandMiddle1": "left_qbhand_middle_proximal_link",
            # "Skeleton_LeftHandMiddle2": "left_qbhand_middle_middle_link",
            # "Skeleton_LeftHandMiddle3": "left_qbhand_middle_distal_link",
            # "Skeleton_LeftHandRing1": "left_qbhand_ring_proximal_link",
            # "Skeleton_LeftHandRing2": "left_qbhand_ring_middle_link",
            # "Skeleton_LeftHandRing3": "left_qbhand_ring_distal_link",
            # "Skeleton_LeftHandPinky1": "left_qbhand_little_proximal_link",
            # "Skeleton_LeftHandPinky2": "left_qbhand_little_middle_link",
            # "Skeleton_LeftHandPinky3": "left_qbhand_little_distal_link",
            # "Skeleton_RightHandThumb1": "right_qbhand_thumb_knuckle_link",
            # "Skeleton_RightHandThumb2": "right_qbhand_thumb_proximal_link",
            # "Skeleton_RightHandThumb3": "right_qbhand_thumb_distal_link",
            # "Skeleton_RightHandIndex1": "right_qbhand_index_proximal_link",
            # "Skeleton_RightHandIndex2": "right_qbhand_index_middle_link",
            # "Skeleton_RightHandIndex3": "right_qbhand_index_distal_link",
            # "Skeleton_RightHandMiddle1": "right_qbhand_middle_proximal_link",
            # "Skeleton_RightHandMiddle2": "right_qbhand_middle_middle_link",
            # "Skeleton_RightHandMiddle3": "right_qbhand_middle_distal_link",
            # "Skeleton_RightHandRing1": "right_qbhand_ring_proximal_link",
            # "Skeleton_RightHandRing2": "right_qbhand_ring_middle_link",
            # "Skeleton_RightHandRing3": "right_qbhand_ring_distal_link",
            # "Skeleton_RightHandPinky1": "right_qbhand_little_proximal_link",
            # "Skeleton_RightHandPinky2": "right_qbhand_little_middle_link",
            # "Skeleton_RightHandPinky3": "right_qbhand_little_distal_link",
        },
        # "rotation": [0.707, 0, 0, 0.707], xyzw
        "rotation": [0.5, 0.5, 0.5, 0.5],
        "scale": 0.001,  # Export millimeter to meter
        "root_height_offset": 0.0,
        "trim_frame_beg": 0,
        "trim_frame_end": -1
    }

    source_motion = motion_from_fbx(fbx_file, root_joint="Skeleton_Hips", fps=120, visualize=False)
    # config["target_motion_path"] = fbx_file.replace('.fbx', '_amp.npy')
    motion_retargeting(config, source_motion, visualize=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--fbx_dir", type=str, default=f"{rf.oslab.get_rofunc_path()}/../examples/data/hotu2/20240509")
    # parser.add_argument("--fbx_dir", type=str, default=None)
    # parser.add_argument("--fbx_file", type=str,
    #                     default=f"{rf.oslab.get_rofunc_path()}/../examples/data/hotu2/test_data_04_optitrack.fbx")
    parser.add_argument("--fbx_file", type=str)
                        # default=f"{rf.oslab.get_rofunc_path()}/../examples/data/hotu2/20240509/Waving hand_Take 2024-05-09 04.20.29 PM_optitrack.fbx")
    # parser.add_argument("--parallel", action="store_true")
    # Available asset:
    #                   1. mjcf/amp_humanoid_spoon_pan_fixed.xml
    #                   2. mjcf/amp_humanoid_sword_shield.xml
    #                   3. mjcf/hotu/hotu_humanoid.xml
    #                   4. mjcf/hotu_humanoid_w_qbhand_no_virtual.xml
    #                   5. mjcf/hotu/hotu_humanoid_w_qbhand_full.xml
    parser.add_argument("--humanoid_asset", type=str, default="mjcf/unitreeG1/g1_29dof.xml")
    parser.add_argument("--target_tpose", type=str,
                        default="utils/datalab/poselib/data/target_g1_29dof_tpose.npy")
    args = parser.parse_args()

    rofunc_path = rf.oslab.get_rofunc_path()
    fbx_file = args.fbx_file

    # if args.fbx_dir is not None:
    #     fbx_dir = args.fbx_dir
    #     fbx_files = rf.oslab.list_absl_path(fbx_dir, suffix='.fbx')
    # elif args.fbx_file is not None:
    #     fbx_files = [args.fbx_file]
    # else:
    #     raise ValueError("Please provide a valid fbx_dir or fbx_file.")
    # fbx_dir = os.path.join(rofunc_path, "../examples/data/hotu")
    # fbx_dir = "/home/ubuntu/Data/2023_11_15_HED/has_gloves"
    # fbx_files = rf.oslab.list_absl_path(fbx_dir, suffix='.fbx')
    # fbx_files = ["/home/ubuntu/Data/2023_11_15_HED/has_gloves/New Session-009.fbx"]
    # fbx_files = [os.path.join(rofunc_path, "../examples/data/hotu/test_data_01_xsens.fbx")]

    # from tqdm import tqdm

    npy_from_fbx(fbx_file)

    # if args.parallel:
    #     pool = multiprocessing.Pool()
    #     pool.map(npy_from_fbx, fbx_files)
    # else:
    #     with tqdm(total=len(fbx_files)) as pbar:
    #         for fbx_file in fbx_files:
    #             if os.path.exists(fbx_file.replace('_optitrack.fbx', '_optitrack2h1_dof_states.npy')):
    #                 continue
    #             npy_from_fbx(fbx_file)
    #             pbar.update(1)
