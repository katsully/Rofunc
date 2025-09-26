# (Assuming you've run the previous code block and `my_motion_data` and `working_npz_data` are loaded)

# --- Step 3.1: Re-evaluate and Correct `body_rotations` (Pelvis Orientation) ---
print("\n--- Re-evaluating and Correcting Body Rotations (Pelvis Orientation) ---")

my_body_rotations = my_motion_data.get('body_rotations')
working_body_rotations = working_npz_data.get('body_rotations')
my_body_names = my_motion_data.get('body_names') # Ensure this is available

if my_body_rotations is None or working_body_rotations is None:
print("ERROR: Body rotations data missing from one or both NPZ files. Cannot proceed.")
exit()

pelvis_idx = np.where(my_body_names == 'pelvis')[0]
if pelvis_idx.size == 0:
print("ERROR: 'pelvis' body not found in my_body_names. Cannot proceed.")
exit()
pelvis_idx = pelvis_idx[0]

# Get the initial pelvis quaternions
my_pelvis_quat_initial = my_body_rotations[0, pelvis_idx, :]
working_pelvis_quat_initial = working_body_rotations[0, pelvis_idx, :]

# Normalize quaternions
my_pelvis_quat_initial = my_pelvis_quat_initial / np.linalg.norm(my_pelvis_quat_initial)
working_pelvis_quat_initial = working_pelvis_quat_initial / np.linalg.norm(working_pelvis_quat_initial)

r_my_pelvis_initial = R.from_quat(my_pelvis_quat_initial)
r_working_pelvis_initial = R.from_quat(working_pelvis_quat_initial)

# Calculate the rotation that transforms 'my_pelvis_initial' to 'working_pelvis_initial'
# R_corr * R_my_initial = R_working_initial  => R_corr = R_working_initial * R_my_initial.inv()
r_correction = r_working_pelvis_initial * r_my_pelvis_initial.inv()

print(f"\nCalculated Initial Pelvis Rotation Correction (Euler XYZ degrees): {r_correction.as_euler('xyz', degrees=True)}")
print(f"Calculated Initial Pelvis Rotation Correction (Quaternion): {r_correction.as_quat()}")

# --- Apply this correction to ALL body rotations in your data ---
# This is a global rotation that aligns your entire motion sequence's initial orientation
# with the working NPZ's initial orientation.
corrected_my_body_rotations = np.zeros_like(my_body_rotations)
for frame_idx in range(my_body_rotations.shape[0]):
for body_idx in range(my_body_rotations.shape[1]):
    original_quat = my_body_rotations[frame_idx, body_idx, :]
    # Normalize to prevent floating point errors
    original_quat = original_quat / np.linalg.norm(original_quat)
    r_original = R.from_quat(original_quat)
    
    # Apply the correction. The order matters: r_correction * r_original applies
    # r_correction *before* r_original in the local frame, or *after* in the global frame.
    # Given we want to align the global orientation, this order is usually correct.
    r_transformed = r_correction * r_original 
    corrected_my_body_rotations[frame_idx, body_idx, :] = r_transformed.as_quat()

# Update your data with the corrected rotations
my_motion_data['body_rotations'] = corrected_my_body_rotations
print("\nApplied initial pelvis rotation correction to all body rotations in 'my_motion_data'.")

# Verify the correction for the pelvis
print("\nFirst 5 frames of Pelvis Rotation (My Motion Data - AFTER CORRECTION):")
print(my_motion_data['body_rotations'][:5, pelvis_idx, :])
print("\nFirst 5 frames of Pelvis Rotation (Working NPZ - for comparison):")
print(working_body_rotations[:5, pelvis_idx, :])

print("-" * 50)

Expected Output from Step 3.1:

After running this, the First 5 frames of Pelvis Rotation (My Motion Data - AFTER CORRECTION) should now closely match the Working NPZ's pelvis rotations for the first few frames. The Calculated Initial Pelvis Rotation Correction will tell us the exact rotational misalignment.

Important Note on Body Name Differences:

You mentioned you're not worrying about body name differences for now. However, the difference at indices 7 and 8 (your right_wrist_pitch_link vs. workingnpz's right_rubber_hand, and similarly for left) is extremely important.

    If Isaac Sim's robot model has right_rubber_hand and left_rubber_hand as its end-effector links, and your body_names refers to right_wrist_pitch_link and left_wrist_pitch_link at those indices, then the pose data for the hands will be completely wrong. Isaac Sim will apply the motion intended for your wrist pitch links to the robot's actual hand links, and the wrist pitch links themselves (if they exist in the model) will remain static or follow some default.
    Recommendation: While we can proceed with other corrections, please keep in mind that you will eventually need to align your body_names with the exact names and order expected by your Isaac Sim robot model. This might involve:

    Renaming your XSens segments during processing.
    Adjusting the body_indices to pick the correct links from your XSens data to match the workingnpz's body_names order.

Let's see the output of Step 3.1. This correction is fundamental to getting the robot oriented correctly.