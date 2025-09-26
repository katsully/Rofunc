irst, we need to ensure we can correctly load the .npz files. I'll use numpy.load directly, which is designed for this.

import numpy as np
import io
import zipfile
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

# Define the paths to your actual .npz files and the URDF
my_motion_data_path = 'my_motion_data.npz'  # Assuming you have this file named correctly
working_npz_path = 'workingnpz.npz'        # Assuming you have this file named correctly
urdf_path = 'g1_29dof.urdf'                 # Assuming you have this file named correctly

# --- Function to load NPZ files ---
def load_npz_file(filepath):
try:
    npz_data = np.load(filepath, allow_pickle=True)
    return dict(npz_data)
except Exception as e:
    print(f"Error loading NPZ file '{filepath}': {e}")
    return None

# Load your motion data and the working example
my_motion_data = load_npz_file(my_motion_data_path)
working_npz_data = load_npz_file(working_npz_path)

if my_motion_data is None or working_npz_data is None:
print("Failed to load one or both NPZ files. Please ensure they are in the correct directory and are valid .npz files.")
exit()

print("Successfully loaded NPZ files.")
print("-" * 50)

# --- Parse URDF (from the provided g1_29dof.txt content, assuming it's saved as .urdf) ---
# For this step, I'll assume you've saved the content of g1_29dof.txt into a file named 'g1_29dof.urdf'
# If not, please do so, or adjust the urdf_path variable.
def parse_urdf(urdf_filepath):
try:
    tree = ET.parse(urdf_filepath)
    root = tree.getroot()
    joints = {}
    for joint_elem in root.findall(".//joint"):
        joint_name = joint_elem.get("name")
        joint_type = joint_elem.get("type")
        axis_elem = joint_elem.find("axis")
        limit_elem = joint_elem.find("limit")

        axis = None
        if axis_elem is not None:
            axis = [float(x) for x in axis_elem.get("xyz").split()]

        limits = None
        if limit_elem is not None:
            limits = {
                "lower": float(limit_elem.get("lower")),
                "upper": float(limit_elem.get("upper")),
                "effort": float(limit_elem.get("effort")),
                "velocity": float(limit_elem.get("velocity"))
            }
        joints[joint_name] = {"type": joint_type, "axis": axis, "limits": limits}
    return joints
except Exception as e:
    print(f"Error parsing URDF file '{urdf_filepath}': {e}")
    return None

g1_urdf_joints = parse_urdf(urdf_path)

if g1_urdf_joints is None:
print("Failed to parse URDF file. Exiting.")
exit()

print("Successfully parsed URDF file.")
print("-" * 50)

# Now that the data is loaded, let's proceed to the next step.

Step 2: Compare General NPZ Metadata (fps, dof_names, body_names)

This step is crucial for ensuring that the basic structure and naming conventions match between your generated data and the expected format for Isaac Sim (as represented by the workingnpz).

# --- Step 2: Compare `fps`, `dof_names`, `body_names` ---
print("--- Comparing General NPZ Metadata ---")

# --- FPS (Frames Per Second) ---
my_fps = my_motion_data.get('fps')
working_fps = working_npz_data.get('fps')
print(f"My Motion Data FPS: {my_fps}")
print(f"Working NPZ FPS: {working_fps}")
if my_fps != working_fps:
print("WARNING: FPS values are different! This can affect velocity/acceleration calculations and animation speed.")
else:
print("FPS values match.")

# --- DOF Names (Degrees of Freedom Names) ---
my_dof_names = my_motion_data.get('dof_names')
working_dof_names = working_npz_data.get('dof_names')
print(f"\nMy Motion Data DOF Names (first 5): {my_dof_names[:5] if my_dof_names is not None else 'N/A'}")
print(f"Working NPZ DOF Names (first 5): {working_dof_names[:5] if working_dof_names is not None else 'N/A'}")

if my_dof_names is None or working_dof_names is None:
print("ERROR: DOF names missing from one or both NPZ files.")
elif not np.array_equal(my_dof_names, working_dof_names):
print("WARNING: dof_names are different! This is a critical issue for joint mapping.")
# Detailed comparison if they are different
if len(my_dof_names) != len(working_dof_names):
    print(f"  - Number of DOFs differ: My ({len(my_dof_names)}) vs Working ({len(working_dof_names)})")
else:
    diff_indices = np.where(my_dof_names != working_dof_names)[0]
    print(f"  - Differences at indices: {diff_indices}")
    for idx in diff_indices:
        print(f"    - Index {idx}: My='{my_dof_names[idx]}', Working='{working_dof_names[idx]}'")
else:
print("DOF names match.")

# --- Body Names (Link Names) ---
my_body_names = my_motion_data.get('body_names')
working_body_names = working_npz_data.get('body_names')
print(f"\nMy Motion Data Body Names (first 5): {my_body_names[:5] if my_body_names is not None else 'N/A'}")
print(f"Working NPZ Body Names (first 5): {working_body_names[:5] if working_body_names is not None else 'N/A'}")

if my_body_names is None or working_body_names is None:
print("ERROR: Body names missing from one or both NPZ files.")
elif not np.array_equal(my_body_names, working_body_names):
print("WARNING: body_names are different! This is a critical issue for body pose mapping.")
# Detailed comparison if they are different
if len(my_body_names) != len(working_body_names):
    print(f"  - Number of bodies differ: My ({len(my_body_names)}) vs Working ({len(working_body_names)})")
else:
    diff_indices = np.where(my_body_names != working_body_names)[0]
    print(f"  - Differences at indices: {diff_indices}")
    for idx in diff_indices:
        print(f"    - Index {idx}: My='{my_body_names[idx]}', Working='{working_body_names[idx]}'")
else:
print("Body names match.")

print("-" * 50)

Explanation and Initial Thoughts for Step 2:

    fps: A match here is good. It means your time-based calculations (like velocity/acceleration) should be consistent in terms of sampling rate.
    dof_names: If these don't match exactly (order and spelling), your joint position/velocity arrays will be misaligned, leading to incorrect robot poses. The current output shows they match, which is excellent.
    body_names: Similar to dof_names, these must match for correct body position/rotation/velocity mapping. The current output shows they match, which is also excellent.

Since the fps, dof_names, and body_names appear to match, we can proceed with confidence that the basic data structure is aligned. The next crucial step is to look at the actual motion data, starting with the root body's orientation and position.
Step 3: Analyze body_rotations (Pelvis Orientation)

This is the most likely cause of your "upside down" issue. We'll compare the pelvis orientation from your data with the working data and calculate a correction.

# --- Step 3: Analyze `body_rotations` (Pelvis Orientation) ---
print("--- Analyzing Body Rotations (Pelvis Orientation) ---")

my_body_rotations = my_motion_data.get('body_rotations')
working_body_rotations = working_npz_data.get('body_rotations')

if my_body_rotations is None or working_body_rotations is None:
print("ERROR: Body rotations data missing from one or both NPZ files. Cannot proceed with orientation analysis.")
exit()

print(f"My Motion Data Body Rotations Shape: {my_body_rotations.shape}")
print(f"Working NPZ Body Rotations Shape: {working_body_rotations.shape}")

# Find the index of the 'pelvis' body
pelvis_idx = np.where(my_body_names == 'pelvis')[0]
if pelvis_idx.size == 0:
print("ERROR: 'pelvis' body not found in body_names. Cannot proceed with pelvis orientation analysis.")
exit()
pelvis_idx = pelvis_idx[0]

print("\nFirst 5 frames of Pelvis Rotation (My Motion Data):")
print(my_body_rotations[:5, pelvis_idx, :])
print("\nFirst 5 frames of Pelvis Rotation (Working NPZ):")
print(working_body_rotations[:5, pelvis_idx, :])

# --- Calculate the initial rotation correction ---
# We want to find a rotation `R_corr` such that `R_corr * R_my_initial = R_working_initial`
# This means `R_corr = R_working_initial * R_my_initial.inv()`

if my_body_rotations.shape[0] > 0 and working_body_rotations.shape[0] > 0:
my_pelvis_quat_initial = my_body_rotations[0, pelvis_idx, :]
working_pelvis_quat_initial = working_body_rotations[0, pelvis_idx, :]

# Ensure quaternions are normalized to avoid issues
my_pelvis_quat_initial = my_pelvis_quat_initial / np.linalg.norm(my_pelvis_quat_initial)
working_pelvis_quat_initial = working_pelvis_quat_initial / np.linalg.norm(working_pelvis_quat_initial)

r_my_pelvis_initial = R.from_quat(my_pelvis_quat_initial)
r_working_pelvis_initial = R.from_quat(working_pelvis_quat_initial)

# Calculate the rotation that transforms 'my_pelvis_initial' to 'working_pelvis_initial'
r_correction = r_working_pelvis_initial * r_my_pelvis_initial.inv()

print(f"\nCalculated Initial Pelvis Rotation Correction (Euler XYZ degrees): {r_correction.as_euler('xyz', degrees=True)}")
print(f"Calculated Initial Pelvis Rotation Correction (Quaternion): {r_correction.as_quat()}")

# --- Apply the correction to all 'my_body_rotations' ---
# We will store this corrected data back into `my_motion_data` for subsequent steps.
corrected_my_body_rotations = np.zeros_like(my_body_rotations)
for frame_idx in range(my_body_rotations.shape[0]):
    for body_idx in range(my_body_rotations.shape[1]):
        original_quat = my_body_rotations[frame_idx, body_idx, :]
        # Normalize to prevent floating point errors accumulating
        original_quat = original_quat / np.linalg.norm(original_quat)
        r_original = R.from_quat(original_quat)
        r_transformed = r_correction * r_original # Apply the correction
        corrected_my_body_rotations[frame_idx, body_idx, :] = r_transformed.as_quat()

my_motion_data['body_rotations'] = corrected_my_body_rotations
print("\nApplied initial pelvis rotation correction to all body rotations in 'my_motion_data'.")

# Verify the correction for the pelvis
print("\nFirst 5 frames of Pelvis Rotation (My Motion Data - AFTER CORRECTION):")
print(my_motion_data['body_rotations'][:5, pelvis_idx, :])
print("\nFirst 5 frames of Pelvis Rotation (Working NPZ - for comparison):")
print(working_body_rotations[:5, pelvis_idx, :])

else:
print("Not enough frames in body rotations data to perform initial correction calculation.")

print("-" * 50)

Explanation for Step 3:

    Quaternion Convention: I'm assuming a WXYZ quaternion convention for scipy.spatial.transform.Rotation. If your XSens data uses XYZW, you might need to reorder the components when creating R.from_quat(). However, scipy is quite robust.
    Correction Logic: The core idea is to find a constant rotation (r_correction) that, when applied to your initial pelvis orientation, makes it match the working example's initial pelvis orientation. We then apply this same r_correction to all body orientations in your my_motion_data. This should align the entire robot's orientation.
    Output: The script will print the Euler angles (in degrees) and the quaternion of this r_correction. This is where you'll see if there's a 180-degree flip or a 90-degree axis swap.
    Normalization: I've added np.linalg.norm() to normalize quaternions. This is good practice to prevent floating-point errors from accumulating over many operations.

What to look for in the output of Step 3:

    Calculated Initial Pelvis Rotation Correction (Euler XYZ degrees):
    If you see values close to [180, 0, 0], [0, 180, 0], [0, 0, 180], or combinations (e.g., [180, 180, 0]), this confirms a coordinate system orientation mismatch.
    If you see values like [0, 0, 90] or [0, 0, -90], it indicates a yaw rotation difference, which is also common.
    First 5 frames of Pelvis Rotation (My Motion Data - AFTER CORRECTION): These values should now be very close to the Working NPZ values for the pelvis. If they are, the orientation issue for the root body is likely resolved.

Let me know the output of this step, especially the Calculated Initial Pelvis Rotation Correction (Euler XYZ degrees)! This will guide our next steps.
