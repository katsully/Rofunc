import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

my_motion_data_path = 'my_motion_data4.npz'
working_npz_path = "C:\\Users\\Kat.Sullivan\\DancingRobot\\converter\\converter\\generated_npz\\dance1_subject1.npz"
urdf_path = "C:\\Users\\Kat.Sullivan\\DancingRobot\\Rofunc\\rofunc\\simulator\\assets\\urdf\\unitreeG1\\urdf\\g1_29dof.urdf"

# # --- Function to load NPZ files ---
# def load_npz_file(filepath):
# 	try:
# 	    npz_data = np.load(filepath, allow_pickle=True)
# 	    return dict(npz_data)
# 	except Exception as e:
# 	    print(f"Error loading NPZ file '{filepath}': {e}")
# 	    return None

# # --- Function to load NPZ files ---
# def load_npz_file(filepath):
# 	try:
# 	    npz_data = np.load(filepath, allow_pickle=True)
# 	    return dict(npz_data)
# 	except Exception as e:
# 	    print(f"Error loading NPZ file '{filepath}': {e}")
# 	    return None

# Load your motion data and the working example
# my_motion_data = load_npz_file(my_motion_data_path)
# working_npz_data = load_npz_file(working_npz_path)

# Load both NPZ files
working_npz = np.load(working_npz_path)
your_npz = np.load(my_motion_data_path)

# Print basic information about both files
print("=== Working NPZ File ===")
print("Keys:", list(working_npz.keys()))
for key in working_npz.keys():
	if hasattr(working_npz[key], 'shape'):
		print(f"{key}: shape={working_npz[key].shape}, dtype={working_npz[key].dtype}")
	else:
		print(f"{key}: {working_npz[key]}")

print("\n=== Your NPZ File ===")
print("Keys:", list(your_npz.keys()))
for key in your_npz.keys():
	if hasattr(your_npz[key], 'shape'):
		print(f"{key}: shape={your_npz[key].shape}, dtype={your_npz[key].dtype}")
	else:
		print(f"{key}: {your_npz[key]}")

# Compare body names to ensure they match
print("\n=== Body Names Comparison ===")
print("Working NPZ body names:", working_npz['body_names'])
print("Your NPZ body names:", your_npz['body_names'])
print("Match:", np.array_equal(working_npz['body_names'], your_npz['body_names']))

# Compare root motion (first few frames)
print("\n=== Root Motion Comparison (first 5 frames) ===")
print("Working NPZ root positions:")
for i in range(min(5, working_npz['body_positions'].shape[0])):
	print(f"Frame {i}: {working_npz['body_positions'][i, 0]}")

print("\nYour NPZ root positions:")
for i in range(min(5, your_npz['body_positions'].shape[0])):
	print(f"Frame {i}: {your_npz['body_positions'][i, 0]}")

# Compare root rotations (first few frames)
print("\n=== Root Rotation Comparison (first 5 frames) ===")
print("Working NPZ root rotations:")
for i in range(min(5, working_npz['body_rotations'].shape[0])):
	print(f"Frame {i}: {working_npz['body_rotations'][i, 0]}")

print("\nYour NPZ root rotations:")
for i in range(min(5, your_npz['body_rotations'].shape[0])):
	print(f"Frame {i}: {your_npz['body_rotations'][i, 0]}")

# Visualize root motion paths
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.title("Root Position X-Y Path")
plt.plot(working_npz['body_positions'][:, 0, 0], working_npz['body_positions'][:, 0, 1], 'b-', label='Working NPZ')
plt.plot(your_npz['body_positions'][:, 0, 0], your_npz['body_positions'][:, 0, 1], 'r-', label='Your NPZ')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title("Root Height Over Time")
plt.plot(working_npz['body_positions'][:, 0, 2], 'b-', label='Working NPZ')
plt.plot(your_npz['body_positions'][:, 0, 2], 'r-', label='Your NPZ')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('root_motion_comparison.png')

# Visualize joint rotations for a few key joints
key_joints = [0, 1, 2, 3]  # Root, left shoulder, right shoulder, etc.
joint_names = [your_npz['body_names'][i] for i in key_joints]

plt.figure(figsize=(12, 10))
for i, joint_idx in enumerate(key_joints):
	plt.subplot(len(key_joints), 1, i+1)
	plt.title(f"Joint {joint_names[i]} Rotation (w component)")
	plt.plot(working_npz['body_rotations'][:, joint_idx, 0], 'b-', label='Working NPZ')
	plt.plot(your_npz['body_rotations'][:, joint_idx, 0], 'r-', label='Your NPZ')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('joint_rotation_comparison.png')

# Create a function to visualize a specific frame
def visualize_frame(working_frame, your_frame, frame_idx):
	fig = plt.figure(figsize=(15, 10))

	# Working NPZ visualization
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.set_title(f"Working NPZ - Frame {frame_idx}")
	for i in range(working_frame.shape[0]):
		ax1.scatter(working_frame[i, 0], working_frame[i, 1], working_frame[i, 2], c='b', marker='o')
		# Draw lines to connect joints (simplified)
		if i > 0:
			# Connect to parent (simplified - not accurate for all skeletons)
			ax1.plot([working_frame[0, 0], working_frame[i, 0]], 
					 [working_frame[0, 1], working_frame[i, 1]], 
					 [working_frame[0, 2], working_frame[i, 2]], 'b-')

	# Your NPZ visualization
	ax2 = fig.add_subplot(122, projection='3d')
	ax2.set_title(f"Your NPZ - Frame {frame_idx}")
	for i in range(your_frame.shape[0]):
		ax2.scatter(your_frame[i, 0], your_frame[i, 1], your_frame[i, 2], c='r', marker='o')
		# Draw lines to connect joints (simplified)
		if i > 0:
			# Connect to parent (simplified - not accurate for all skeletons)
			ax2.plot([your_frame[0, 0], your_frame[i, 0]], 
					 [your_frame[0, 1], your_frame[i, 1]], 
					 [your_frame[0, 2], your_frame[i, 2]], 'r-')

	# Set equal aspect ratio and limits
	max_range = max(
		np.max(working_frame) - np.min(working_frame),
		np.max(your_frame) - np.min(your_frame)
	)
	mid_x = (np.max(working_frame[:, 0]) + np.min(working_frame[:, 0])) / 2
	mid_y = (np.max(working_frame[:, 1]) + np.min(working_frame[:, 1])) / 2
	mid_z = (np.max(working_frame[:, 2]) + np.min(working_frame[:, 2])) / 2

	ax1.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
	ax1.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
	ax1.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

	ax2.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
	ax2.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
	ax2.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

	plt.tight_layout()
	plt.savefig(f'frame_{frame_idx}_comparison.png')

# Visualize a few key frames
for frame_idx in [0, 10, 20]:
	if frame_idx < min(working_npz['body_positions'].shape[0], your_npz['body_positions'].shape[0]):
		visualize_frame(
			working_npz['body_positions'][frame_idx], 
			your_npz['body_positions'][frame_idx], 
			frame_idx
		)

# Create a corrected version based on the working NPZ
corrected_data = {}
for key in your_npz.keys():
	corrected_data[key] = your_npz[key].copy()

# 1. Match the root motion pattern
# Calculate the average direction vector in working NPZ
working_direction = np.mean(np.diff(working_npz['body_positions'][:, 0, :2], axis=0), axis=0)
working_direction = working_direction / np.linalg.norm(working_direction)

# Calculate the average direction vector in your NPZ
your_direction = np.mean(np.diff(your_npz['body_positions'][:, 0, :2], axis=0), axis=0)
your_direction = your_direction / np.linalg.norm(your_direction)

# Calculate the rotation angle between the two directions
angle = np.arccos(np.dot(your_direction, working_direction))
if np.cross(your_direction, working_direction) < 0:
	angle = -angle

# Create a rotation matrix to align the directions
cos_angle, sin_angle = np.cos(angle), np.sin(angle)
rotation_matrix = np.array([
[cos_angle, -sin_angle],
[sin_angle, cos_angle]
])

# Apply the rotation to all positions
for i in range(corrected_data['body_positions'].shape[0]):
	for j in range(corrected_data['body_positions'].shape[1]):
		xy = corrected_data['body_positions'][i, j, :2]
		corrected_data['body_positions'][i, j, :2] = np.dot(rotation_matrix, xy)

# 2. Match the root rotations
for i in range(min(corrected_data['body_rotations'].shape[0], working_npz['body_rotations'].shape[0])):
	# Copy the root rotation from working NPZ
	corrected_data['body_rotations'][i, 0] = working_npz['body_rotations'][i, 0]

# Save the corrected NPZ
np.savez('corrected_motion_data.npz', **corrected_data)
print("\nCreated corrected_motion_data.npz based on working NPZ patterns")

# Close the files
working_npz.close()
your_npz.close()

