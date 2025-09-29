NOTESSSSS


Looking at your code, I can see the issue. The chest/pelvis facing forward while limbs face backwards suggests there's an inconsistent rotation transformation being applied. Let me help you fix this.

The main issue appears to be in how you're applying rotations to different body parts. Here's what I think is happening and how to fix it:

Key Issues:
Inconsistent rotation transformation: You're negating w and z components for all joints, but this might not be appropriate for all body parts.
Coordinate system mismatch: The transformation from XSens to G1 coordinate system needs to be handled more carefully.
Local vs Global rotations: The retargeting might be producing correct local rotations, but the global rotations need additional transformation.
Here's the fixed version of the critical section:
# Replace the body rotation section (around line 285-300) with this:

# Apply coordinate system transformation
# XSens to G1/IsaacSim requires careful handling of each body part
for i in range(N):
for j, idx in enumerate(body_indices):
    # Position transformation: Keep X, swap Y and Z
    x, y, z = global_positions[i, idx]
    body_positions[i, j] = np.array([x, z, y])
    
    # Get the original rotation
    w, x, y, z = global_rotations[i, idx]
    
    # Apply different transformations based on body part type
    body_name = isaac_body_names[j]
    
    # For the pelvis and torso, we need a different transformation
    if body_name in ["pelvis", "torso_link"]:
        # These are already facing the correct direction
        # Just apply the coordinate swap
        body_rotations[i, j] = np.array([w, x, z, y])
    else:
        # For limbs, we need to apply additional rotation
        # First apply coordinate swap
        swapped_quat = np.array([w, x, z, y])
        
        # Then apply 180° rotation around the up axis (Z in IsaacSim)
        # This is quaternion multiplication with [0, 0, 0, 1] (180° around Z)
        z_180_quat = np.array([0, 0, 0, 1])  # 180° around Z in [w,x,y,z]
        
        # Quaternion multiplication to apply the rotation
        body_rotations[i, j] = quaternion_multiply(swapped_quat, z_180_quat)

# Normalize all quaternions to ensure they're valid
for i in range(N):
for j in range(len(body_indices)):
    norm = np.linalg.norm(body_rotations[i, j])
    if norm > 0:
        body_rotations[i, j] /= norm
    else:
        body_rotations[i, j] = np.array([1, 0, 0, 0])  # Identity quaternion

Additional fixes:
Update the retarget config rotation to better match the coordinate systems:
"rotation": [0.5, 0.5, 0.5, 0.5],  # This represents a 120° rotation that aligns XSens to G1

Add joint-specific corrections for problematic joints:
# After the main rotation transformation, add specific corrections
joint_corrections = {
"left_shoulder_pitch_link": np.array([0.7071, 0, 0.7071, 0]),  # 90° around Y
"right_shoulder_pitch_link": np.array([0.7071, 0, -0.7071, 0]),  # -90° around Y
# Add more corrections as needed
}

for i in range(N):
for j, body_name in enumerate(isaac_body_names):
    if body_name in joint_corrections:
        # Apply correction
        body_rotations[i, j] = quaternion_multiply(body_rotations[i, j], joint_corrections[body_name])

Fix the angular velocity calculation to account for the transformation:
# After calculating body_angular_velocities, apply coordinate transformation
for j in range(len(isaac_body_names)):
# Swap Y and Z components to match position transformation
temp = body_angular_velocities[:, j, 1].copy()
body_angular_velocities[:, j, 1] = body_angular_velocities[:, j, 2]
body_angular_velocities[:, j, 2] = temp

Debug visualization to verify orientations:
# Add this after computing body_rotations to visualize orientations
if visualize and i == 0:  # Check first frame
print(f"\nFirst frame orientations:")
for j, body_name in enumerate(isaac_body_names):
    quat = body_rotations[0, j]
    # Convert to Euler angles for easier understanding
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to scipy format
    euler = r.as_euler('xyz', degrees=True)
    print(f"{body_name}: euler={euler}")


The key insight is that different body parts may need different rotation transformations based on their orientation in the source skeleton versus the target skeleton. The pelvis and torso might already be correctly oriented, while the limbs need additional rotation to face forward.

Try these fixes and let me know if the orientation issues persist!