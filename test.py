import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

my_motion_data_path = 'my_motion_data.npz'
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

print("Working NPZ body names:", working_npz['body_names'])

