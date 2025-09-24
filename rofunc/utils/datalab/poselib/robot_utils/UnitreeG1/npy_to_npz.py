import numpy as np

def npy_to_npz(npy_filepath, npz_filepath, array_name='data'):
    """
    Converts a single .npy file into a .npz file.

    Args:
        npy_filepath (str): Path to the input .npy file.
        npz_filepath (str): Path to the output .npz file.
        array_name (str): The name to give the array when saving it inside the .npz archive.
                          Defaults to 'data'.
    """
    try:
        # 1. Load the array from the .npy file
        data_array = np.load(npy_filepath, allow_pickle=True)
        print(f"Successfully loaded array from {npy_filepath} with shape: {data_array.shape}")

        # 2. Save the array (or multiple arrays) to a .npz file
        # np.savez allows you to pass keyword arguments, where the keyword
        # becomes the name of the array inside the .npz archive.
        # For a single array, you just pass one keyword argument.
        np.savez(npz_filepath, **{array_name: data_array})
        print(f"Successfully saved array to {npz_filepath} under the name '{array_name}'.")

    except FileNotFoundError:
        print(f"Error: File not found at {npy_filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    # Convert the dummy .npy file to .npz
    npy_to_npz('Bajo-001_xsens2g1.npy', 'my_motion_data.npz', array_name='motion_data')