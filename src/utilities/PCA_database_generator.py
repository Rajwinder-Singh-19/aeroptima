import numpy as np
import os
from multiprocessing import Pool, cpu_count
from classes.bezierfoil import BezierFoil
from database.PCA_aerofoils import NACAFoil as NF

DATASET_PATH = os.getcwd() + "/src/database/PCA_files/NACA/NACA_bezier_controls.npy"
N_SEGMENTS = 10


def process_airfoil(aerofoil_name):
    """Processes a single airfoil and returns its control points."""
    try:
        bezfoil = BezierFoil(NF[aerofoil_name], n_segments=N_SEGMENTS)
        return np.concatenate([bezfoil.upper_control.flatten(), bezfoil.lower_control.flatten()])
    except Exception as e:
        print(f"Skipping {aerofoil_name} due to error: {e}")
        return None


if __name__ == "__main__":
    # Use all available CPU cores
    num_workers = 4  # Adjust this based on your system

    with Pool(num_workers) as pool:
        dataset = pool.map(process_airfoil, NF.keys())

    # Remove failed cases (None values)
    dataset = np.array([data for data in dataset if data is not None])

    np.save(DATASET_PATH, dataset)
    print(f"Dataset saved to {DATASET_PATH}")
