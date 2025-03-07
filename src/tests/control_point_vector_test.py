import numpy as np
from classes.bezierfoil import BezierFoil
from database.UIUC_aerofoils import UIUC_DATABASE as UDB
import time


def split_upper_lower_control(control_vector: np.ndarray, n_segments: int):
    index_mid = len(control_vector) // 2

    upper_vector = control_vector[:index_mid]
    lower_vector = control_vector[index_mid:]

    upper_control = upper_vector.reshape(4, 2, n_segments)
    lower_control = lower_vector.reshape(4, 2, n_segments)
    return upper_control, lower_control


start = time.time()
Foil = BezierFoil(
    database_index=UDB["atr72sm_dat"],
    n_segments=2,
    pca_components="NACA/naca_pca_components.npy",
    mean_pca_foil="NACA/naca_pca_mean_airfoil.npy",
)
end = time.time()
exec_time = end - start
upper_control = Foil.upper_control
lower_control = Foil.lower_control
control_vector = np.hstack((upper_control.flatten(), lower_control.flatten()))

upper_test, lower_test = split_upper_lower_control(control_vector, 2)
print(upper_test == upper_control)
print(lower_test == lower_control)
print(f"Instantiation took {exec_time} seconds")
