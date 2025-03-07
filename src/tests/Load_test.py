import joblib
import numpy as np
from src.database.UIUC_aerofoils import UIUC_DATABASE as UDB
from classes.bezierfoil import BezierFoil
from xfoil.analysis import aero_analysis
from bezier.spline import __enforce_continuity

kriging_model = joblib.load("gpr_3000.pkl")


TEST_FOIL = BezierFoil(UDB['davis_corrected_dat'], 10, pca_components="NACA/naca_pca_components.npy",
    mean_pca_foil="NACA/naca_pca_mean_airfoil.npy")
TEST_FOIL.upper_control = __enforce_continuity(TEST_FOIL.upper_control)
TEST_FOIL.lower_control = __enforce_continuity(TEST_FOIL.lower_control)
TEST_FOIL.close_curve()
upper = TEST_FOIL.upper_control.flatten()
lower = TEST_FOIL.lower_control.flatten()
test_control = np.hstack((upper, lower))

y_plate = kriging_model.predict(test_control.reshape(1, -1))

TEST_FOIL.save_foil("TestFoil", "Foil", "Foil.dat", 10, 10)

cd = aero_analysis("Foil", "Foil.dat", 1, 10, 300, 6e6, 10, 1000, 0, 5)[1]

print(f"Predicted: {y_plate}")
print(f"Xfoil: {cd}")