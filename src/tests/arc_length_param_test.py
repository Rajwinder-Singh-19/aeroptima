from classes.bezierfoil import BezierFoil
from xfoil.analysis import aero_analysis
from database.UIUC_aerofoils import UIUC_DATABASE as UDB

TEST_FOIL = BezierFoil(UDB['a18_dat'], param_method='arc_length', arc_length=0.05)

TEST_FOIL.save_foil("TestFoil", "Foil", "Foil.dat", 10, 5)

Result = aero_analysis("Foil", "Foil.dat", 1, 10, 300, 6e6, 10, 1000, 0, 10)

print(f"{Result[0]:.1f}, {Result[1]:.3f}")
print(f"{TEST_FOIL.n_segments:.1f}")
print(TEST_FOIL.upper_control.flatten().size)