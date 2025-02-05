from classes.bezierfoil import BezierFoil
from database.UIUC_aerofoils import UIUC_DATABASE as UDB
import matplotlib.pyplot as plt
from xfoil.analysis import aero_analysis

foil = BezierFoil(UDB["naca001034a08cli0_2_dat"], n_segments=10)
foil.close_curve()

foil.save_foil("A18_Bezier", "Foil", "Foil.dat", 10, 8)

cl, cd = aero_analysis("Foil", "Foil.dat", 1, 15.0, 300, 6e6, 10, 1000, 0.0)

print(cl)
print(cd)
