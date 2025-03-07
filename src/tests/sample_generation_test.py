import numpy as np
from classes.bezierfoil import BezierFoil
from database.UIUC_aerofoils import UIUC_DATABASE as UDB
from xfoil.analysis import aero_analysis
import pandas as pd
from bezier.spline import __enforce_continuity
import os

CSV_PATH = (
    os.getcwd()
    + "/src/database/NACA_perturbed_dataset/NACA_PCA_perturbed_airfoil_data_seed_griffith30SymSuction_dat.csv"
)

INITIAL_FOIL = BezierFoil(
    UDB["griffith30SymSuction_dat"],
    n_segments=10,
    pca_components="NACA/naca_pca_components.npy",
    mean_pca_foil="NACA/naca_pca_mean_airfoil.npy",
)
INITIAL_UPPER = INITIAL_FOIL.upper_control
INITIAL_LOWER = INITIAL_FOIL.lower_control
INITIAL_FOIL.close_curve()
PERTURB_SCALE = 0.15
NUM_PERTURBATIONS = 1000
data = []

INITIAL_FOIL.save_foil("PCA_Foil", "Foil", "Foil.dat", 10, 8)
(cl, cd) = aero_analysis("Foil", "Foil.dat", 1, 10, 300, 6e6, 10, 1000, 0)[0:2]

data.append([*INITIAL_FOIL.upper_control.flatten(), cl, cd])

for i in range(NUM_PERTURBATIONS):
    coefficients = np.random.uniform(-PERTURB_SCALE, PERTURB_SCALE, size=5)
    INITIAL_FOIL.perturb_pca(coefficients, 0)
    INITIAL_FOIL.upper_control = __enforce_continuity(INITIAL_FOIL.upper_control)
    INITIAL_FOIL.lower_control = __enforce_continuity(INITIAL_FOIL.lower_control)
    INITIAL_FOIL.close_curve()

    try:
        INITIAL_FOIL.save_foil("PCA_Foil", "Foil", "Foil.dat", 10, 8)
        (cl, cd) = aero_analysis("Foil", "Foil.dat", 1, 10, 300, 6e6, 10, 1000, 0, 1)[
            0:2
        ]
    except:
        cl = None
        cd = None
        INITIAL_FOIL.upper_control = INITIAL_UPPER
        INITIAL_FOIL.lower_control = INITIAL_LOWER

    cl = cl if cl is not None else np.nan
    cd = cd if cd is not None else np.nan

    data.append(
        [
            *INITIAL_FOIL.upper_control.flatten(),
            *INITIAL_FOIL.upper_control.flatten(),
            cl,
            cd,
        ]
    )

df = pd.DataFrame(
    data,
    columns=[f"u{i}" for i in range(80)] + [f"l{i}" for i in range(80)] + ["Cl", "Cd"],
)

df.to_csv(CSV_PATH, index=False)
