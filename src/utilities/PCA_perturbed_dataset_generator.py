import numpy as np
from classes.bezierfoil import BezierFoil
from database.UIUC_aerofoils import UIUC_DATABASE as UDB
from xfoil.analysis import aero_analysis
import pandas as pd
from bezier.spline import __enforce_continuity
import os


def generate_aero_dataset(
    seed: str,
    dataset_size: str,
    pca_components: str,
    mean_pca_foil: str,
    n_pca_components: int,
    perturb_scale: float,
    cadd_adj_no: int,
    angle_thresh: float,
    n_pts: int,
    reynolds: float,
    ncrit: int,
    niter: int,
    alfa: float,
    timeout: int,
    save_path: str,
):
    INITIAL_FOIL = BezierFoil(
        seed,
        n_segments=10,
        pca_components=pca_components,
        mean_pca_foil=mean_pca_foil,
    )
    INITIAL_UPPER = INITIAL_FOIL.upper_control
    INITIAL_LOWER = INITIAL_FOIL.lower_control
    INITIAL_FOIL.close_curve()
    data = []

    INITIAL_FOIL.save_foil("PCA_Foil", "Foil", "Foil.dat", 10, 8)
    (cl, cd) = aero_analysis(
        "Foil",
        "Foil.dat",
        cadd_adj_no,
        angle_thresh,
        n_pts,
        reynolds,
        ncrit,
        niter,
        alfa,
        5,
    )[0:2]

    data.append([*INITIAL_FOIL.upper_control.flatten(), cl, cd])

    for i in range(dataset_size):
        coefficients = np.random.uniform(
            -perturb_scale, perturb_scale, size=n_pca_components
        )
        INITIAL_FOIL.perturb_pca(coefficients, 0)
        INITIAL_FOIL.upper_control = __enforce_continuity(INITIAL_FOIL.upper_control)
        INITIAL_FOIL.lower_control = __enforce_continuity(INITIAL_FOIL.lower_control)
        INITIAL_FOIL.close_curve()

        try:
            INITIAL_FOIL.save_foil("PCA_Foil", "Foil", "Foil.dat", 10, 8)
            (cl, cd) = aero_analysis(
                "Foil",
                "Foil.dat",
                cadd_adj_no,
                angle_thresh,
                n_pts,
                reynolds,
                ncrit,
                niter,
                alfa,
                timeout,
            )[0:2]
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
        columns=[f"u{i}" for i in range(80)]
        + [f"l{i}" for i in range(80)]
        + ["Cl", "Cd"],
    )

    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    CSV_PATH = (
        os.getcwd()
        + "/src/database/NACA_perturbed_dataset/NACA_PCA_perturbed_airfoil_data_seed_griffith30SymSuction_dat.csv"
    )
    generate_aero_dataset(
        UDB["griffith30SymSuction_dat"],
        1000,
        "NACA/naca_pca_components.npy",
        "NACA/naca_pca_mean_airfoil.npy",
        5,
        0.15,
        1,
        10,
        300,
        6e6,
        10,
        1000,
        0,
        3,
        CSV_PATH,
    )
