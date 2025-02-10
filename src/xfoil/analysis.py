"""
xfoil.analysis
================

Python wrapper on XFOIL 6.99 application to perform aerodynamic analysis on aerofoils
"""

import os
import subprocess


def __gen_xfoil_commands(
    folder: str,
    foil_dat_file: str,
    cadd_adj_no: int,
    angle_thresh: float,
    n_pts: int,
    reynolds: float,
    ncrit: int,
    niter: int,
    alfa: float,
):

    return f"""
    LOAD {folder}/{foil_dat_file}
    GDES CADD {cadd_adj_no} {angle_thresh} 0.0 1.0
    \n
    PCOP
    PPAR n {n_pts}
    \n
    OPER
    VISC
    {reynolds}
    VPAR
    N {ncrit} \n
    ITER {niter}
    PACC
    {folder}/Data.dat
    {folder}/Dump.dat
    ALFA {alfa}
    \n
    QUIT
    """


def aero_analysis(
    folder: str,
    foil_dat_file: str,
    cadd_adj_no: int,
    angle_thresh: float,
    n_pts: int,
    reynolds: float,
    ncrit: int,
    niter: int,
    alfa: float,
    timeout: int = 5,  # ⏳ Added timeout parameter
):
    """
    Runs XFoil analysis with a timeout to handle cases where it freezes.
    If XFoil fails, the function returns None.

    PARAMETERS:
        (same as before)

    RETURNS:
        Tuple (cl, cd, cl/cd) if successful, else None.
    """
    filepath = os.path.join(os.getcwd(), folder, foil_dat_file)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Aerofoil file '{foil_dat_file}' not found in '{folder}'"
        )

    # Remove old XFoil output files if they exist
    for filename in ["Data.dat", "Dump.dat"]:
        path = os.path.join(os.getcwd(), folder, filename)
        if os.path.isfile(path):
            os.remove(path)

    command = __gen_xfoil_commands(
        folder,
        foil_dat_file,
        cadd_adj_no,
        angle_thresh,
        n_pts,
        reynolds,
        ncrit,
        niter,
        alfa,
    )

    try:
        process = subprocess.Popen(
            ["xfoil.exe"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        process.communicate(command, timeout=timeout)  # ⏳ Enforce timeout
        process.wait()

    except subprocess.TimeoutExpired:
        print(f"XFoil timed out for {foil_dat_file}. Skipping.")
        process.kill()
        return None  # Skip this perturbation

    # Check if XFoil produced a valid output
    data_path = os.path.join(os.getcwd(), folder, "Data.dat")
    if not os.path.isfile(data_path):
        print(f"XFoil failed to generate results for {foil_dat_file}. Skipping.")
        return None

    try:
        with open(data_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines:
                last_line = lines[-1].split()
                cl = float(last_line[1])
                cd = float(last_line[2])
                return cl, cd, cl / cd
    except Exception as e:
        print(f"Error reading results for {foil_dat_file}: {e}")
        return None

    return None  # If no valid data is found
