import os
import subprocess


def gen_xfoil_commands(
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
):
    if not os.path.isfile(os.getcwd() + f"/{folder}/{foil_dat_file}"):
        raise FileNotFoundError(
            f"Aerofoil Coordinates file {foil_dat_file} file not found in the folder {folder}"
        )

    if os.path.isfile(os.getcwd() + f"/{folder}/Data.dat"):
        os.remove(os.getcwd() + f"/{folder}/Data.dat")

    if os.path.isfile(os.getcwd() + f"/{folder}/Dump.dat"):
        os.remove(os.getcwd() + f"/{folder}/Dump.dat")

    command = gen_xfoil_commands(
        folder, foil_dat_file, cadd_adj_no, angle_thresh, n_pts, reynolds, ncrit, niter, alfa
    )
    process = subprocess.Popen(
        ["xfoil.exe"],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.communicate(command)
    process.wait()

    with open(os.getcwd() + f"/{folder}/Data.dat", "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        if lines:
            last_line = lines[-1].split()
            cl = float(last_line[1])
            cd = float(last_line[2])

    return cl, cd
