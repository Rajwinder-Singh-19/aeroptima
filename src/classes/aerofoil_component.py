import openmdao.api as om
import numpy as np
from classes.bezierfoil import BezierFoil
import os
import subprocess


def split_upper_lower_control(control_vector: np.ndarray, n_segments: int):
    index_mid = len(control_vector) // 2

    upper_vector = control_vector[:index_mid]
    lower_vector = control_vector[index_mid:]

    upper_control = upper_vector.reshape(4, 2, n_segments)
    lower_control = lower_vector.reshape(4, 2, n_segments)
    return upper_control, lower_control


def gen_xfoil_commands(
    folder: str,
    foil_dat_file: str,
    cadd_adj_no: int,
    reynolds: float,
    ncrit: int,
    niter: int,
    alfa: float,
):
    return f"""
    LOAD {folder}/{foil_dat_file}
    GDES CADD {cadd_adj_no} 10 0.0 1.0
    \n
    PCOP
    PPAR n 250
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


class Aerofoil(om.ExplicitComponent, BezierFoil):
    def setup(self):
        self.add_input(
            "control_vector",
            shape_by_conn=True,
            desc="Control points of both upper and lower surface flattened",
        )
        self.add_output(
            "aero_coeffs", val=np.zeros(3), desc="Coefficients of Lift, Drag and Cl/Cd"
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        self.upper_control, self.lower_control = split_upper_lower_control(
            inputs["control_vector"], len(inputs["control_vector"] // 16)
        )
        self.save_foil("BezierFoil", "Foil", "Foil.dat", 10, 8)

        if os.path.isfile(os.getcwd() + "/Foil/Data.dat"):
            os.remove(os.getcwd() + "/Foil/Data.dat")

        if os.path.isfile(os.getcwd() + "/Foil/Dump.dat"):
            os.remove(os.getcwd() + "/Foil/Dump.dat")

        xfoil_commands = gen_xfoil_commands("Foil", "Foil.dat", 2, 6.2e6, 10, 1000, 0)
        process = subprocess.Popen(
            ["xfoil.exe"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        process.communicate(xfoil_commands)
        process.wait()

        cl = -1000.0
        cd = -1000.0

        POLAR_PATH = os.getcwd() + "/Foil/Data.dat"
        with open(POLAR_PATH, "r") as f:
            last = f.readlines()[-1]
            vals = last.split()
            cl = float(vals[1])
            cd = float(vals[2])
        outputs["aero_coeffs"][0] = cl
        outputs["aero_coeffs"][1] = cd
        outputs["aero_coeffs"][2] = cl / cd
