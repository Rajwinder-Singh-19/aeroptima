import openmdao.api as om
import numpy as np
from classes.bezierfoil import BezierFoil
import os
import subprocess
import math
from parser.parsefoil import *


def compute_area(file: str):
    FILEPATH = os.getcwd() + "/" + file
    with open(FILEPATH, "r") as f:
        lines = f.readlines()

    points = []

    for line in lines[3:]:
        try:
            x, y = map(float, line.strip().split())
            points.append((x, y))
        except ValueError:
            continue

    points = np.array(points)
    n = len(points)
    area = 0.0
    for i in range(n - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        area += (x2 - x1) * (y2 + y1)
    return math.fabs(area)


def split_upper_lower_control(control_vector: np.ndarray, n_segments: int):
    index_mid = len(control_vector) // 2

    upper_vector = control_vector[:index_mid]
    lower_vector = control_vector[index_mid:]

    upper_control = upper_vector.reshape(4, 2, n_segments)
    lower_control = lower_vector.reshape(4, 2, n_segments)
    return upper_control, lower_control


def gen_control_vector(Foil: BezierFoil):
    upper_control = Foil.upper_control
    lower_control = Foil.lower_control
    control_vector = np.hstack((upper_control.flatten(), lower_control.flatten()))
    return control_vector


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
    GDES CADD {cadd_adj_no} 5 0.0 1.0
    \n
    PCOP
    PPAR n 300
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
        self.add_input(
            "n_segments",
            val=2,
            desc="Number of upper and lower surface segments",
        )
        self.add_output("cl", val=1000, desc="Coefficient of Lift")
        self.add_output("cd", val=1000, desc="Coefficient of Drag")
        self.add_output("cl_cd", val=1000, desc="negative of L/D ratio")
        self.add_output("area", val=0, desc="Area enclosed by the aerofoil")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        self.upper_control, self.lower_control = split_upper_lower_control(
            inputs["control_vector"], int(inputs["n_segments"])
        )
        self.save_foil("BezierFoil", "Foil", "Foil.dat", 10, 8)
        outputs["area"] = compute_area("Foil/Foil.dat")

        if os.path.isfile(os.getcwd() + "/Foil/Data.dat"):
            os.remove(os.getcwd() + "/Foil/Data.dat")

        if os.path.isfile(os.getcwd() + "/Foil/Dump.dat"):
            os.remove(os.getcwd() + "/Foil/Dump.dat")

        xfoil_commands = gen_xfoil_commands("Foil", "Foil.dat", 2, 6.2e6, 5, 1000, 0)
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
        outputs["cl"] = cl
        outputs["cd"] = cd
        outputs["cl_cd"] = -(cl / cd)


if __name__ == "__main__":
    from database.UIUC_aerofoils import UIUC_DATABASE as UDB

    n = 7
    Foil = BezierFoil(UDB['naca0006_dat'], n_segments=n)
    control_vector = gen_control_vector(Foil)
    init_area = compute_area("UIUC_aerofoils/" + UDB['naca0006_dat'])
    model = om.Group()

    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output("control_vector", val=control_vector)
    model.add_subsystem("indep_var_comp", indep_var_comp, promotes=["*"])
    model.add_subsystem(
        "Aerofoil_comp", Aerofoil(), promotes_inputs=["control_vector", "n_segments"]
    )

    prob = om.Problem(model)
    prob.model.set_input_defaults("control_vector", control_vector)
    prob.model.set_input_defaults("n_segments", n)
    prob.model.add_design_var(
        "control_vector",
    )
    prob.model.add_objective("Aerofoil_comp.cl_cd")

    prob.model.add_constraint("Aerofoil_comp.area", lower=0.80 * init_area)
    prob.driver = om.ScipyOptimizeDriver()
    prob.model.approx_totals()
    prob.driver.options["optimizer"] = "trust-constr"
    prob.setup()
    prob.run_driver()
