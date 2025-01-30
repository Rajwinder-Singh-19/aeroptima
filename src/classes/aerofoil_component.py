'''
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


def __enforce_c0_continuity(control_points: np.ndarray) -> np.ndarray:
    """
    Enforces C0 continuity where each segment endpoints are connected.

    PARAMETERS:

        `control_points` -> Control points of all the segments of the curve. Type(np.ndarray)

    RETURNS:

        `control_points` -> C0 continuous control points where all segments are well connected. Type(np.ndarray)

    """
    for i in range(control_points.shape[2] - 1):  # Iterate over segments
        control_points[0, :, i + 1] = control_points[3, :, i]  # Match end to start
    return control_points


def __enforce_c1_continuity(control_points: np.ndarray) -> np.ndarray:
    """
    Enforces C1 continuity where each segment endpoint tangents are aligned.

    PARAMETERS:

        `control_points` -> Control points of all the segments of the curve. Type(np.ndarray)

    RETURNS:

        `control_points` -> C1 continuous control points where all segment tangents are aligned. Type(np.ndarray)

    """
    for i in range(control_points.shape[2] - 1):  # Iterate over segments
        # Compute the tangent at the end of the current segment
        tangent = control_points[3, :, i] - control_points[2, :, i]
        # Adjust the tangent of the next segment
        control_points[1, :, i + 1] = control_points[0, :, i + 1] + tangent
    return control_points


def __enforce_c2_continuity(control_points: np.ndarray) -> np.ndarray:
    """
    Enforces C2 continuity (smooth second derivative) across Bézier curve segments.

    PARAMETERS:

        `control_points` -> Control points of all the segments of the curve. Type(np.ndarray)

    RETURNS:
        `control_points` -> Modified control points with enforced C2 continuity. Type(np.ndarray)
    """
    num_segments = control_points.shape[2]

    for i in range(num_segments - 1):  # Iterate over each segment boundary
        # Compute the curvature condition
        p_prev = control_points[2, :, i]  # Third control point of the current segment
        p_next = control_points[
            0, :, i + 1
        ]  # First control point of the next segment (C0 already enforced)
        p_end = control_points[3, :, i]  # End point of the current segment
        """p_start = control_points[
            1, :, i + 1
        ]"""  # Second control point of the next segment

        # Adjust the second control point of the next segment for C2 continuity
        control_points[1, :, i + 1] = 2 * p_next - p_prev - (p_end - p_next)

    return control_points


def enforce_continuity(control_points: np.ndarray) -> np.ndarray:
    """
    Wrapper function to enforce C0, C1 and C2 continuity

    PARAMETERS:

        `control_points` -> Control points of all the segments of the curve

    RETURNS:
        `control_points` -> C0, C1 and C2 continuous control points

    """
    control_points = __enforce_c0_continuity(control_points)
    control_points = __enforce_c1_continuity(control_points)
    control_points = __enforce_c2_continuity(control_points)
    return control_points


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
        self.upper_control[0,0,0] = 0.0
        self.upper_control[0,1,0] = 0.0
        self.upper_control[-1,0,-1] = 1.0
        self.upper_control[-1,1,-1] = 0.0
        self.lower_control[0,0,0] = 0.0
        self.lower_control[0,1,0] = 0.0
        self.lower_control[-1,0,-1] = 1.0
        self.lower_control[-1,1,-1] = 0.0
        self.upper_control = enforce_continuity(self.upper_control)
        self.lower_control = enforce_continuity(self.lower_control)
        self.save_foil("BezierFoil", "Foil", "Foil.dat", 10, 8)
        outputs["area"] = compute_area("Foil/Foil.dat")

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
        outputs["cl"] = cl
        outputs["cd"] = cd
        outputs["cl_cd"] = -(cl / cd)


if __name__ == "__main__":
    from database.UIUC_aerofoils import UIUC_DATABASE as UDB

    n = 7
    Foil = BezierFoil(UDB["naca0006_dat"], n_segments=n)
    control_vector = gen_control_vector(Foil)
    init_area = compute_area("UIUC_aerofoils/" + UDB["naca0006_dat"])
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

    #prob.model.add_constraint(
    #    "Aerofoil_comp.area", lower=0.60 * init_area, upper=1.1 * init_area
    #)
    prob.driver = om.ScipyOptimizeDriver()
    prob.model.approx_totals()
    prob.driver.options["optimizer"] = "nelder-mead"
    prob.setup()
    prob.run_driver()
'''

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
    index_mid = 4 * 2 * n_segments  # 4 points * 2 coordinates * n_segments
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
    reynolds: float,
    ncrit: int,
    niter: int,
    alfa: float,
):
    return f"""
    LOAD {folder}/{foil_dat_file}
    PPAR
    N 300
    \n
    OPER
    VISC
    {reynolds}
    VPAR
    N {ncrit}
    ITER {niter}
    PACC
    {folder}/Data.dat
    {folder}/Dump.dat
    ALFA {alfa}
    \n
    QUIT
    """


def __enforce_c0_continuity(control_points: np.ndarray) -> np.ndarray:
    for i in range(control_points.shape[2] - 1):
        control_points[0, :, i + 1] = control_points[3, :, i]
    return control_points


def enforce_continuity(control_points: np.ndarray) -> np.ndarray:
    return __enforce_c0_continuity(control_points)


def is_airfoil_valid(points: np.ndarray) -> bool:
    try:
        split_idx = len(points) // 2
        upper = points[:split_idx]
        lower = points[split_idx:]

        # Check x-monotonicity with tolerance
        if not (np.all(np.diff(upper[:, 0]) >= -1e-6) and 
                np.all(np.diff(lower[:, 0]) >= -1e-6)):
            return False

        # Check thickness with aligned points
        lower_flipped = lower[::-1]
        if len(upper) != len(lower_flipped):
            return False
            
        thickness = upper[:, 1] - lower_flipped[:, 1]
        if np.any(thickness < -1e-6):
            return False

        return True
    except Exception as e:
        print(f"Validity check error: {str(e)}")
        return False


class Aerofoil(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("bezier_foil", types=BezierFoil)

    def setup(self):
        self.add_input(
            "control_vector",
            shape_by_conn=True,
            desc="Control points of both surfaces flattened",
        )
        self.add_input(
            "n_segments",
            val=1,
            desc="Number of segments per surface",
        )
        self.add_output("cl", val=0.0)
        self.add_output("cd", val=1000.0)
        self.add_output("cl_cd", val=0.0)
        self.add_output("area", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # Initialize with penalty values
        cl, cd = 0.0, 1000.0
        area = 0.0
        
        try:
            # Extract scalar value for n_segments
            n_segments = int(inputs["n_segments"].item())
            
            # Split control points
            upper, lower = split_upper_lower_control(
                inputs["control_vector"], n_segments
            )

            # Set fixed points
            upper[0, 0, 0] = 0.0
            upper[0, 1, 0] = 0.0
            upper[3, 0, -1] = 1.0
            upper[3, 1, -1] = 0.0
            lower[0, 0, 0] = 0.0
            lower[0, 1, 0] = 0.0
            lower[3, 0, -1] = 1.0
            lower[3, 1, -1] = 0.0

            # Enforce continuity
            upper = enforce_continuity(upper)
            lower = enforce_continuity(lower)

            # Update Bezier foil
            bezier_foil = self.options["bezier_foil"]
            bezier_foil.upper_control = upper
            bezier_foil.lower_control = lower

            # Generate airfoil file
            bezier_foil.save_foil("BezierFoil", "Foil", "Foil.dat", 10, 8)
            
            # Check validity
            points = np.loadtxt(os.getcwd()+"/Foil/Foil.dat", skiprows=3)
            if not is_airfoil_valid(points):
                raise ValueError("Invalid airfoil geometry")

            # Compute area
            area = compute_area(os.getcwd()+"/Foil/Foil.dat")
            outputs["area"] = area

            # Run XFOIL analysis
            if os.path.exists(os.getcwd()+"/Foil/Foil.dat"):
                os.remove(os.getcwd()+"/Foil/Foil.dat")

            xfoil_commands = gen_xfoil_commands("Foil", "Foil.dat", 6.2e6, 10, 1000, 0)
            process = subprocess.Popen(
                ["xfoil.exe"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            stdout, stderr = process.communicate(xfoil_commands)
            process.wait()

            # Parse results
            if process.returncode == 0:
                with open(os.getcwd()+"Foil/Data.dat", "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        last_line = lines[-1].split()
                        if len(last_line) >= 3:
                            cl = float(last_line[1])
                            cd = float(last_line[2])
            else:
                print(f"XFOIL failed with error:\n{stderr}")

        except Exception as e:
            print(f"Error in computation: {str(e)}")
            cl, cd = 0.0, 1000.0

        finally:
            outputs["cl"] = cl
            outputs["cd"] = cd
            outputs["cl_cd"] = -cl / cd if cd != 0 else -1000
            outputs["area"] = area


if __name__ == "__main__":
    from database.UIUC_aerofoils import UIUC_DATABASE as UDB

    # Initialize baseline airfoil
    n_segments = 5
    base_foil = BezierFoil(UDB["naca0006_dat"], n_segments=n_segments)
    init_area = compute_area(f"UIUC_aerofoils/{UDB['naca0006_dat']}")

    # Setup optimization problem
    prob = om.Problem()
    model = prob.model

    # Add independent variables
    model.add_subsystem(
        "af",
        Aerofoil(bezier_foil=base_foil),
        promotes_inputs=["control_vector", "n_segments"]
    )

    # Design variables and constraints
    model.add_design_var("control_vector", 
                       lower=-0.5, upper=1.5,  # Reasonable bounds for coordinates
                       indices=np.arange(len(gen_control_vector(base_foil))))
    model.add_constraint("af.area", lower=0.6*init_area, upper=1.1*init_area)
    model.add_objective("af.cl_cd")

    # Setup driver
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-6)
    prob.driver.options["debug_print"] = ["objs", "desvars", "nl_cons"]

    # Initialize variables
    prob.model.set_input_defaults(
        "control_vector", 
        gen_control_vector(base_foil)
    )
    prob.model.set_input_defaults("n_segments", n_segments)

    # Setup and run
    prob.setup()
    prob.run_driver()

    # Print results
    print("\nOptimization complete:")
    print(f"Final CL/CD: {-prob.get_val('af.cl_cd')[0]:.2f}")
    print(f"Area: {prob.get_val('af.area')[0]:.5f} (Initial: {init_area:.5f})")