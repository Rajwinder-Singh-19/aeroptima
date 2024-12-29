import numpy as np
from bezier.castigliano import cubic as cb
from scipy.optimize import curve_fit
from parser.aerofoil import *
from database.aerofoil_data import UIUC_DATABASE as UDB
import math


def cubic(start, mid_1, mid_2, end, t):
    term_1 = (1 - t) ** 3 * start
    term_2 = 3 * (1 - t) ** 2 * t * mid_1
    term_3 = 3 * (1 - t) * t**2 * mid_2
    term_4 = t**3 * end
    return term_1 + term_2 + term_3 + term_4


def bezier_curve(t, *control_points):
    """Compute a Bezier curve using control points."""
    n = len(control_points) - 1
    return sum(
        control_points[i] * (math.comb(n, i) * (1 - t) ** (n - i) * t**i)
        for i in range(n + 1)
    )


def fit_bezier(curve, degree):
    """Fits a Bezier curve to given x and y coordinates."""
    t = np.linspace(0, 1, len(curve))  # Parameterize chordwise distance

    def bezier_fit_func(t, *control_points):
        return bezier_curve(t, *control_points)

    # Initial guess for control points
    p0 = np.linspace(min(curve[:, 1]), max(curve[:, 1]), degree + 1)

    control_points, _ = curve_fit(bezier_fit_func, t, curve[:, 1], p0=p0)
    return control_points


def reconstruct_bezier(curve, control_points):
    """Reconstruct airfoil surface using Bezier control points."""
    t = np.linspace(0, 1, len(curve[:, 0]))
    return bezier_curve(t, *control_points)


if __name__ == "__main__":
    upper, lower = split_surfaces(UDB["fx66h80_dat"])
    degree = 6  # Degree of Bezier curve (adjustable)
    upper_control_points = fit_bezier(upper, degree)
    lower_control_points = fit_bezier(lower, degree)
    print(upper_control_points)
    upper_reconstructed = reconstruct_bezier(upper, upper_control_points)
    lower_reconstructed = reconstruct_bezier(lower, lower_control_points)

    # Plot original vs reconstructed
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(upper[:, 0], upper[:, 1], label="Original Upper Surface", color="red")
    plt.plot(
        upper[:, 0],
        upper_reconstructed,
        "--",
        label="Bezier Upper Surface",
        color="black",
    )
    plt.plot(lower[:, 0], lower[:, 1], label="Original Lower Surface", color="blue")
    plt.plot(
        lower[:, 0],
        lower_reconstructed,
        "--",
        label="Bezier Lower Surface",
        color="orange",
    )
    plt.legend()
    plt.title("Bezier Curve Fitting for Airfoil")
    plt.xlabel("x (Chordwise)")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()
