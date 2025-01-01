import numpy as np
from scipy.optimize import minimize
from bezier.cubic import cubic_bezier_curve


def __residuals(flat_control_points, curve_data):
    num_points = len(curve_data)
    control_points = np.array(flat_control_points.reshape(4, 2))
    bezier_points = cubic_bezier_curve(*control_points, n_points=num_points)
    return np.sum(np.linalg.norm(bezier_points - curve_data))


def __fit_bezier(curve_data, method):
    P0 = curve_data[0]
    P3 = curve_data[-1]
    P1 = curve_data[len(curve_data) // 3]
    P2 = curve_data[2 * len(curve_data) // 3]

    result = minimize(
        __residuals,
        np.array([P0, P1, P2, P3]).flatten(),
        args=(curve_data,),
        method=method,
        tol=1e-15,
    )

    return result.x.reshape(4, 2)


def __split_data(curve_data, num_segments):
    segment_size = len(curve_data) // num_segments
    return [
        curve_data[i : i + segment_size + 1]
        for i in range(0, len(curve_data), segment_size)
    ]

def enforce_c0_continuity(control_points):
    for i in range(control_points.shape[2] - 1):  # Iterate over segments
        control_points[0, :, i + 1] = control_points[3, :, i]  # Match end to start
    return control_points

def enforce_c1_continuity(control_points):
    for i in range(control_points.shape[2] - 1):  # Iterate over segments
        # Compute the tangent at the end of the current segment
        tangent = control_points[3, :, i] - control_points[2, :, i]
           # Adjust the tangent of the next segment
        control_points[1, :, i + 1] = control_points[0, :, i + 1] + tangent
    return control_points

def enforce_continuity(control_points):
    control_points = enforce_c0_continuity(control_points)
    control_points = enforce_c1_continuity(control_points)
    return control_points

def get_control_tensor(curve_data, num_segments, method):
    control_tensor = np.zeros(shape=(4, 2, num_segments))
    for i in range(num_segments):
        control_tensor[:, :, i] = __fit_bezier(
            __split_data(curve_data, num_segments)[i], method=method
        )

    control_tensor = enforce_continuity(control_tensor)
    return control_tensor


def bezier_spline(control_tensor, p_per_seg):
    curve = cubic_bezier_curve(*control_tensor[:, :, 0], n_points=p_per_seg)
    for i in range(1, len(control_tensor) + 1):
        curve = np.append(
            curve,
            cubic_bezier_curve(*control_tensor[:, :, i], n_points=p_per_seg),
            axis=0,
        )
    return curve

if __name__ == "__main__":
    from parser.aerofoil import *
    import matplotlib.pyplot as plt
    from database.aerofoil_data import UIUC_DATABASE as UDB
    upper, lower = split_surfaces(UDB['ag45c03_dat'])
    num_segments = 5
    control = get_control_tensor(upper, num_segments, method="L-BFGS-B")
    bez = bezier_spline(control, 30)
    plt.plot(upper[:, 0], upper[:, 1], c="b", label="original")
    plt.plot(lower[:, 0], lower[:, 1], c="b", label="original")
    plt.plot(bez[:, 0], bez[:, 1], c="r", label="cubic spline")
    control = get_control_tensor(lower, num_segments, method="L-BFGS-B")
    bez = bezier_spline(control, 30)
    plt.plot(bez[:, 0], bez[:, 1], c="r", label="cubic spline")
    """for i in range(num_segments):
        seg = cubic_bezier_curve(*control[:, :, i], n_points=30)
        plt.plot(seg[:, 0], seg[:, 1], label=f"segment{i}")"""
    plt.xlim((-1,2))
    plt.ylim((-0.5,0.5))
    plt.show()    
