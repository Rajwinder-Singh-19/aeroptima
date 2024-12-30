import numpy as np
from database.cached_matrices import *


def __create_point_matrix(*points):
    return np.vstack((points))


def __curve_point(t, matrix, control_points):
    if len(matrix) != len(matrix[0]):
        raise TypeError("Only square matrices are allowed as the matrix inputs")
    if len(matrix) != len(control_points):
        raise TypeError(
            "The number of columns in matrix is not equal to the number of control points"
        )

    control_product = np.zeros_like(control_points)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            control_product[i] += matrix[i, j] * control_points[j]

    t_vector = np.zeros(len(matrix))
    for i in range(len(matrix)):
        t_vector[i] = t**i

    x_sum = 0
    y_sum = 0
    for i in range(len(t_vector)):
        x_sum += t_vector[i] * control_product[i, 0]
        y_sum += t_vector[i] * control_product[i, 1]

    point = np.array([x_sum, y_sum])
    return point


def cubic_bezier_curve(*control_points, n_points):
    control_points = __create_point_matrix(control_points)
    if len(control_points) != 4:
        raise ValueError("Cubic bezier curve requires 4 control points as input")
    t = np.linspace(0, 1, n_points)
    return np.array([__curve_point(ti, BEZIER_MATRIX, control_points) for ti in t])
