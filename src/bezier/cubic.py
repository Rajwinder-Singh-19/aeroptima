"""
bezier.cubic
============

Interpolating a set of points using a cubic bezier curve
"""

import numpy as np
from database.basis_matrices import *  # contains the characteristic matrices for curve interpolation


def __create_point_matrix(*points: list[np.ndarray]) -> np.ndarray:
    """
    Creates a matrix of control points which can be multiplied with the cached characteristic matrix

        PARAMETERS:

            `*points` -> List of control points (IN ORDER). Type(list[np.ndarray])

        RETURNS:

            `np.vstack((points))` -> Vertically stacked control points (IN ORDER). Type(np.ndarray)
    """
    return np.vstack((points))


def __curve_point(
    t: float, matrix: np.ndarray, control_points: np.ndarray
) -> np.ndarray:
    """
    Outputs a single point of the curve corresponding to the t parameter.

        PARAMETERS:

            `t` -> t parameter of the curve ranging from 0 to 1. Type(float)

            `matrix` -> characteristic symmetric matrix of the curve. Type(np.ndarray)

            `control_points` -> array of control points of the curve in a single array. Type(np.ndarray)

        RETURNS:

            `point` -> Point on the curve that corresponds to the input t value. Type(np.ndarray)
    """
    if len(matrix) != len(matrix[0]):  # characteristic matrix is not valid
        raise TypeError("Only square matrices are allowed as the matrix inputs")
    if len(matrix) != len(
        control_points
    ):  # Cannot multiply characteristic and control point matrix
        raise TypeError(
            "The number of columns in matrix is not equal to the number of control points"
        )

    control_product = np.zeros_like(
        control_points
    )  # this variable will hold the product of characteristic matrix and control matrix

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


def cubic_bezier_curve(*control_points: list[np.ndarray], n_points: int) -> np.ndarray:
    """
    Creates a bezier curve with a certain number of points

        PARAMETERS:

            `*control_points` -> List of control points defining the curve. Should have 4 elements. Type(list[np.ndarray])

            `n_points` -> number of curve points to output. Type(int)

        RETURNS:

            `np.array([__curve_point(ti, BEZIER_MATRIX, control_points) for ti in t])` -> Array of coordinates for the cubic bezier curve. Type(np.ndarray)
    """
    control_points = __create_point_matrix(control_points)
    if len(control_points) != 4:
        raise ValueError("Cubic bezier curve requires 4 control points as input")
    t = np.linspace(0, 1, n_points)
    return np.array([__curve_point(ti, BEZIER_MATRIX, control_points) for ti in t])
