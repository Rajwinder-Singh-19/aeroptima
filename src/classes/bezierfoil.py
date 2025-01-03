"""
classes.bezierfoil
================

Using cubic bezier splines to model aerofoil shapes
"""

import numpy as np
from parser.parsefoil import split_surfaces
from bezier.spline import get_control_tensor, bezier_spline
from database.UIUC_aerofoils import UIUCDict


class BezierFoil:
    """
    Objects of this class model an aerofoil.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Reads a dat file of the coordinates and interpolates it using a cubic bezier spline

    ATTRIBUTES

        `database_index` -> UIUC database dictionary. Type(UIUCDict).

        `upper_coords` -> upper surface coordinates from the dat file. Type(np.ndarray).

        `lower_coords` -> lower surface coordinates from the dat file. Type(np.ndarray).

        `upper_control` -> upper surface cubic spline control points in the form of a 4 x 2 x n_segments matrix. Type(np.ndarray).

        `lower_control` -> lower surface cubic spline control points in the form of a 4 x 2 x n_segments matrix. Type(np.ndarray).

        `n_segments` -> number of bezier curve segments in both upper and lower surfaces. Type(int).

    """

    database_index: UIUCDict
    upper_coords: np.ndarray
    lower_coords: np.ndarray
    upper_control: np.ndarray
    lower_control: np.ndarray
    n_segments: int

    def __init__(
        self, database_index: UIUCDict, n_segments: int, method: str = "L-BFGS-B"
    ) -> None:
        """
        Default constructor for Aerofoil class.

        PARAMETERS:

            `database_index` -> Tells which aerofoil to model. Type(str) but recommended to be used with a UIUCDict value.

            `n_segments` -> Number of cubic bezier segments. Type(int).

            `method` -> The optimization solver to converge the bezier control points,
                      default is Low Memory Broyden-Fletcher-Goldfarb-Shanno solver.
        RETURNS:

            None

        """
        self.database_index = database_index
        self.n_segments = n_segments
        self.upper_coords, self.lower_coords = split_surfaces(self.database_index)
        self.upper_control = get_control_tensor(
            self.upper_coords, self.n_segments, method
        )
        self.lower_control = get_control_tensor(
            self.lower_coords, self.n_segments, method
        )

    def close_curve(self) -> None:
        """
        Forces the leading and trailing edge of the aerofoil cubic spline to be closed.
        It sets the first control point of the first segment of upper and lower beziers to be (0,0),
        and sets the last control point of the last segment of upper and lower beziers to be (1,0).

        PARAMETERS:

            None

        RETURNS:

            None
        """
        self.lower_control[0, :, 0] = self.upper_control[0, :, 0] = np.array([0, 0])
        self.lower_control[-1, :, -1] = self.upper_control[-1, :, -1] = np.array([1, 0])

    def getUpperCurve(self, points_per_seg: int) -> np.array:
        """
        To generate the upper surface cubic bezier spline coordinates

        PARAMETERS:

            `points_per_seg` -> Number of points in each cubic bezier segment. Type(int).

        RETURNS:

            `bezier_spline(self.upper_control, points_per_seg)` -> Upper surface (x,y) coordinates calculated
            from cubic spline interpolation. Type(np.array).
        """
        return bezier_spline(self.upper_control, points_per_seg)

    def getLowerCurve(self, points_per_seg: int) -> np.array:
        """
        To generate the lower surface cubic bezier spline coordinates

        PARAMETERS:

            `points_per_seg` -> Number of points in each cubic bezier segment. Type(int).

        RETURNS:

            `bezier_spline(self.lower_control, points_per_seg)` -> Lower surface (x,y) coordinates calculated
            from cubic spline interpolation. Type(np.array).
        """
        return bezier_spline(self.lower_control, points_per_seg)
