"""
classes.bezierfoil
================

Using cubic bezier splines to model aerofoil shapes
"""

import numpy as np
from parser.parsefoil import split_surfaces
from bezier.spline import get_control_tensor, bezier_spline
from database.UIUC_aerofoils import UIUCDict
import os


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
        self,
        database_index: UIUCDict,
        n_segments: int,
        pca_components: str,
        mean_pca_foil: str,
        method: str = "L-BFGS-B",
    ) -> None:
        """
        Default constructor for Aerofoil class.

        PARAMETERS:

            `database_index` -> Tells which aerofoil to model. Type(str) but recommended to be used with a UIUCDict value.

            `n_segments` -> Number of cubic bezier segments. Type(int).

            `pca_components` -> PCA coefficients npy file for perturbation. Type(str).

            `mean_pca_foil` -> Mean npy file of the perturbing dataset. Type(str).

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
        self.pca_components = np.load(
            os.getcwd() + "/src/database/PCA_files/" + f"{pca_components}"
        )
        self.mean_airfoil = np.load(os.getcwd() + "/src/database/PCA_files/" + f"{mean_pca_foil}")

    def perturb_pca(self, coefficients, freeze_last_n=1):

        # Flatten current shape
        current_shape = np.concatenate(
            [self.upper_control.flatten(), self.lower_control.flatten()]
        )

        if current_shape.shape[0] != self.mean_airfoil.shape[0]:
            raise ValueError(
                f"Shape mismatch: {current_shape.shape} vs {self.mean_airfoil.shape}"
            )

        # Apply PCA perturbations
        perturbation = sum(
            coeff * pca_comp
            for coeff, pca_comp in zip(coefficients, self.pca_components)
        )

        # Prevent modification of the last 'freeze_last_n' segments
        if freeze_last_n > 0:
            perturbation[-2 * freeze_last_n :] = (
                0  # Ensuring trailing edge points remain unchanged
            )

        # Apply perturbation
        current_shape += perturbation

        # Split back into upper and lower surfaces
        split_idx = self.upper_control.size
        perturbed_upper = current_shape[:split_idx].reshape(self.upper_control.shape)
        perturbed_lower = current_shape[split_idx:].reshape(self.lower_control.shape)

        # Ensure lower surface does not exceed upper surface at each x
        perturbed_lower = np.minimum(perturbed_lower, perturbed_upper - 1e-5)

        # Assign modified controls
        self.upper_control = perturbed_upper
        self.lower_control = perturbed_lower

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

    def getUpperCurve(self, points_per_seg: int) -> np.ndarray:
        """
        To generate the upper surface cubic bezier spline coordinates

        PARAMETERS:

            `points_per_seg` -> Number of points in each cubic bezier segment. Type(int).

        RETURNS:

            `bezier_spline(self.upper_control, points_per_seg)` -> Upper surface (x,y) coordinates calculated
            from cubic spline interpolation. Type(np.ndarray).
        """
        return bezier_spline(self.upper_control, points_per_seg)

    def getLowerCurve(self, points_per_seg: int) -> np.ndarray:
        """
        To generate the lower surface cubic bezier spline coordinates

        PARAMETERS:

            `points_per_seg` -> Number of points in each cubic bezier segment. Type(int).

        RETURNS:

            `bezier_spline(self.lower_control, points_per_seg)` -> Lower surface (x,y) coordinates calculated
            from cubic spline interpolation. Type(np.ndarray).
        """
        return bezier_spline(self.lower_control, points_per_seg)

    def save_foil(
        self,
        aerofoil_header_name: str,
        save_folder: str,
        save_filename: str,
        points_per_seg: int,
        write_precision: int,
    ) -> None:
        """
        To save the cubic bezier interpolated aerofoil in a .dat file

        PARAMETERS:
            `aerofoil_header_name` -> The header of the saved file. Typically aerofoil name. Type(str)

            `save_folder` -> The folder inn the current working directory where the file will be saved. Type(str)

            `save_filename` -> Name of the .dat file to be saved. Type(str)

            `write_precision` -> The floating point precision of the co-ordinates in the saved file. Type(int)

            `points_per_seg` -> Number of points in each cubic bezier segment. Type(int).

        RETURNS:

            None

        """
        upper = self.getUpperCurve(points_per_seg)
        upper = np.flip(upper, axis=0)
        lower = self.getLowerCurve(points_per_seg)

        selig_coords = np.vstack((upper, lower))

        import os

        FOLDER_PATH = os.getcwd() + "/" + save_folder
        FILE_PATH = FOLDER_PATH + "/" + save_filename
        if not os.path.exists(FOLDER_PATH):
            os.mkdir(FOLDER_PATH)
            print(f"{save_folder} created in the current working directory")

        with open(FILE_PATH, "w") as f:
            f.write(f"{aerofoil_header_name}" + "\n")

            for x, y in selig_coords:
                f.write(f"{x:.{write_precision}f} {y:.{write_precision}f}\n")

        print(f"{aerofoil_header_name} saved in {FILE_PATH}")
