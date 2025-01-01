"""
database.basis_matrices
=======================

Basis matrices to generate a different types of parametrized curves
"""

import numpy as np

BEZIER_MATRIX = np.array(([1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]))
"""
Chracteristic matrix of a cubic bezier curve. Needs 4 control points to generate a curve.
"""
