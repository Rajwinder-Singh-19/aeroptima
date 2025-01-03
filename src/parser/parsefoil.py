"""
parsers.parsefoil
================

Parsers for coordinate .dat files. Only selig and lednicer formats are supported
"""

import numpy as np
import os

__DAT_FILE_PATH: str = (
    os.getcwd() + "/UIUC_aerofoils/"
)  # aerofoils are stored in this folder


def getFormat(filename: str) -> str:
    """
    Returns the order style of the .dat file.

    PARAMETERS:

        `filename` -> Dat file name. Type(str)

    RETURNS:

        `selig` or `lednicer` -> file type. Type(str)
    """
    path = __DAT_FILE_PATH + filename
    with open(path, "r") as f:
        lines = f.readlines()

    if lines[2] == "\n":
        return "lednicer"
    else:
        return "selig"


def __selig2numpy(filename: str) -> np.ndarray:
    """
    Parses the coordinates from the .dat file into a numpy array.

    PARAMETERS:

        `filename` -> .dat file name. Coordinates in the file must be ordered in selig format. Type(str)

    RETURNS:

        `coords` -> Array of coordinates in the .dat file. Type(np.ndarray)
    """
    path = __DAT_FILE_PATH + filename
    with open(path, "r") as f:
        lines = f.readlines()

    coords = []

    for line in lines[1:]:
        try:
            x, y = map(float, line.strip().split())
            coords.append((x, y))
        except ValueError:
            continue

    coords = np.array(coords)
    return coords


def __lednicer2numpy(filename: str) -> np.ndarray:
    """
    Parses the coordinates from the .dat file into a numpy array.

    PARAMETERS:

        `filename` -> .dat file name. Coordinates in the file must be ordered in lednicer format. Type(str)

    RETURNS:

        `coords` -> Array of coordinates in the .dat file. Type(np.ndarray)
    """

    path = __DAT_FILE_PATH + filename
    with open(path, "r") as f:
        lines = f.readlines()

    coords = []

    for line in lines[3:]:
        try:
            x, y = map(float, line.strip().split())
            coords.append((x, y))
        except ValueError:
            continue

    coords = np.array(coords)
    return coords


def dat2numpy(filename: str) -> np.ndarray:
    """
    Parses the coordinates from the .dat file into a numpy array.

    PARAMETERS:

        `filename` -> .dat file name. Coordinates in the file must be ordered in either selig or lednicer format. Type(str)

    RETURNS:

        `__lednicer2numpy(filename)` or `__selig2numpy(filename)` -> Array of coordinates in the .dat file. Type(np.ndarray)
    """

    if getFormat(filename) == "lednicer":
        return __lednicer2numpy(filename)
    else:
        return __selig2numpy(filename)


def __selig_upper_lower(filename) -> tuple[np.ndarray]:
    """
    Parses the coordinates from the .dat file into a numpy array and divides them into upper and lower surface.

    PARAMETERS:

        `filename` -> .dat file name. Coordinates in the file must be ordered in selig format. Type(str)

    RETURNS:

        `upper, lower` -> Array of coordinates in the .dat file `upper` contains upper surface coordinates and `lower` contains lower surface coordinates. Type(tuple[np.ndarray])
    """

    coords = __selig2numpy(filename)
    x, y = coords[:, 0], coords[:, 1]
    mid_index = np.argmin(x)  # Leading edge is at x = 0
    upper_x, upper_y = x[0 : mid_index + 1], y[: mid_index + 1]
    lower_x, lower_y = x[mid_index:], y[mid_index:]
    upper = np.column_stack((np.flip(upper_x), np.flip(upper_y)))
    lower = np.column_stack((lower_x, lower_y))
    return upper, lower


def __lednicer_upper_lower(filename: str) -> tuple[np.ndarray]:
    """
    Parses the coordinates from the .dat file into a numpy array and divides them into upper and lower surface.

    PARAMETERS:

        `filename` -> .dat file name. Coordinates in the file must be ordered in lednicer format. Type(str)

    RETURNS:

        `upper, lower` -> Array of coordinates in the .dat file `upper` contains upper surface coordinates and `lower` contains lower surface coordinates. Type(tuple[np.ndarray])
    """

    coords = __lednicer2numpy(filename)
    x, y = coords[:, 0], coords[:, 1]
    mid_index = np.argmax(x)
    upper_x, upper_y = x[: mid_index + 1], y[: mid_index + 1]
    lower_x, lower_y = x[mid_index + 1 :], y[mid_index + 1 :]
    upper = np.column_stack((upper_x, upper_y))
    lower = np.column_stack((lower_x, lower_y))
    return upper, lower


def split_surfaces(filename: str) -> tuple[np.ndarray]:
    """
    Parses the coordinates from the .dat file into a numpy array and divides them into upper and lower surface.

    PARAMETERS:

        `filename` -> .dat file name. Coordinates in the file must be ordered in either lednicer or selig format. Type(str)

    RETURNS:

        `__lednicer_upper_lower(filename)` or `__selig_upper_lower(filename)` -> Array of coordinates in the .dat file `upper` contains upper surface coordinates and `lower` contains lower surface coordinates. Type(tuple[np.ndarray])
    """

    if getFormat(filename) == "lednicer":
        return __lednicer_upper_lower(filename)
    else:
        return __selig_upper_lower(filename)
