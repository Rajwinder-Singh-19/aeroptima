import numpy as np
import os

DAT_FILE_PATH = os.getcwd() + "/aerofoil_data/"

# __all__ = ['getFormat', 'dat2numpy', 'split_surfaces']


def getFormat(filename):
    path = DAT_FILE_PATH + filename
    with open(path, "r") as f:
        lines = f.readlines()

    if lines[2] == "\n":
        return "lednicer"
    else:
        return "selig"


def __selig2numpy(filename: str):
    path = DAT_FILE_PATH + filename
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


def __lednicer2numpy(filename: str):
    path = DAT_FILE_PATH + filename
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


def dat2numpy(filename: str):
    if getFormat(filename) == "lednicer":
        return __lednicer2numpy(filename)
    else:
        return __selig2numpy(filename)


def __selig_upper_lower(filename):
    coords = __selig2numpy(filename)
    x, y = coords[:, 0], coords[:, 1]
    mid_index = np.argmin(x)  # Leading edge is at x = 0
    upper_x, upper_y = x[0 : mid_index + 1], y[: mid_index + 1]
    lower_x, lower_y = x[mid_index:], y[mid_index:]
    upper = np.column_stack((np.flip(upper_x), np.flip(upper_y)))
    lower = np.column_stack((lower_x, lower_y))
    return upper, lower


def __lednicer_upper_lower(filename):
    coords = __lednicer2numpy(filename)
    x, y = coords[:, 0], coords[:, 1]
    mid_index = np.argmax(x)
    upper_x, upper_y = x[: mid_index + 1], y[: mid_index + 1]
    lower_x, lower_y = x[mid_index + 1 :], y[mid_index + 1 :]
    upper = np.column_stack((upper_x, upper_y))
    lower = np.column_stack((lower_x, lower_y))
    return upper, lower


def split_surfaces(filename):
    if getFormat(filename) == "lednicer":
        return __lednicer_upper_lower(filename)
    else:
        return __selig_upper_lower(filename)
