import numpy as np
from dataclasses import dataclass
from parser.aerofoil import *
from bezier.cubic_spline import *
from database.aerofoil_data import FileDict
@dataclass
class Aerofoil:
    database_index: FileDict 
    upper_coords: np.array
    lower_coords: np.array
    upper_control: np.array
    lower_control: np.array
    n_segments: int
    points_per_seg: int

    def __init__(self, database_index, n_segments, method):
        self.database_index = database_index
        self.n_segments = n_segments
        self.upper_coords, self.lower_coords = split_surfaces(self.database_index)
        self.upper_control = get_control_tensor(self.upper_coords, self.n_segments, method)
        self.lower_control = get_control_tensor(self.lower_coords, self.n_segments, method)

    def getUpperCurve(self, points_per_seg):
        self.points_per_seg = points_per_seg
        return bezier_spline(self.upper_control, self.points_per_seg)
    
    def getLowerCurve(self, points_per_seg):
        self.points_per_seg = points_per_seg
        return bezier_spline(self.lower_control, self.points_per_seg)



if __name__=="__main__":
    pass
