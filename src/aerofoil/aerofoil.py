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
    from database.aerofoil_data import UIUC_DATABASE as UDB
    a18 = Aerofoil(UDB['apex16_NASA_CR_201062_dat'], 5, 'L-BFGS-B')
    a18.upper_control[2,1,0] = 1.1*a18.upper_control[2,1,0]
    a18.upper_control = enforce_continuity(a18.upper_control)
    a18.upper_control[2,1,2] = 1.4*a18.upper_control[2,1,2]
    a18.upper_control = enforce_continuity(a18.upper_control)
    upper = a18.getUpperCurve(20)
    lower = a18.getLowerCurve(20)
    
    import matplotlib.pyplot as plt
    plt.plot(upper[:,0], upper[:,1])
    plt.plot(lower[:,0], lower[:,1])
    plt.scatter(a18.upper_control[:,0], a18.upper_control[:,1])
    plt.xlim((-1, 2))
    plt.ylim((-1,1))
    plt.show()
