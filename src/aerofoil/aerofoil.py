import numpy as np
from dataclasses import dataclass
@dataclass
class Aerofoil:
    coord_data: np.array
    upper_control: np.array
    lower_control: np.array
    pass