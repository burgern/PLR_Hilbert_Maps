from . import Cell
from typing import Tuple
import numpy as np
from PLR_Hilbert_Maps.utils import concatenate_ones


class Square(Cell):
    """
    Square
    TODO Description
    """
    def __init__(self, center: Tuple[float, float], width: float, nx: float, ny: float):
        r1 = (width / 2, 0)
        r2 = (0, width / 2)
        super().__init__(center, r1, r2, nx, ny)

    def is_point_in_cell(self, points: np.array) -> bool:
        points_norm_abs = np.absolute(self.original_to_normalized(points))
        mask = ((points_norm_abs[0, :]) <= self.nx) & (points_norm_abs[1, :] <= self.ny)
        return mask
