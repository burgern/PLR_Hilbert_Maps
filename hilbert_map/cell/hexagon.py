from . import Cell
from typing import Tuple
import numpy as np
from PLR_Hilbert_Maps.utils import concatenate_ones


class Hexagon(Cell):
    """
    Hexagon
    TODO Description
    """
    def __init__(self, center: Tuple[float, float], width: float, length: float, nx: float, ny: float):
        r1 = (width / 2, 0)
        r2 = (0, length / 2)
        super().__init__(center, r1, r2, nx, ny)

    def is_point_in_cell(self, points: np.array) -> bool:
        points_ = concatenate_ones(points, 0)
        points_norm_abs = np.absolute(self.original_to_normalized(points_))

        # hexagon equation: x + y * nx / (2 * ny) <= nx
        hexagon_eq = points_norm_abs[0, :] + points_norm_abs[1, :] * self.nx / (2 * self.ny)
        mask = (points_norm_abs[0, :] <= self.nx) & (points_norm_abs[1, :] <= self.ny) & (hexagon_eq <= self.nx)
        return mask
