from . import Cell
from typing import Tuple
import numpy as np
from PLR_Hilbert_Maps.utils import concatenate_ones


class Ellipsoid(Cell):
    """
    Ellipsoid
    TODO Description
    """
    def __init__(self, center: Tuple[float, float], r1: Tuple[float, float], r2: Tuple[float, float], nx: float,
                 ny: float):
        super().__init__(center, r1, r2, nx, ny)

    def is_point_in_cell(self, points: np.array) -> bool:
        points_ = concatenate_ones(points, 0)
        points_norm = self.original_to_normalized(points_)

        # ellipsoid equation: x^2 / a + y^2 / b + z^2 / c = 1
        ellipsoid_eq = np.square(points_norm[0, :]) / (self.nx ** 2) \
                       + np.square(points_norm[1, :]) / (self.ny ** 2)
        mask = ellipsoid_eq <= 1
        return mask
