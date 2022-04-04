from . import Cell
from typing import Tuple
import numpy as np
import matplotlib.patches as patches


class Rectangle(Cell):
    """
    Rectangle
    TODO Description
    """
    def __init__(self, center: Tuple[float, float], width: float, length: float, nx: float, ny: float):
        r1 = (width / 2, 0)
        r2 = (0, length / 2)
        super().__init__(center, r1, r2, nx, ny)

    def is_point_in_cell(self, points: np.array) -> bool:
        points_norm_abs = np.absolute(self.original_to_normalized(points))
        mask = ((points_norm_abs[0, :]) <= self.nx) & (points_norm_abs[1, :] <= self.ny)
        return mask

    def patch(self) -> patches:
        bottom_left = self.center - self.r1 - self.r2
        patch = patches.Rectangle((bottom_left[0], bottom_left[1]), 2 * self.r1_mag, 2 * self.r2_mag,
                                  edgecolor=self.patch_edgecolor, linewidth=self.patch_linewidth, facecolor='none')
        return patch
