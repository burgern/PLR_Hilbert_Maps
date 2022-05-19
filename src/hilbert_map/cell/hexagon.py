from . import Cell
from typing import Tuple
import numpy as np
import matplotlib.patches as patches
from typing import Optional


class Hexagon(Cell):
    """
    Hexagon
    TODO Description
    """
    def __init__(self, center: Optional[Tuple[float, float]], width: float,
                 length: float, nx: float, ny: float):
        r1 = (width / 2, 0)
        r2 = (0, length / 2)
        super().__init__(center, r1, r2, nx, ny)

        r1, r2 = np.expand_dims(self.r1, axis=1), np.expand_dims(self.r2, axis=1)
        self.edges = np.concatenate((r1,
                                     r1 / 2 + r2,
                                     - r1 / 2 + r2,
                                     - r1,
                                     - r1 / 2 - r2,
                                     r1 / 2, - r2,
                                     r1), axis=1)

    def is_point_in_cell(self, points: np.array) -> bool:
        points_norm_abs = np.absolute(self.original_to_normalized(points))

        # hexagon equation: x + y * nx / (2 * ny) <= nx
        hexagon_eq = points_norm_abs[0, :] + points_norm_abs[1, :] * \
                     self.nx / (2 * self.ny)
        mask = (points_norm_abs[0, :] <= self.nx) & \
               (points_norm_abs[1, :] <= self.ny) & (hexagon_eq <= self.nx)
        return mask

    def patch(self) -> patches:
        patch = patches.Rectangle(self.edges.T, closed=True,
                                  edgecolor=self.patch_edgecolor,
                                  linewidth=self.patch_linewidth,
                                  facecolor='none')
        return patch
