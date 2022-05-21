from .cell import Cell
from typing import Tuple
import numpy as np
import matplotlib.patches as patches
from typing import Optional


class Circle(Cell):
    """
    Circle
    TODO Description
    """
    def __init__(self, center: Optional[Tuple[float, float]], radius: float,
                 nx: float, ny: float, patch_edgecolor: str,
                 patch_linewidth: float):
        r1 = (radius, 0)
        r2 = (0, radius)
        super().__init__(center, r1, r2, nx, ny, patch_edgecolor,
                         patch_linewidth)

    def is_point_in_cell(self, points: np.array) -> bool:
        points_norm = self.original_to_normalized(points)

        # ellipsoid equation: x^2 / a + y^2 / b + z^2 / c = 1
        ellipsoid_eq = np.square(points_norm[0, :]) / (self.nx ** 2) \
                       + np.square(points_norm[1, :]) / (self.ny ** 2)
        mask = ellipsoid_eq <= 1
        return mask

    def patch(self) -> patches:
        patch = patches.Ellipse((self.center[0], self.center[1]),
                                width=self.r1_mag, height=self.r2_mag, angle=0,
                                edgecolor=self.patch_edgecolor,
                                linewidth=self.patch_linewidth,
                                facecolor='none')
        return patch

