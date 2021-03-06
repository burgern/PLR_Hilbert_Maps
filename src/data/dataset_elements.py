from src.hilbert_map import Square, Rectangle, Circle, Ellipsoid, Hexagon
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple


def load_from_config(config: Dict):
    n = config["points"]
    size = config["size"]
    center = (config["center_x"], config["center_y"])
    updates = config["updates"]
    return n, size, center, updates


class DatasetHexagon:
    """"
    DatasetHexagon
    size [m] x size [m] area with elements
    """
    def __init__(self, config: Optional[Dict], n: int = 10000, size: int = 5,
                 center: Tuple[float, float] = (0, 0),
                 patch_edgecolor: str = "r", patch_linewidth: int = 1,
                 occlusion_zone=0.3, nx=0.5, ny=0.5, updates=20):
        # input parameters
        if config is not None:
            config_dummy = config["dataset"]["dummy"]
            n, size, center, updates = \
                load_from_config(config_dummy)
        self.config = config
        self.n = n
        self.size = size
        self.center = center
        self.occlusion_zone = occlusion_zone
        self.nx = nx
        self.ny = ny
        self.updates = updates

        # update steps counter
        self.curr_update = -1

        # initialize points
        self.points = np.random.uniform(-self.size/2, self.size/2, (2, self.n)) + np.array([[center[0]], [center[1]]])
        self.reflectance = np.random.rand(self.n)  # set non-occupied to np.nan

        # create wall
        wall_out = Hexagon(center=self.center, width=self.size,
                           length=self.size, nx=self.nx, ny=self.ny,
                           patch_edgecolor=patch_edgecolor,
                           patch_linewidth=patch_linewidth)
        wall_in = Hexagon(center=self.center,
                          width=self.size-self.occlusion_zone,
                          length=self.size-self.occlusion_zone,
                          nx=self.nx, ny=self.ny,
                          patch_edgecolor=patch_edgecolor,
                          patch_linewidth=patch_linewidth)
        self.points = self.points[:, wall_out.is_point_in_cell(self.points)]
        self.occupancy = ~wall_in.is_point_in_cell(self.points)

        # create hexagon object
        hex_out = Hexagon(center=self.center, width=self.size/4,
                          length=self.size/4, nx=self.nx, ny=self.ny,
                          patch_edgecolor=patch_edgecolor,
                          patch_linewidth=patch_linewidth)
        hex_in = Hexagon(center=self.center,
                         width=self.size/4 - self.occlusion_zone,
                         length=self.size/4 - self.occlusion_zone, nx=self.nx,
                         ny=self.ny, patch_edgecolor=patch_edgecolor,
                         patch_linewidth=patch_linewidth)
        hex_in_mask = hex_in.is_point_in_cell(self.points)
        self.points = self.points[:, ~hex_in_mask]
        self.occupancy = self.occupancy[~hex_in_mask]
        self.occupancy = self.occupancy | hex_out.is_point_in_cell(self.points)

        # create circle object
        circle_out = Circle(center=(self.center[0] + self.size / 4,
                                    self.center[1]), radius=self.size / 12,
                            nx=self.nx, ny=self.ny,
                            patch_edgecolor=patch_edgecolor,
                            patch_linewidth=patch_linewidth)
        circle_in = Circle(center=(self.center[0] + self.size / 4, self.center[1]),
                           radius=self.size / 12 - self.occlusion_zone / 2,
                           nx=self.nx, ny=self.ny,
                           patch_edgecolor=patch_edgecolor,
                           patch_linewidth=patch_linewidth)
        circle_in_mask = circle_in.is_point_in_cell(self.points)
        self.points = self.points[:, ~circle_in_mask]
        self.occupancy = self.occupancy[~circle_in_mask]
        self.occupancy = self.occupancy | circle_out.is_point_in_cell(self.points)

        # create ellipsoid object
        ell_out = Ellipsoid(center=(self.center[0],
                                    self.center[1] + self.size / 4), angle=30,
                            radius_primary=self.size/8,
                            radius_secondary=self.size/16, nx=self.nx,
                            ny=self.ny, patch_edgecolor=patch_edgecolor,
                            patch_linewidth=patch_linewidth)
        ell_in = Ellipsoid(center=(self.center[0],
                                   self.center[1] + self.size / 4), angle=30,
                           radius_primary=self.size/8 - self.occlusion_zone / 2,
                           radius_secondary=self.size/16 -
                                            self.occlusion_zone / 2,
                           nx=self.nx, ny=self.ny,
                           patch_edgecolor=patch_edgecolor,
                           patch_linewidth=patch_linewidth)
        ell_in_mask = ell_in.is_point_in_cell(self.points)
        self.points = self.points[:, ~ell_in_mask]
        self.occupancy = self.occupancy[~ell_in_mask]
        self.occupancy = self.occupancy | ell_out.is_point_in_cell(self.points)

        # create square object
        squ_out = Square(center=(self.center[0] - self.size / 4,
                                 self.center[1]), width=self.size/8, nx=self.nx,
                         ny=self.ny, patch_edgecolor=patch_edgecolor,
                         patch_linewidth=patch_linewidth)
        squ_in = Square(center=(self.center[0] - self.size / 4, self.center[1]),
                        width=self.size/8 - self.occlusion_zone, nx=self.nx,
                        ny=self.ny, patch_edgecolor=patch_edgecolor,
                        patch_linewidth=patch_linewidth)
        squ_in_mask = squ_in.is_point_in_cell(self.points)
        self.points = self.points[:, ~squ_in_mask]
        self.occupancy = self.occupancy[~squ_in_mask]
        self.occupancy = self.occupancy | squ_out.is_point_in_cell(self.points)

        # create rectangle object
        rec_out = Rectangle(center=(self.center[0],
                                    self.center[1] - self.size / 4),
                            width=self.size / 4, length=self.size / 8,
                            nx=self.nx, ny=self.ny,
                            patch_edgecolor=patch_edgecolor,
                            patch_linewidth=patch_linewidth)
        rec_in = Rectangle(center=(self.center[0],
                                   self.center[1] - self.size / 4),
                           width=self.size / 4 - self.occlusion_zone,
                           length=self.size / 8 - self.occlusion_zone,
                           nx=self.nx, ny=self.ny,
                           patch_edgecolor=patch_edgecolor,
                           patch_linewidth=patch_linewidth)
        rec_in_mask = rec_in.is_point_in_cell(self.points)
        self.points = self.points[:, ~rec_in_mask]
        self.occupancy = self.occupancy[~rec_in_mask]
        self.occupancy = self.occupancy | rec_out.is_point_in_cell(self.points)

    def __iter__(self):
        return self

    def __next__(self):
        curr_update = self.curr_update + 1
        self.__init__(config=self.config, n=self.n, size=self.size,
                      center=self.center, occlusion_zone=self.occlusion_zone,
                      nx=self.nx, ny=self.ny, updates=self.updates)
        self.curr_update = curr_update
        if curr_update >= self.updates:
            raise StopIteration
        return self.points, self.occupancy, self.reflectance

    def visualize(self, ax: Axes, step_size: int):
        ax.scatter(self.points[0, :], self.points[1, :], c=self.occupancy, s=1)
        plt.xlim(self.center[0] - self.size / 2, self.center[0] + self.size / 2)
        plt.ylim(self.center[1] - self.size / 2, self.center[1] + self.size / 2)
        ax.axis('scaled')

    def concatenated(self):
        return self.points, self.occupancy
