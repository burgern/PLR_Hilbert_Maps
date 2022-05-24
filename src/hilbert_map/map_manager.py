from abc import ABC, abstractmethod
from .cell import *
import numpy as np
from typing import List, Union


class MapManager(ABC):
    """
    Map Manager
    TODO Description
    """
    def __init__(self, cell_template: Cell, x_neighbour_dist: float,
                 y_neighbour_dist: float):
        # get params
        self.cell_template = cell_template
        self.x_neighbour_dist = x_neighbour_dist
        self.y_neighbour_dist = y_neighbour_dist

        self.map_indices = np.empty((2, 0))
        self.x_min, self.x_max = 0, 0
        self.y_min, self.y_max = 0, 0
        self.first_iteration = True

    @abstractmethod
    def update(self, points: np.array) -> List[Cell]:
        """ compute required local hilbert maps for map given new points """
        raise NotImplementedError

    def update_intervals(self, x_min, x_max, y_min, y_max):
        if self.first_iteration:
            self.x_min = x_min - self.cell_template.r1_mag
            self.x_max = x_max + self.cell_template.r1_mag
            self.y_min = y_min - self.cell_template.r2_mag
            self.y_max = y_max + self.cell_template.r2_mag
            self.first_iteration = False
        else:
            if x_min < self.x_min:
                self.x_min = x_min - self.cell_template.r1_mag
            if x_max > self.x_max:
                self.x_max = x_max + self.cell_template.r1_mag
            if y_min < self.y_min:
                self.y_min = y_min - self.cell_template.r2_mag
            if y_max > self.y_max:
                self.y_max = y_max + self.cell_template.r2_mag


class GridMap(MapManager):
    """
    Grid Map
    TODO Description
    :param cell_template standard cell which will be added
    :param x_neighbour_dist cell neighbour dist from center to center in x-dir
    :param y_neighbour_dist cell neighbour dist from center to center in y-dir
    """
    def __init__(self, cell_template: Union[Cell, Rectangle],
                 x_neighbour_dist: float, y_neighbour_dist: float):
        super(GridMap, self).__init__(cell_template, x_neighbour_dist,
                                      y_neighbour_dist)

    def update(self, points: np.array) -> List[Cell]:
        # cells for all points
        new_cells = []
        x_ind = np.expand_dims(points[0, :] // self.x_neighbour_dist, axis=0)
        y_ind = np.expand_dims(points[1, :] // self.y_neighbour_dist, axis=0)
        indices = np.concatenate((x_ind, y_ind), axis=0)
        indices = np.unique(indices, axis=1)

        # compare with already existing cells (keep only new cells)
        new_indices = ~((self.map_indices[:, None] ==
                         indices[..., None]).all(0).any(1))

        # only proceed if there are new indices
        if new_indices.any():
            indices = indices[:, new_indices]
            self.map_indices = np.hstack((self.map_indices, indices))

            # new cells
            for index in indices.T:
                center_x = index[0] * self.x_neighbour_dist + \
                           self.x_neighbour_dist / 2
                center_y = index[1] * self.y_neighbour_dist + \
                           self.y_neighbour_dist / 2
                new_cells.append(self.cell_template.new_cell((center_x,
                                                              center_y)))

            # update intervals
            x_min, x_max = np.min(indices[0, :]) * self.x_neighbour_dist,\
                           (np.max(indices[0, :]) + 1) * self.x_neighbour_dist
            y_min, y_max = np.min(indices[1, :]) * self.y_neighbour_dist,\
                           (np.max(indices[1, :]) + 1) * self.y_neighbour_dist
            self.update_intervals(x_min, x_max, y_min, y_max)

        return new_cells
