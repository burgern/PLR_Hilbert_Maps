from abc import ABC, abstractmethod
from .cell import *
import numpy as np
from typing import List, Union


class MapManager(ABC):
    """
    Map Manager
    TODO Description
    """
    def __init__(self, cell_template: Cell, x_neighbour_dist: float, y_neighbour_dist: float):
        # get params
        self.cell_template = cell_template
        self.x_neighbour_dist = x_neighbour_dist
        self.y_neighbour_dist = y_neighbour_dist

        self.map_indices = np.empty((2, 0))
        self.x_min, self.x_max = 0, 0
        self.y_min, self.y_max = 0, 0

    @abstractmethod
    def update(self, points: np.array) -> List[Cell]:
        """ compute required local hilbert maps for map given new points """
        raise NotImplementedError

    def update_intervals(self, x_min, x_max, y_min, y_max):
        # TODO if x_min and y_min are more than 0, 0 we currently plot too much
        if x_min < self.x_min:
            self.x_min = x_min
        if x_max > self.x_max:
            self.x_max = x_max
        if y_min < self.y_min:
            self.y_min = y_min
        if y_max > self.y_max:
            self.y_max = y_max


class GridMap(MapManager):
    """
    Grid Map
    TODO Description
    Inputs:
        cell_template:              standard cell which will be added
        x_neighbour_dist:           cell neighbour distance from center to center in x-direction
        y_neighbour_dist:           cell neighbour distance from center to center in y-direction
    """
    def __init__(self, cell_template: Union[Cell, Rectangle], x_neighbour_dist: float, y_neighbour_dist: float):
        super(GridMap, self).__init__(cell_template, x_neighbour_dist, y_neighbour_dist)

    def update(self, points: np.array) -> List[Cell]:
        # cells for all points
        new_cells = []
        indices = np.concatenate((np.expand_dims(points[0, :] // self.x_neighbour_dist, axis=0),
                                  np.expand_dims(points[1, :] // self.y_neighbour_dist, axis=0)), axis=0)
        indices = np.unique(indices, axis=1)

        # compare with already existing cells (keep only new cells)
        new_indices = ~((self.map_indices[:, None] == indices[..., None]).all(0).any(1))

        # only proceed if there are new indices
        if new_indices.any():
            indices = indices[:, new_indices]
            self.map_indices = np.hstack((self.map_indices, indices))

            # new cells
            for center in indices.T:
                new_cells.append(self.cell_template.new_cell((center[0] * self.x_neighbour_dist + self.x_neighbour_dist / 2,
                                                              center[1] * self.y_neighbour_dist + self.y_neighbour_dist / 2)))

            # update intervals
            x_min, x_max = np.min(indices[0, :]) * self.x_neighbour_dist, (np.max(indices[0, :]) + 1) * self.x_neighbour_dist
            y_min, y_max = np.min(indices[1, :]) * self.y_neighbour_dist, (np.max(indices[1, :]) + 1) * self.y_neighbour_dist
            self.update_intervals(x_min, x_max, y_min, y_max)

        return new_cells
