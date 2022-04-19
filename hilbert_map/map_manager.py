from abc import ABC, abstractmethod
from .cell import *
import numpy as np
from typing import List, Union


class MapManager(ABC):
    """
    Map Manager
    TODO Description
    """
    def __init__(self, cell_template: Cell):
        # get params
        self.cell_template = cell_template

        self.map_indices = np.empty((2, 0))
        self.x_min, self.x_max = 0, 0
        self.y_min, self.y_max = 0, 0

    @abstractmethod
    def update(self, points: np.array) -> List[Cell]:
        """ compute required local hilbert maps for map given new points """
        raise NotImplementedError

    def update_intervals(self, x_min, x_max, y_min, y_max):
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
    """
    def __init__(self, cell_template: Union[Cell, Rectangle]):
        super(GridMap, self).__init__(cell_template)

    def update(self, points: np.array) -> List[Cell]:
        # cells for all points
        new_cells = []
        x_size = self.cell_template.r1_mag * 2
        y_size = self.cell_template.r2_mag * 2
        indices = np.concatenate((np.expand_dims(points[0, :] // x_size, axis=0),
                                  np.expand_dims(points[1, :] // y_size, axis=0)), axis=0)
        indices = np.unique(indices, axis=1)

        # compare with already existing cells (keep only new cells)
        new_indices = ~((self.map_indices[:, None] == indices[..., None]).all(0).any(1))

        # only proceed if there are new indices
        if new_indices.any():
            indices = indices[:, new_indices]
            self.map_indices = np.hstack((self.map_indices, indices))

            # new cells
            for center in indices.T:
                new_cells.append(self.cell_template.new_cell((center[0] * x_size + self.cell_template.r1_mag,
                                                              center[1] * y_size + self.cell_template.r2_mag)))

            # update intervals
            x_min, x_max = np.min(indices[0, :]) * x_size, (np.max(indices[0, :]) + 1) * x_size
            y_min, y_max = np.min(indices[1, :]) * y_size, (np.max(indices[1, :]) + 1) * y_size
            self.update_intervals(x_min, x_max, y_min, y_max)

        return new_cells
