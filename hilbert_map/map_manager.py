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

        self.cell_list = []
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
        # add new cells for all points which are not within Local Hilbert Map
        new_cells = []
        x_size = self.cell_template.r1_mag * 2
        y_size = self.cell_template.r2_mag * 2
        indices = np.concatenate((np.expand_dims(points[0, :] // x_size, axis=0),
                                  np.expand_dims(points[1, :] // y_size, axis=0)), axis=0)
        indices = np.unique(indices.T, axis=0)
        for center in indices:
            new_cells.append(self.cell_template.new_cell((center[0] * x_size + self.cell_template.r1_mag,
                                                          center[1] * y_size + self.cell_template.r2_mag)))

        # update intervals
        x_min, x_max = np.min(indices.T[0, :]) * x_size, (np.max(indices.T[0, :]) + 1) * x_size
        y_min, y_max = np.min(indices.T[1, :]) * y_size, (np.max(indices.T[1, :]) + 1) * y_size
        self.update_intervals(x_min, x_max, y_min, y_max)

        return new_cells
