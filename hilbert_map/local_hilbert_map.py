from .composite_design import Leaf
from .cell.cell import Cell
from PLR_Hilbert_Maps.models import LocalModel


import numpy as np


class LocalHilbertMap(Leaf):
    """
    Local Hilbert Map
    TODO Description
    """
    def __init__(self, cell: Cell, local_model: LocalModel):
        self.cell = cell
        self.local_model = local_model

    def update(self, points: np.array, occupancy: np.array):
        points, occupancy = self.preprocessing(points, occupancy)  # preprocessing
        print(f'training model of cell: x_pos = {self.cell.center[0]}, y_pos = {self.cell.center[1]}')
        self.local_model.train(points, occupancy)  # train local model

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def preprocessing(self, points: np.array, occupancy: np.array):
        mask = self.cell.is_point_in_cell(points)
        points, occupancy = points[:, mask], occupancy[mask]  # only use data which lies within cell
        points = self.cell.original_to_normalized(points)  # normalize data
        return points, occupancy
