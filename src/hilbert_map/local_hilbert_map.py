from .composite_design import Leaf
from src.hilbert_map.cell import Cell
from src.models.base_model import BaseModel
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


class LocalHilbertMap(Leaf):
    """
    Local Hilbert Map
    TODO Description
    """
    def __init__(self, cell: Cell, local_model: BaseModel, id: Optional[int] = None):
        self.cell = cell
        self.local_model = local_model
        self.id = id

    def update(self, points: np.array, occupancy: np.array):
        points, occupancy = self.preprocessing(points, occupancy)  # preprocessing
        # print(f'training model of cell: x_pos = {self.cell.center[0]}, y_pos = {self.cell.center[1]} --- start')
        self.local_model.train(points, occupancy)  # train local model
        # print(f'training model of cell: x_pos = {self.cell.center[0]}, y_pos = {self.cell.center[1]} --- finish')

    def predict(self, points: np.array):
        points = self.preprocessing(points)
        return self.local_model.predict(points)

    def predict_2(self, points: np.array):
        out = np.empty(points.shape[1])
        mask = self.cell.is_point_in_cell(points)
        out[~mask] = np.nan
        if np.any(mask):
            x = np.squeeze(self.local_model.predict(points[:, mask]))
            out[mask] = x
        return out

    def plot(self, size_x: float, size_y: float, resolution: int):
        # get grid points
        x = np.linspace(self.cell.center[0] - size_x / 2, self.cell.center[0] + size_x / 2, resolution)
        y = np.linspace(self.cell.center[1] - size_y / 2, self.cell.center[1] + size_y / 2, resolution)
        xx, yy = np.meshgrid(x, y)
        points = np.concatenate((np.expand_dims(xx.flatten(), axis=0), np.expand_dims(yy.flatten(), axis=0)), axis=0)

        # normalize and predict for all points
        points_norm = self.cell.original_to_normalized(points)
        zz = self.local_model.predict(points_norm).reshape(len(y), len(x))

        # plot
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        mapping = axs[0].contourf(x, y, zz, levels=10, cmap='binary')
        fig.colorbar(mapping)
        axs[0].add_patch(self.cell.patch())
        axs[1].contourf(np.linspace(-self.cell.nx, self.cell.nx, resolution),
                        np.linspace(-self.cell.ny, self.cell.ny, resolution),
                        zz, levels=10, cmap='binary')
        plt.axis('scaled')
        plt.show()

    def preprocessing(self, points: np.array, occupancy: Optional[np.array] = None):
        mask = self.cell.is_point_in_cell(points)
        points = points[:, mask]  # only use data which lies within cell
        points = self.cell.original_to_normalized(points)  # normalize data
        if occupancy is None:
            return points
        occupancy = occupancy[mask]  # only use data which lies within cell
        return points, occupancy
