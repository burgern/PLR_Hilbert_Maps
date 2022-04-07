from .composite_design import Composite
from .cell.cell import Cell
from PLR_Hilbert_Maps.models import LocalModel
from .map_manager import MapManager, GridMap
import numpy as np
from .local_hilbert_map import LocalHilbertMap
import matplotlib.pyplot as plt


class LocalHilbertMapCollection(Composite):
    """
    Local Hilbert Map Collection
    TODO Description
    """
    def __init__(self, cell_template: Cell, local_model: LocalModel, map_manager: str = 'GridMap'):
        # get params
        self.cell_template = cell_template
        self.local_model = local_model
        self.map_manager = GridMap(cell_template) if map_manager is 'GridMap' else None

        self.lhm_collection = []  # store all local hilbert maps in list

    def update(self, points: np.array, occupancy: np.array):
        # check for points which are not within Leafs (LocalHilbertMaps)
        mask_points_out_of_collection = ~ self.is_point_in_collection(points)
        points_out_of_collection = points[:, mask_points_out_of_collection]

        # get required new Leafs from MapManager
        new_cells = self.map_manager.update(points_out_of_collection)
        new_lhms = [LocalHilbertMap(cell, self.local_model.new_model()) for cell in new_cells]

        # add and update new Leafs to lhm collection
        self.lhm_collection.extend(new_lhms)
        for lhm in self.lhm_collection:
            lhm.update(points, occupancy)

    def predict(self, points: np.array):
        out = np.empty((len(self.lhm_collection), points.shape[1]))
        for lhm_idx, lhm in enumerate(self.lhm_collection):
            out[lhm_idx, :] = lhm.predict_2(points)
        return out

    def evaluate(self):
        raise NotImplementedError

    def plot(self, resolution):
        # get grid points
        x = np.linspace(self.map_manager.x_min, self.map_manager.x_max, resolution)
        y = np.linspace(self.map_manager.y_min, self.map_manager.y_max, resolution)
        xx, yy = np.meshgrid(x, y)
        points = np.concatenate((np.expand_dims(xx.flatten(), axis=0), np.expand_dims(yy.flatten(), axis=0)), axis=0)

        # get predictions
        zz = np.empty((points.shape[1],))
        for lhm in self.lhm_collection:
            mask = lhm.cell.is_point_in_cell(points)
            points_in_cell = points[:, mask]
            zz_in_cell = np.squeeze(lhm.predict(points_in_cell))
            zz[mask] = zz_in_cell
        zz = zz.reshape(len(y), len(x))
        np.nan_to_num(zz, nan=-1.0)  # nan values are points where no predictions

        # plot
        fig, ax = plt.subplots(figsize=(10, 5))
        mapping = ax.contourf(x, y, zz, levels=10, cmap='binary')
        fig.colorbar(mapping)
        for lhm in self.lhm_collection:
            ax.add_patch(lhm.cell.patch())  # add patches
        plt.show()

    def is_point_in_collection(self, points: np.array):
        mask = np.zeros(points.shape[1], dtype=bool)
        for lhm in self.lhm_collection:
            mask = mask | lhm.cell.is_point_in_cell()
        return mask
