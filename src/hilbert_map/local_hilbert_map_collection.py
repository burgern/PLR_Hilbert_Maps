from .composite_design import Composite
from src.hilbert_map.cell import Cell
from src.models.local_model import LocalModel
from .map_manager import GridMap
import numpy as np
from .local_hilbert_map import LocalHilbertMap
import matplotlib.pyplot as plt
from typing import Optional
from config import PATH_LOG
import os
import pickle


class LocalHilbertMapCollection(Composite):
    """
    Local Hilbert Map Collection
    TODO Description
    """
    def __init__(self, cell_template: Cell, local_model: LocalModel, x_neighbour_dist: float,
                 y_neighbour_dist: float, map_manager: str = 'GridMap'):
        # get params
        self.cell_template = cell_template
        self.local_model = local_model
        self.map_manager = GridMap(cell_template, x_neighbour_dist, y_neighbour_dist) if map_manager is 'GridMap' else None

        self.prev_id = -1
        self.lhm_collection = []  # store all local hilbert maps in list

    def update(self, points: np.array, occupancy: np.array):
        # check for points which are not within Leafs (LocalHilbertMaps)
        mask_points_out_of_collection = ~ self.is_point_in_collection(points)
        points_out_of_collection = points[:, mask_points_out_of_collection]

        # get required new Leafs from MapManager
        new_cells = self.map_manager.update(points_out_of_collection)
        new_lhms = [LocalHilbertMap(cell, self.local_model.new_model(), self.prev_id + cell_idx + 1) for cell_idx, cell in enumerate(new_cells)]
        self.prev_id += len(new_cells)

        # add and update new Leafs to lhm collection
        self.lhm_collection.extend(new_lhms)
        for lhm in self.lhm_collection:
            lhm.update(points, occupancy)

    def predict(self, points: np.array):
        # TODO burgern: Please use torch.tensors as type instead of np to avoid conversions at global level.
        out = np.empty((len(self.lhm_collection), points.shape[1]))
        for lhm_idx, lhm in enumerate(self.lhm_collection):
            out[lhm_idx, :] = lhm.predict_2(points)
        return out

    def evaluate(self):
        raise NotImplementedError

    def plot(self, resolution, exp_name: Optional[str] = None, name: Optional[str] = None, show_patch: bool = True,
             show_id: bool = True):
        # get grid points
        x = np.linspace(self.map_manager.x_min, self.map_manager.x_max, resolution)
        y = np.linspace(self.map_manager.y_min, self.map_manager.y_max, resolution)
        xx, yy = np.meshgrid(x, y)
        points = np.concatenate((np.expand_dims(xx.flatten(), axis=0), np.expand_dims(yy.flatten(), axis=0)), axis=0)

        # get predictions
        zz = np.empty((points.shape[1],))
        zz[:] = np.nan
        for lhm in self.lhm_collection:
            mask = lhm.cell.is_point_in_cell(points)
            points_in_cell = points[:, mask]
            zz_in_cell = np.squeeze(lhm.predict(points_in_cell))
            zz[mask] = zz_in_cell
        zz = zz.reshape(len(y), len(x))
        #zz = np.nan_to_num(zz, nan=-1.0)  # nan values are points where no predictions

        # plot
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor('lightcoral')
        mapping = ax.contourf(x, y, zz, levels=10, cmap='binary')
        fig.colorbar(mapping)
        for lhm in self.lhm_collection:
            if show_patch:
                ax.add_patch(lhm.cell.patch())  # add patches
            if show_id:
                ax.text(lhm.cell.center[0], lhm.cell.center[1], str(lhm.id), color="blue", fontsize=12)
        plt.axis('scaled')

        if exp_name is not None:
            path_exp = os.path.join(PATH_LOG, exp_name)
            if not os.path.exists(path_exp):
                os.makedirs(path_exp)
                print("created new experiment log folder")
            path = os.path.join(path_exp, name)
            plt.savefig(path)
        else:
            plt.show()

    def is_point_in_collection(self, points: np.array):
        mask = np.zeros(points.shape[1], dtype=bool)
        for lhm in self.lhm_collection:
            mask = mask | lhm.cell.is_point_in_cell(points)
        return mask

    def log(self, exp_name: str, name: str):
        if exp_name is not None:
            path_exp = os.path.join(PATH_LOG, exp_name)
            if not os.path.exists(path_exp):
                os.makedirs(path_exp)
                print("created new experiment log folder")
            path = os.path.join(path_exp, name)
        with open(path, "wb") as f:
            pickle.dump(self, f)
