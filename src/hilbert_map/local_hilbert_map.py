from typing import Optional, Tuple, Dict, Union
from matplotlib.contour import ContourSet
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from math import sqrt
import pickle
import numpy as np
from os.path import join

from .composite_design import Leaf
from src.hilbert_map.cell import Cell, Square, Rectangle, Circle, Ellipsoid,\
    Hexagon
from src.models.base_model import BaseModel
from src.utils.math_utils import meshgrid_points
from src.models import MLP
from torch import nn
import json
from config import PATH_PLR


class LocalHilbertMap(Leaf):
    """
    Local Hilbert Map
    TODO Description
    """
    def __init__(self, config: Union[Dict, Tuple[Cell, BaseModel, bool, int]],
                 center: Optional[Tuple[float, float]] = None,
                 id: Optional[int] = None):
        self.config = config
        self.id = id
        self.center = center
        if type(config) == dict:
            self.cell, self.local_model = self._init_from_config()
        else:
            self.cell, self.local_model = config[0], config[1]
            self.center = self.cell.center
        self.point_buffer = None
        self.occupancy_buffer = None
        if type(config) == dict:
            self.use_buffer = config["local"]["use_buffer"]
            self.max_data_buffer_length = \
                config["local"]["max_data_buffer_length"]
        else:
            self.use_buffer = config[2]
            self.max_data_buffer_length = config[3]

    def update(self, points: np.array, occupancy: np.array):
        points, mask = self.mask_points(points)
        occupancy = occupancy[mask]
        points = self.cell.original_to_normalized(points)
        if self.use_buffer and np.sum(mask) > 0:
            self.do_buffer_management(points, occupancy)
            self.local_model.train(self.point_buffer, self.occupancy_buffer)
        else:
            self.local_model.train(points, occupancy)

    def predict(self, points: np.array) -> Tuple[np.array, np.array]:
        points, mask = self.mask_points(points)
        points = self.cell.original_to_normalized(points)
        pred = self.local_model.predict(points)
        return pred, mask

    def mask_points(self, points: np.array) -> np.array:
        """ masking points which are part of current cell """
        mask = self.cell.is_point_in_cell(points)
        points = points[:, mask]
        return points, mask

    def plot(self, ax: Optional[Axes] = None, resolution: int = 100) -> \
            Optional[ContourSet]:
        """ plot the local hilbert map
        :param ax matplotlib.axes to be plotted onto
        :param resolution of predicted points per unit of LHM which are plotted
        :return mapping colormap
        min square size: s is computed as defined on following website
        see https://math.stackexchange.com/questions/2928820/
        how-to-calculate-the-bounding-square-of-an-ellipse
        """
        # generate meshgrid formatted points for plotting
        s = sqrt(2) * sqrt(self.cell.r1_mag ** 2 + self.cell.r2_mag ** 2)
        center_x, center_y = self.cell.center[0], self.cell.center[1]
        points, x, y = meshgrid_points(x_start=center_x-s/2, x_end=center_x+s/2,
                                       y_start=center_y-s/2, y_end=center_y+s/2,
                                       resolution=resolution)

        # compute predictions of relevant points
        pred, mask = self.predict(points)
        pred_all = np.empty(points.shape[1])
        pred_all[:] = np.nan
        pred_all[mask] = pred

        if ax is None:
            # plot directly
            mapping = plt.contourf(x, y, pred_all.reshape(len(y), len(x)),
                                   levels=np.linspace(0, 1, 11), cmap='binary')
            plt.colorbar(mapping)
            plt.show()
            return None
        else:
            # plot onto axes
            mapping = ax.contourf(x, y, pred_all.reshape(len(y), len(x)),
                                  levels=np.linspace(0, 1, 11), cmap='binary')
            return mapping

    def save(self, path: str):
        with open(path, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _init_from_config(self):
        # load cell from config
        config_cell = self.config["cell"]
        if config_cell["type"] == "square":
            config_square = config_cell["square"]
            cell = Square(center=self.center, width=config_square["width"],
                          nx=config_cell["nx"], ny=config_cell["ny"],
                          patch_edgecolor=config_cell["patch_edgecolor"],
                          patch_linewidth=config_cell["patch_linewidth"])
        elif config_cell["type"] == "rectangle":
            config_rect = config_cell["rectangle"]
            cell = Rectangle(center=self.center, width=config_rect["width"],
                             length=config_rect["length"],
                             nx=config_cell["nx"], ny=config_cell["ny"],
                             patch_edgecolor=config_cell["patch_edgecolor"],
                             patch_linewidth=config_cell["patch_linewidth"])
        elif config_cell["type"] == "circle":
            config_circle = config_cell["circle"]
            cell = Circle(center=self.center, radius=config_circle["radius"],
                          nx=config_cell["nx"], ny=config_cell["ny"],
                          patch_edgecolor=config_cell["patch_edgecolor"],
                          patch_linewidth=config_cell["patch_linewidth"])
        elif config_cell["type"] == "ellipsoid":
            config_ell = config_cell["ellipsoid"]
            cell = Ellipsoid(center=self.center, angle=config_ell["width"],
                             radius_primary=config_ell["radius_primary"],
                             radius_secondary=config_ell["radius_secondary"],
                             nx=config_cell["nx"], ny=config_cell["ny"],
                             patch_edgecolor=config_cell["patch_edgecolor"],
                             patch_linewidth=config_cell["patch_linewidth"])
        elif config_cell["type"] == "hexagon":
            config_hex = config_cell["hexagon"]
            cell = Hexagon(center=self.center, width=config_hex["width"],
                           length=config_hex["length"],
                           nx=config_cell["nx"], ny=config_cell["ny"],
                           patch_edgecolor=config_cell["patch_edgecolor"],
                           patch_linewidth=config_cell["patch_linewidth"])
        else:
            raise ValueError

        # load local model from config
        config_local = self.config["local"]
        model_local_config_path = join(PATH_PLR, config_local["config_path"])
        with open(model_local_config_path) as f:
            model_local_config = json.load(f)
        model = MLP(model_local_config) if config_local["model"] == "MLP" else None
        loss = nn.BCELoss() if config_local["loss"] == "BCE" else None
        local_model = BaseModel(model, loss, lr=config_local["lr"],
                                batch_size=config_local["batch_size"],
                                epochs=config_local["epochs"])

        return cell, local_model

    def do_buffer_management(self, points: np.array, occupancy: np.array):
        if (self.point_buffer is None) and (self.occupancy_buffer is None):
            self.point_buffer = points
            self.occupancy_buffer = occupancy
        else:
            self.point_buffer = np.concatenate((self.point_buffer, points), axis=1)
            self.occupancy_buffer = np.concatenate((self.occupancy_buffer, occupancy))

            assert self.point_buffer.shape[1] == self.occupancy_buffer.shape[0]

            buffer_overflow = self.point_buffer.shape[1] - self.max_data_buffer_length

            if buffer_overflow > 0:
                self.point_buffer = self.point_buffer[:, buffer_overflow:]
                self.occupancy_buffer = self.occupancy_buffer[buffer_overflow:]
