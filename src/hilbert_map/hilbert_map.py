from .composite_design import Composite
from .map_manager import GridMap
from .local_hilbert_map_collection import LocalHilbertMapCollection
from ..models.global_model.logistic_regression import LogisticRegression
from src.models.base_model import BaseModel
from src.models.local_model.mlp import MLP
import torch.nn as nn
from torch import no_grad
from .cell.square import Square
import json
import numpy as np

class HilbertMap(Composite):
    """
    Hilbert Map
    TODO hadzica: Description
    """
    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as f:
            config = json.load(f)

        if config["local"]["model"] == "MLP":
            local_model = MLP()
        else:
            # TODO hadzica: Add other cases.
            raise NotImplementedError

        if config["local"]["loss"] == "BCE":
            local_loss = nn.BCELoss()
        else:
            # TODO hadzica: Add other cases.
            raise NotImplementedError

        if config["cell"]["type"] == "SQUARE":
            nx = config["cell"]["nx"]
            ny = config["cell"]["ny"]
            width = config["cell"]["width"]
            if "None" in config["cell"]["center"]:
                center = None
            cell = Square(center=center, width=width, nx=nx, ny=ny)
        else:
            # TODO hadzica: Add other cases.
            raise NotImplementedError

        if config["global"]["loss"] == "BCE":
            global_loss = nn.BCELoss()
        elif config["global"]["loss"] == "BCEWithLogitsLoss":
            global_loss = nn.BCEWithLogitsLoss()
        else:
            # TODO hadzica: Add other cases.
            raise NotImplementedError

        if config["global"]["model"] == "LogisticRegression":
            global_lr = config["global"]["lr"]
            global_bs = config["global"]["batch_size"]
            global_epochs = config["global"]["epochs"]
            global_model = LogisticRegression()
            self.global_map = BaseModel(global_model, global_loss, global_lr, global_bs, global_epochs)
        else:
            # TODO hadzica: Add other cases.
            raise NotImplementedError

        x_neighbour_dist = config["global"]["overlap_in_x"]
        y_neighbour_dist = config["global"]["overlap_in_y"]

        local_lr = config["local"]["lr"]
        local_bs = config["local"]["batch_size"]
        local_epochs = config["local"]["epochs"]

        local_map = BaseModel(local_model, local_loss, lr=local_lr, batch_size=local_bs, epochs=local_epochs)
        self.local_map_collection = LocalHilbertMapCollection(cell, local_map, x_neighbour_dist=x_neighbour_dist,
                                                              y_neighbour_dist=y_neighbour_dist)

    def update(self, points: np.array, occupancy: np.array):
        self.local_map_collection.update(points, occupancy)
        with no_grad():
            local_map_outputs = self.local_map_collection.predict(points)
        self.global_map.train(local_map_outputs, occupancy, print_loss=True)

        self.x_limits["min"] = self.local_map_collection.map_manager.x_min
        self.x_limits["max"] = self.local_map_collection.map_manager.x_max
        self.y_limits["min"] = self.local_map_collection.map_manager.y_min
        self.y_limits["max"] = self.local_map_collection.map_manager.y_max

    def predict(self, points: np.array):
        local_map_outputs = self.local_map_collection.predict(points)
        predictions = self.global_map.predict(local_map_outputs)
        return predictions

    def predict_meshgrid(self, points: np.array):
        zz = self.predict(points)
        size = int(np.sqrt(zz.shape[0]))
        zz = zz.reshape(size, size)
        return zz

    def evaluate(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def get_lhm_collection(self):
        return self.local_map_collection.lhm_collection
