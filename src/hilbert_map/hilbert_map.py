from .composite_design import Composite
from .map_manager import GridMap
from .local_hilbert_map_collection import LocalHilbertMapCollection
from ..models.global_model.logistic_regression import LogisticRegression
from src.models.base_model import BaseModel
from src.models.local_model.mlp import MLP
import torch.nn as nn
from .cell.square import Square
import json
import numpy as np

class HilbertMap(Composite):
    """
    Hilbert Map
    TODO hadzica: Description
    """
    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = json.load(f)

        if config["local_model"]["model"] == "MLP":
            local_model = MLP()
        else:
            # TODO hadzica: Add other cases.
            raise NotImplementedError

        if config["local_model"]["loss"] == "BCE":
            local_loss = nn.BCELoss()
        else:
            # TODO hadzica: Add other cases.
            raise NotImplementedError

        if config["cell"]["type"] == "SQUARE":
            # TODO burgern: Fix correct nx, ny pass.
            cell = Square(config["cell"]["width"], 0, 0)
        else:
            # TODO hadzica: Add other cases.
            raise NotImplementedError

        if config["global"]["loss"] == "BCE":
            global_loss = nn.BCELoss()
        if config["global"]["model"] == "LogisticRegression":
            global_config = config["global"]
            self.global_map = LogisticRegression(global_config, global_loss)

        local_lr = config["local_model"]["lr"]
        local_bs = config["local_model"]["batch_size"]
        local_epochs = config["local_model"]["epochs"]

        map_manager = GridMap(cell)
        local_map = BaseModel(local_model, local_loss, lr=local_lr, batch_size=local_bs, epochs=local_epochs)
        self.local_map_collection = LocalHilbertMapCollection(cell, local_map, map_manager)

    def update(self, points: np.array, occupancy: np.array):
        self.local_map_collection.update(points, occupancy)
        local_map_outputs = self.local_map_collection.predict(points)
        self.global_map.update(local_map_outputs, occupancy)

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError
