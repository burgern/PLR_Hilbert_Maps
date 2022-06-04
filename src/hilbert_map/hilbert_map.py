from .composite_design import Composite
from .map_manager import GridMap
from .local_hilbert_map_collection import LocalHilbertMapCollection
from ..models.global_model.logistic_regression import LogisticRegression
from ..models.global_model.average_layer import AverageLayer
from ..models.global_model.mlp_global import MLP_global
from src.models.base_model import BaseModel
from src.models.local_model.mlp import MLP
import torch.nn as nn
from torch import no_grad
from .cell.square import Square
import json
import numpy as np
from typing import Union, Dict

class HilbertMap(Composite):
    """
    Hilbert Map
    TODO hadzica: Description
    """
    def __init__(self, config: Union[str, Dict]):
        super().__init__()
        if type(config) == str:
            with open(config) as f:
                config = json.load(f)

        if config["global"]["loss"] == "BCE":
            global_loss = nn.BCELoss()
        elif config["global"]["loss"] == "BCEWithLogitsLoss":
            global_loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError

        global_lr = config["global"]["lr"]
        global_bs = config["global"]["batch_size"]
        global_epochs = config["global"]["epochs"]
        if config["global"]["model"] == "LogisticRegression":
            global_model = LogisticRegression()
        elif config["global"]["model"] == "AverageLayer":
            global_model = AverageLayer()
        elif config["global"]["model"] == "MLP_global":
            global_model = MLP_global()
        else:
            raise ValueError

        self.global_map = BaseModel(global_model, global_loss, global_lr, global_bs, global_epochs)

        self.local_map_collection = LocalHilbertMapCollection(config)

    def update(self, points: np.array, occupancy: np.array):
        self.local_map_collection.update(points, occupancy)
        with no_grad():
            local_map_outputs, local_map_output_masks = self.local_map_collection.predict_lhm(points)

        self.global_map.train(local_map_outputs, occupancy, print_loss=False)

        self.x_limits["min"] = self.local_map_collection.map_manager.x_min
        self.x_limits["max"] = self.local_map_collection.map_manager.x_max
        self.y_limits["min"] = self.local_map_collection.map_manager.y_min
        self.y_limits["max"] = self.local_map_collection.map_manager.y_max

    def predict(self, points: np.array):
        local_map_outputs, local_map_output_masks = self.local_map_collection.predict_lhm(points)
        predictions = self.global_map.predict(local_map_outputs)
        return predictions, local_map_output_masks

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
