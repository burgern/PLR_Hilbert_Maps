import numpy as np
import pickle
from typing import Dict, Tuple

from src.hilbert_map.cell import Cell
from src.models.base_model import BaseModel
from src.hilbert_map import Composite
from src.hilbert_map.local_hilbert_map import LocalHilbertMap
from .map_manager import GridMap


class LocalHilbertMapCollection(Composite):
    """
    Local Hilbert Map Collection
    TODO Description
    """
    def __init__(self, config: Dict, map_manager: str = 'GridMap'):
        super().__init__()
        # get params
        self.config = config
        self.cell_template, self.local_model = self._init_from_config()

        # choose map_manager
        if map_manager is 'GridMap':
            x_neighbour_dist = self.config["map_manager"]["x_neighbour_dist"]
            y_neighbour_dist = self.config["map_manager"]["y_neighbour_dist"]
            self.map_manager = GridMap(cell_template=self.cell_template,
                                       x_neighbour_dist=x_neighbour_dist,
                                       y_neighbour_dist=y_neighbour_dist)
        else:
            self.map_manager = None

        self.prev_id = -1
        self.lhm_collection = []  # store all local hilbert maps in list
        self.x_limits = {"min": 0, "max": 0}
        self.y_limits = {"min": 0, "max": 0}

    def _init_from_config(self) -> Tuple[Cell, BaseModel]:
        lhm_template = LocalHilbertMap(config=self.config, center=None, id=None)
        return lhm_template.cell, lhm_template.local_model

    def update(self, points: np.array, occupancy: np.array):
        # check for points which are not within Leafs (LocalHilbertMaps)
        mask_point_in_lhmc = self.is_point_in_collection(points)
        points_not_in_lhmc = points[:, ~mask_point_in_lhmc]

        # get required new Leafs from MapManager
        new_cells = self.map_manager.update(points_not_in_lhmc)
        new_lhms = [LocalHilbertMap(config=(cell, self.local_model.new_model(),
                                            self.config["local"]["max_data_buffer_length"]),
                                    id=self.prev_id + cell_idx + 1)
                    for cell_idx, cell in enumerate(new_cells)]
        self.prev_id += len(new_cells)

        # add and update new Leafs to lhm collection
        self.lhm_collection.extend(new_lhms)
        for lhm in self.lhm_collection:
            lhm.update(points, occupancy)

        self.x_limits["min"] = self.map_manager.x_min
        self.x_limits["max"] = self.map_manager.x_max
        self.y_limits["min"] = self.map_manager.y_min
        self.y_limits["max"] = self.map_manager.y_max

    def predict(self, points: np.array) -> Tuple[np.array, np.array]:
        weights = np.ones(len(self.lhm_collection), dtype=np.float)
        return self.predict_weighted(points, weights)

    def is_point_in_collection(self, points: np.array):
        mask = np.zeros(points.shape[1], dtype=bool)
        for lhm in self.lhm_collection:
            mask = mask | lhm.cell.is_point_in_cell(points)
        return mask

    def get_lhm_collection(self):
        return self.lhm_collection
