import os
from typing import Dict
import sys
import json
from typing import Optional, List, Union
import numpy as np
import torch
import pickle

from config import load_config, PATH_LOG
from src.hilbert_map import LocalHilbertMap, LocalHilbertMapCollection, \
    MapManager, HilbertMap



class Logger:
    """
    Logger for logging generated assets for evaluation
    Structure:
        log
        |---config.json  (config_path)
        |
        |---model.pkl  (use model.save_model(model) function after last update)
        |
        |---update_00001  (curr_update_dir)
        |    |
        |    |---lhm  (lhm_dir)
        |    |    |
        |    |    |---lhm_00001  (curr_lhm_dir)
        |    |    |    |---pred.npy
        |    |    |    |---gth.npy
        |    |    |    |---model.pt
        |    |    |
        |    |    |---lhm_00002
        |    |    |    |---pred.npy
        |    |    |    |---gth.npy
        |    |    |    |---model.pt
        |    |    |
        |    |    |---...
        |    |
        |    |---lhmc  (lhmc_dir)
        |    |    |
        |    |    |---pred.npy
        |    |    |---gth.npy
        |    |    |---x_meshgrid.npy
        |    |    |---y_meshgrid.npy
        |    |    |---zz_meshgrid.npy
        |    |
        |    |---map.npy  (map_path)
        |    |
        |    |---data.npy  (data_path)
        |
        |---...

    """
    def __init__(self, exp_name: str, config: Dict = {}):
        # read args and setup class
        self.exp_name = exp_name
        self.exp_path = os.path.join(PATH_LOG, exp_name)
        self.config = {**load_config(), **config}
        self._setup()

        # running parameters
        self.step = 1  # we start with update step 1
        self.curr_update_dir = None

    def _setup(self):
        # set up log directory for experiment if not yet already existing
        if os.path.exists(self.exp_path):
            continue_flag = input("log path already exists, do you want to " +
                                  "overwrite? hit enter -> yes, any input -> no")
            if continue_flag == "":
                sys.exit()
        self.create_dir(self.exp_path)

        # log configuration
        config_path = os.path.join(self.exp_path, "config.json")
        with open(config_path, 'w') as file:
            json.dump(self.config, file, indent=4)

    def update(self, data: Optional[np.array] = None,
               model: Optional[Union[LocalHilbertMap,
                                     LocalHilbertMapCollection]] = None):
        # set up directory for logging this specific update
        self.curr_update_dir = os.path.join(self.exp_path,
                                            f"update_{self.step:05}")
        self.create_dir(self.curr_update_dir)

        # save data if given
        if data is not None:
            self.log_data(data)

        # save model specific files (different for LHM, LHMC and NHM)
        if type(model) == LocalHilbertMap:
            self.log_lhm([model, data]) if data is not None else \
                self.log_lhm([model])
        elif type(model) == LocalHilbertMapCollection:
            if data is not None:
                self.log_gm(model, data)
                self.log_lhm(model.lhm_collection, data)
            else:
                self.log_gm(model)
                self.log_lhm(model.lhm_collection)
            self.log_map_manager(model.map_manager)
        elif type(model) == HilbertMap:
            if data is not None:
                self.log_gm(model, data)
                self.log_lhm(model.local_map_collection.lhm_collection, data)
            else:
                self.log_gm(model)
                self.log_lhm(model.lhm_collection)
            self.log_map_manager(model.local_map_collection.map_manager)

        else:
            return ValueError

        # increase step for next update
        self.step += 1

    def log_data(self, data: np.array):
        data_path = os.path.join(self.curr_update_dir, "data.npy")
        np.save(data_path, data)

    def log_lhm(self, lhm_: List[LocalHilbertMap], data: Optional[np.array]):
        # set up directory for logging lhm's
        lhm_dir = os.path.join(self.curr_update_dir, "lhm")
        self.create_dir(lhm_dir)

        # save model and evaluations
        for lhm_idx, lhm in enumerate(lhm_):
            # set up directory for logging this specific lhm
            curr_lhm_dir = os.path.join(lhm_dir, f"lhm_{lhm_idx:05}")
            self.create_dir(curr_lhm_dir)

            # save model
            model = lhm.local_model.model.state_dict()
            model_path = os.path.join(curr_lhm_dir, "model.pt")
            torch.save(model, model_path)

            # save prediction and ground truth if there is data given
            if data is not None:
                self.save_pred_and_gth(model=lhm, dir_path=curr_lhm_dir,
                                       points=data[0:2, :],
                                       occupancy=data[2, :])

    def log_gm(self, lhmc: Union[LocalHilbertMapCollection, HilbertMap],
               data: Optional[np.array]):
        # set up directory for logging lhmc
        lhmc_dir = os.path.join(self.curr_update_dir, "global_map")
        self.create_dir(lhmc_dir)

        # save lhmc numpy plot
        x, y, zz = lhmc.predicted_meshgrid_points()
        np.save(os.path.join(lhmc_dir, "x_meshgrid.npy"), x)
        np.save(os.path.join(lhmc_dir, "y_meshgrid.npy"), y)
        np.save(os.path.join(lhmc_dir, "zz_meshgrid.npy"), zz)

        # save prediction and ground truth if there is data given
        if data is not None:
            self.save_pred_and_gth(model=lhmc, dir_path=lhmc_dir,
                                   points=data[0:2, :], occupancy=data[2, :])

    @staticmethod
    def save_pred_and_gth(model: Union[LocalHilbertMap,
                                       LocalHilbertMapCollection],
                          dir_path: str, points: np.array, occupancy: np.array):
        pred, mask = model.predict(points)
        gth = occupancy[mask]
        np.save(os.path.join(dir_path, "pred.npy"), pred)
        np.save(os.path.join(dir_path, "gth.npy"), gth)

    def log_map_manager(self, map_manager: MapManager):
        map_path = os.path.join(self.curr_update_dir, "map.npy")
        np.save(map_path, map_manager.map_indices)

    def save_model(self, source):
        pkl_path = os.path.join(self.exp_path, "model.pkl")
        with open(pkl_path, 'wb') as file:
            pickle.dump(source, file)

    @staticmethod
    def create_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)
