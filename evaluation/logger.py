from config import PATH_LOG
import os
from typing import Dict
from config import load_config
import json
from src.hilbert_map import LocalHilbertMap, LocalHilbertMapCollection, MapManager
from typing import Optional, List
import numpy as np
import torch
import pickle


class Logger:
    """
    Logger for logging generated assets for evaluation
    Structure:
        log
        |---config.json  (config_path)
        |
        |---lhmc.pkl  (example of usage with save_as_pickle function)
        |---hm.pkl  (example of usage with save_as_pickle function)
        |
        |---update_00001  (curr_update_dir)
        |    |
        |    |---lhm  (lhm_dir)
        |    |    |
        |    |    |---lhm_00001  (curr_lhm_dir)
        |    |    |    |---tpr.npy
        |    |    |    |---fpr.npy
        |    |    |    |---precision.npy
        |    |    |    |---recall.npy
        |    |    |    |---model.pt
        |    |    |
        |    |    |---lhm_00002
        |    |    |    |---tpr.npy
        |    |    |    |---fpr.npy
        |    |    |    |---precision.npy
        |    |    |    |---recall.npy
        |    |    |    |---model.pt
        |    |    |
        |    |    |---...
        |    |
        |    |---lhmc  (lhmc_dir)
        |    |    |
        |    |    |---tpr.npy
        |    |    |---fpr.npy
        |    |    |---precision.npy
        |    |    |---recall.npy
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
        # set up log directory for experiment
        self.create_dir(self.exp_path)

        # log configuration
        config_path = os.path.join(self.exp_path, "config.json")
        with open(config_path, 'w') as file:
            json.dump(self.config, file, indent=4)

    def update(self, data: Optional[np.array] = None, lhm: Optional[List[LocalHilbertMap]] = None,
               lhmc: Optional[LocalHilbertMapCollection] = None, map_manager: Optional[MapManager] = None):
        # set up directory for logging this specific update
        self.curr_update_dir = os.path.join(self.exp_path, f"update_{self.step:05}")
        self.create_dir(self.curr_update_dir)

        # log updates if they are given
        if data is not None:
            self.log_data(data)
            if lhm is not None:
                self.log_lhm(lhm, data)
            if lhmc is not None:
                self.log_lhmc(lhmc, data)
        else:
            if lhm is not None:
                self.log_lhm(lhm)
            if lhmc is not None:
                self.log_lhmc(lhmc)
        if map_manager is not None:
            self.log_map_manager(map_manager)

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

            # save evaluations if there is data given
            if data is not None:
                fpr, tpr, prec, recall = lhm.evaluate(points=data[0:2, :],
                                                      occupancy=data[2, :])
                np.save(os.path.join(curr_lhm_dir, "fpr.npy"), fpr)
                np.save(os.path.join(curr_lhm_dir, "tpr.npy"), tpr)
                np.save(os.path.join(curr_lhm_dir, "precision.npy"), prec)
                np.save(os.path.join(curr_lhm_dir, "recall.npy"), recall)

    def log_lhmc(self, lhmc: LocalHilbertMapCollection, data: Optional[np.array]):
        # set up directory for logging lhmc
        lhmc_dir = os.path.join(self.curr_update_dir, "lhmc")
        self.create_dir(lhmc_dir)

        # save lhmc numpy plot
        x, y, zz = lhmc.predict_meshgrid(resolution=1001)
        np.save(os.path.join(lhmc_dir, "x_meshgrid.npy"), x)
        np.save(os.path.join(lhmc_dir, "y_meshgrid.npy"), y)
        np.save(os.path.join(lhmc_dir, "zz_meshgrid.npy"), zz)

        # save evaluations if there is data given
        # TODO implement this -> adapt lhmc predict function
        # if data is not None:
        #     fpr, tpr, prec, recall = lhmc.evaluate(points=data[0:2, :],
        #                                            occupancy=data[2, :])
        #     np.save(os.path.join(lhmc_dir, "fpr.npy"), fpr)
        #     np.save(os.path.join(lhmc_dir, "tpr.npy"), tpr)
        #     np.save(os.path.join(lhmc_dir, "precision.npy"), prec)
        #     np.save(os.path.join(lhmc_dir, "recall.npy"), recall)

    def log_map_manager(self, map_manager: MapManager):
        map_path = os.path.join(self.curr_update_dir, "data.npy")
        np.save(map_path, map_manager.map_indices)

    def save_as_pickle(self, file_name: str, source):
        pickel_path = os.path.join(self.exp_path, file_name)
        with open(pickel_path, 'wb') as file:
            pickle.dump(source, file)

    @staticmethod
    def create_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)
