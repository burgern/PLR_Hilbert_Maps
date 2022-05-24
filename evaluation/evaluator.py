import os
from os.path import join, exists
from typing import Optional, Dict, Union, Tuple, List
from tkinter import Tk
from tkinter.filedialog import askdirectory
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from config import PATH_LOG
from src.hilbert_map import LocalHilbertMap, LocalHilbertMapCollection
from src.utils.plot_utils import plot_meshgrid
from utils.plot_utils import plot_pr, plot_roc
from utils.model_setup_utils import generate_data

EVAL_FILES = ("pred.npy", "gth.npy")
PLOT_FILES = ("x_meshgrid.npy", "y_meshgrid.npy", "zz_meshgrid.npy")


class Evaluator:
    def __init__(self, exp_name: Optional[str] = None):
        # choose experiment through gui if not given
        if exp_name is None:
            Tk().withdraw()
            exp_name = askdirectory(title="choose experiment to be evaluated",
                                    initialdir=PATH_LOG)
        self.exp_name = exp_name
        self.exp_path = join(PATH_LOG, exp_name)

        # updates sorted
        self.update_dirs = [join(self.exp_path, update_dir)
                            for update_dir in os.listdir(self.exp_path)
                            if 'update' in update_dir]
        self.update_dirs.sort()
        self.nr_updates = int(self.update_dirs[-1][-5:].lstrip('0'))

    def evaluate_model(self):
        fig, ((ax_roc, ax_pr), (ax_data, ax_plot)) = \
            plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

        model = self.load_model()
        config = self.load_config()

        # evaluations: ax_roc and ax_pr
        dataset = generate_data(config=config)
        points, occupancy = dataset.data_concatenated()
        pred, mask = model.predict(points=points)
        gth = occupancy[mask]
        plot_roc(ax=ax_roc, pred=pred, gth=gth)
        plot_pr(ax=ax_pr, pred=pred, gth=gth)

        # ax_data
        dataset.visualize(ax=ax_data, step_size=1)
        ax_data.axis('equal')

        # ax_plot
        if type(model) == LocalHilbertMap:
            model.plot(ax=ax_plot, resolution=100)
        elif type(model) == LocalHilbertMapCollection:
            model.plot(ax=ax_plot, resolution=10, show_patch=True,
                       show_id=False)
        else:
            raise ValueError
        ax_plot.axis('equal')

        fig.show()

    def evaluate_lhmc_at_update(self, update: Optional[int] = None):
        update = self.nr_updates if update is None else update
        fig, ((ax_roc, ax_pr), (ax_data, ax_plot)) = \
            plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

        # evaluations: ax_roc and ax_pr
        pred, gth = self.load_lhmc_eval(update=update)
        plot_roc(ax=ax_roc, pred=pred, gth=gth)
        plot_pr(ax=ax_pr, pred=pred, gth=gth)

        # ax_data
        data = self.load_data(update=update)
        ax_data.scatter(data[0, :], data[1, :], c=data[2, :], cmap="viridis",
                        s=1)
        ax_data.axis('equal')

        # ax_plot
        x, y, zz = self.load_lhmc_plot(update=update)
        mapping = plot_meshgrid(ax=ax_plot, x=x, y=y, zz=zz)
        fig.colorbar(mapping)
        ax_plot.axis('equal')

        fig.show()

    def load_config(self) -> Dict:
        config_path = join(self.exp_path, "config.json")
        with open(config_path) as file:
            out = json.load(file)
        return out

    def load_model(self) -> Union[LocalHilbertMap, LocalHilbertMapCollection]:
        model_path = join(self.exp_path, "model.pkl")
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model

    def load_map(self, update: int) -> np.array:
        update_dir = self.get_update_dir_path(update=update)
        map_path = join(update_dir, "map.npy")
        return np.load(map_path)

    def load_data(self, update: int) -> np.array:
        update_dir = self.get_update_dir_path(update=update)
        data_path = join(update_dir, "data.npy")
        return np.load(data_path)

    def load_lhm_model(self, update: int, lhm_id: int) -> LocalHilbertMap:
        update_dir = self.get_update_dir_path(update=update)
        lhm_path = join(update_dir, "lhm", f"lhm_{lhm_id:05}")
        with open(lhm_path, "rb") as file:
            lhm = pickle.load(file)
        return lhm

    def load_lhm_eval(self, update: int, lhm_id: int) -> Tuple:
        update_dir = self.get_update_dir_path(update=update)
        lhm_path = join(update_dir, "lhm", f"lhm_{lhm_id:05}")
        out = self.load_np_from_dir(dir_path=lhm_path, files=EVAL_FILES)
        return out[0], out[1]

    def load_lhmc_eval(self, update: int) -> Tuple:
        update_dir = self.get_update_dir_path(update=update)
        lhmc_path = join(update_dir, "lhmc")
        out = self.load_np_from_dir(dir_path=lhmc_path, files=EVAL_FILES)
        return out[0], out[1]

    def load_lhmc_plot(self, update: int) -> Tuple:
        update_dir = self.get_update_dir_path(update=update)
        lhmc_path = join(update_dir, "lhmc")
        out = self.load_np_from_dir(dir_path=lhmc_path, files=PLOT_FILES)
        return out[0], out[1], out[2]

    @staticmethod
    def load_np_from_dir(dir_path: str, files: Tuple) -> List:
        out = []
        for file in files:
            file_path = join(dir_path, file)
            out.append(np.load(file_path)) if exists(file_path) else \
                out.append(None)
        return out

    def get_update_dir_path(self, update: int) -> str:
        assert update >= 1, "update must be >= 1"
        return self.update_dirs[update - 1]


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate_model()
