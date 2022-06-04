import os
from os.path import join, exists
from typing import Optional, Dict, Union, Tuple, List
from tkinter import Tk
from tkinter.filedialog import askdirectory
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import cm
from math import cos, sin

from config import PATH_LOG
from src.hilbert_map import LocalHilbertMap, LocalHilbertMapCollection,\
    HilbertMap
from src.utils.plot_utils import plot_meshgrid
from utils.plot_utils import plot_pr, plot_roc
from utils.model_setup_utils import generate_data
from src.utils.evaluation_utils import create_video_stream, create_gif_from_mp4
from argparse import ArgumentParser

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

        # extract relevant config information
        self.config = self.load_config()
        self.model = self.config["model"]["model"]
        assert self.model in ["lhm", "lhmc", "hm"], "[EVALUATOR] model found " \
                                                    "in config not possible"

    def create_update_video_stream(self, fps: int = 3) -> \
            Tuple[str, str]:
        fig, ((ax_roc, ax_pr, empty),
              (ax_data, ax_data_cumul, ax_plot)) = \
            plt.subplots(nrows=2, ncols=3, figsize=(12, 12))

        # generated images path creation (empty)
        generated_images_path = join(self.exp_path, "generated_images")
        if not os.path.exists(generated_images_path):
            os.makedirs(generated_images_path)

        # ax_data (stays static)
        dataset = self.plot_dataset(ax=ax_data)

        for update in range(1, self.nr_updates+1):
            # ax_data_cumul
            self.plot_dataset_at_update(ax=ax_data_cumul, update=update,
                                        cmap="cool")

            # ax_plot
            pose = dataset[update - 1]["pose"]
            mapping = self.plot_model(ax=ax_plot, update=update,
                                      pose=(pose["position"][0],
                                            pose["position"][1],
                                            pose["angle"]))
            cb = fig.colorbar(mapping)

            # evaluations: ax_roc and ax_pr
            ax_roc.clear()
            ax_pr.clear()
            self.plot_roc_at_update(ax=ax_roc, update=update)
            self.plot_pr_at_update(ax=ax_pr, update=update)

            # save figure
            fig.savefig(join(generated_images_path, f"update_{update:05}.png"))

            # set figure to standard for next iteration
            self.plot_dataset_at_update(ax=ax_data_cumul, update=update,
                                        cmap="viridis")
            cb.remove()

        # generate video in .mp4 and .gif format
        mp4_path = create_video_stream(image_folder=generated_images_path,
                                       fps=fps)
        gif_path = create_gif_from_mp4(mp4_path)
        return mp4_path, gif_path

    def evaluate_model(self, model: Optional[Union[LocalHilbertMap,
                                                   LocalHilbertMapCollection,
                                                   HilbertMap]] = None):
        fig, ((ax_roc, ax_pr), (ax_data, ax_plot)) = \
            plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

        model = self.load_model() if model is None else model

        # ax_data (stays static)
        dataset = self.plot_dataset(ax=ax_data)

        # evaluations: ax_roc and ax_pr
        points, occupancy = dataset.data_concatenated()
        pred, mask = model.predict(points=points)
        gth = occupancy[mask]
        plot_roc(ax=ax_roc, pred=pred, gth=gth)
        plot_pr(ax=ax_pr, pred=pred, gth=gth)

        # ax_plot
        if self.model == "lhm":
            mapping = model.plot(ax=ax_plot, resolution=100)
            fig.colorbar(mapping)
        elif self.model in ["lhmc", "hm"]:
            mapping = model.plot(ax=ax_plot, resolution=10, show_patch=True,
                                 show_id=False)
            fig.colorbar(mapping)
        else:
            raise ValueError
        ax_plot.axis('equal')
        # ax_data.get_shared_x_axes().join(ax_data, ax_plot)

        fig.show()

    def plot_dataset(self, ax: Axes):
        dataset = generate_data(config=self.config)
        dataset.visualize(ax=ax, step_size=1)
        ax.axis('equal')
        return dataset

    def plot_dataset_at_update(self, ax: Axes, update: int, cmap: str):
        data = self.load_data(update=update)
        ax.scatter(data[0, :], data[1, :], c=data[2, :],
                   cmap=cmap, s=1)
        ax.axis('equal')

    def plot_model(self, ax: Axes, update: int, size: float = 2,
                   pose: Optional[Tuple[float, float, float]] = None):
        """ visualize the models predictions at the current state over all
        possible points
        @:param pose creates an arrow showing the current (x_pos, y_pos, angle)
        @:return returns colormapping which can be added to figure
        """
        ax.clear()
        if self.model == "lhm":
            model = self.load_lhm_model(update=update)
            mapping = model.plot(ax=ax)
            return mapping
        elif self.model in ["lhmc", "hm"]:
            x, y, zz = self.load_gm_plot(update=update)
        else:
            raise ValueError
        mapping = plot_meshgrid(ax=ax, x=x, y=y, zz=zz)
        if pose is not None:
            dir_vect = np.array([cos((pose[2])), sin((pose[2]))]) * size
            ax.quiver(pose[0], pose[1], dir_vect[0], dir_vect[1])
        ax.axis('equal')
        return mapping

    def plot_roc_at_update(self, ax: Axes, update: int):
        if self.model == "lhm":
            pred, gth = self.load_lhm_eval(update=update)
        elif self.model in ["lhmc", "hm"]:
            pred, gth = self.load_gm_eval(update=update)
        else:
            raise ValueError
        plot_roc(ax=ax, pred=pred, gth=gth)

    def plot_pr_at_update(self, ax: Axes, update: int):
        if self.model == "lhm":
            pred, gth = self.load_lhm_eval(update=update)
        elif self.model in ["lhmc", "hm"]:
            pred, gth = self.load_gm_eval(update=update)
        else:
            raise ValueError
        plot_pr(ax=ax, pred=pred, gth=gth)

    def load_config(self) -> Dict:
        config_path = join(self.exp_path, "config.json")
        with open(config_path) as file:
            out = json.load(file)
        return out

    def load_model(self) -> Union[LocalHilbertMap, LocalHilbertMapCollection,
                                  HilbertMap]:
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

    def load_lhm_model(self, update: int, lhm_id: int = 1) -> LocalHilbertMap:
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

    def load_gm_eval(self, update: int) -> Tuple:
        update_dir = self.get_update_dir_path(update=update)
        lhmc_path = join(update_dir, "global_map")
        out = self.load_np_from_dir(dir_path=lhmc_path, files=EVAL_FILES)
        return out[0], out[1]

    def load_gm_plot(self, update: int) -> Tuple:
        update_dir = self.get_update_dir_path(update=update)
        gm_path = join(update_dir, "global_map")
        out = self.load_np_from_dir(dir_path=gm_path, files=PLOT_FILES)
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
    parser = ArgumentParser()
    parser.add_argument("--gen_video", action="store_true")
    args = parser.parse_args()

    evaluator = Evaluator()
    evaluator.evaluate_model()
    if args.gen_video:
        evaluator.create_update_video_stream()
