from argparse import ArgumentParser
from typing import Optional
import os
import numpy as np
import pickle
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from config.paths import PATH_LOG_DATASET, PATH_LOG_EXP
from src.data import DataIntelLab
from src.hilbert_map import LocalHilbertMap, Square, Cell
from config import load_config
from src.models import MLP
from src.models import BaseModel
from src.utils.evaluation_utils import create_video_stream, create_gif_from_mp4
from utils.plot_utils import plot_roc, plot_pr

CENTER = (-2, 2)
WIDTH = 4
NX, NY = 0.5, 0.5

LOCAL_MLP_CONFIG = {
  "n_inputs": 2,
  "n_outputs": 1,
  "n_hidden_layers": 3,
  "w_hidden_layers": 8,
  "activation_hidden_layers": "Tanhshrink",
  "activation_input_layer": "Tanhshrink",
  "activation_output_layer": "Sigmoid"
}

LOSS = nn.BCELoss()
LEARNING_RATE = 0.01
BATCH_SIZE = 16
EPOCHS = 1

USE_BUFFER, BUFFER_LENGTH = True, 1000


def create_lhm():
    config_local = load_config()["local"]
    cell = Square(center=CENTER, width=WIDTH, nx=NX, ny=NY, patch_linewidth=1,
                  patch_edgecolor="r")
    model = MLP(LOCAL_MLP_CONFIG)
    loss = LOSS
    local_model = BaseModel(model, loss, lr=LEARNING_RATE,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS)
    lhm = LocalHilbertMap(config=(cell, local_model, USE_BUFFER, BUFFER_LENGTH))
    return lhm

def load_dataset(load_data: bool, cell: Cell):
    # create data and cut data for lhm specific region in dataset
    # or load already generated dataset
    if load_data:
        if not os.path.exists(PATH_LOG_DATASET): raise ValueError
        with open(PATH_LOG_DATASET, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = DataIntelLab()
        dataset.crop_to_cell(cell=cell)
        dataset.save()
    return dataset


def set_up_log(exp_name: str):
    if not exp_name:
        exp_name = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    exp_path = os.path.join(PATH_LOG_EXP, exp_name)
    os.makedirs(exp_path)
    return exp_name, exp_path


def main(load_data: bool, gen_eval: bool, gen_video: bool,
         exp_name: Optional[str] = None):
    # create lhm
    lhm = create_lhm()
    x_min = lhm.cell.center[0] - lhm.cell.r1_mag
    x_max = lhm.cell.center[0] + lhm.cell.r1_mag
    y_min = lhm.cell.center[1] - lhm.cell.r2_mag
    y_max = lhm.cell.center[1] + lhm.cell.r2_mag

    # load dataset
    dataset = load_dataset(load_data=load_data, cell=lhm.cell)
    points_all, occupancy_all = dataset.data_concatenated()
    points_curr, occupancy_curr = np.empty((2, 0)), np.empty(0)

    # create log folder for experiment
    exp_name, exp_path = set_up_log(exp_name=exp_name)

    # run experiment
    fig, ((ax_roc, ax_pr, empty), (ax_data_all, ax_data, ax_plot)) = \
        plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
    ax_data.axis('equal')
    ax_data.set_xlim(x_min, x_max)
    ax_data.set_ylim(y_min, y_max)
    ax_plot.axis('equal')
    ax_plot.set_xlim(x_min, x_max)
    ax_plot.set_ylim(y_min, y_max)
    ax_data_all.scatter(x=points_all[0, :], y=points_all[1, :], c=occupancy_all,
                            cmap="viridis", s=1)
    ax_data_all.axis('equal')
    ax_data_all.set_xlim(x_min, x_max)
    ax_data_all.set_ylim(y_min, y_max)
    for update_ind, (points, occupancy, _) in enumerate(dataset):
        print(f"current update: {update_ind+1}")
        lhm.update(points, occupancy)
        points_curr = np.hstack((points_curr, points))
        occupancy_curr = np.hstack((occupancy_curr, occupancy))

        if gen_video:
            # evaluation
            pred, mask = lhm.predict(points_all)
            plot_roc(ax=ax_roc, pred=pred, gth=occupancy_all)
            plot_pr(ax=ax_pr, pred=pred, gth=occupancy_all)

            # plot current data
            ax_data.scatter(x=points[0, :], y=points[1, :], c=occupancy,
                            cmap="cool", s=1)

            # plot current lhm
            mapping = lhm.plot(ax=ax_plot, resolution=100)
            cb = fig.colorbar(mapping)

            fig.savefig(os.path.join(exp_path, f"update_{update_ind:05}.png"))

            # clear stuff
            ax_roc.clear()
            ax_pr.clear()
            ax_data.scatter(x=points[0, :], y=points[1, :], c=occupancy,
                            cmap="viridis", s=1)
            cb.remove()

        if update_ind >= 20:
            break


    # if gen_eval and not gen_video:
    #     evaluate()

    if gen_video:
        video_path = create_video_stream(image_folder=exp_path, fps=3)
        new_video_path = os.path.join(Path(video_path).parent, exp_name,
                                      Path(video_path).name)
        os.rename(video_path, new_video_path)
        gif_path = create_gif_from_mp4(video_path=new_video_path)

    del lhm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_data", action="store_true")
    parser.add_argument("--gen_eval", action="store_true")
    parser.add_argument("--gen_video", action="store_true")
    parser.add_argument("--exp_name", type=str, default="")
    args = parser.parse_args()
    main(load_data=args.load_data, gen_eval=args.gen_eval,
         gen_video=args.gen_video, exp_name=args.exp_name)
