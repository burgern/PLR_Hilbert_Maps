import random

from src.hilbert_map import Square, LocalHilbertMapCollection
from src.models import *
from src.data import DatasetHexagon
from torch import nn
import numpy as np
from src.data.replica_dataset import ReplicaDataSet
from config import PATH_CONFIG
import os
import json
import matplotlib.pyplot as plt
from config import PATH_LOG
import pickle


def main():
    # configuration
    exp_name = "lhmc_test_v016"

    # load dataset
    nr_of_viewpoints = 200
    config_path = os.path.join(PATH_CONFIG, 'test_config.json')
    with open(config_path) as f:
        config = json.load(f)
    data = ReplicaDataSet(config)

    # lhmc setup
    cell = Square(center=None, width=1, nx=0.5, ny=0.5)
    local_model = LocalModel(MLP(), nn.BCELoss(), lr=0.05, batch_size=32, epochs=1)
    lhmc = LocalHilbertMapCollection(cell, local_model, x_neighbour_dist=1, y_neighbour_dist=1)

    # run local hilbert map collection
    points_total, occupancy_total = np.empty((2, 0)), np.empty(0)
    path_exp = os.path.join(PATH_LOG, exp_name)

    for i in range(nr_of_viewpoints):
        # update data
        data_step = data.__getitem__(i)
        points, occupancy = data_step[:2, :], data_step[2, :]

        # update lhmc
        lhmc.update(points, occupancy)

        # plot lhmc and new data
        lhmc.plot(1001, exp_name, f'iteration_{i:03}.png', show_id=False)
        plt.close('all')
        plt.scatter(points_total[0, :], points_total[1, :], c=occupancy_total, s=1, cmap='viridis')
        plt.scatter(points[0, :], points[1, :], c=occupancy, s=1, cmap='cool')
        path = os.path.join(path_exp, f'viewpoint_{i:03}.png')
        plt.axis('scaled')
        plt.savefig(path)

        # update dataset
        points_total = np.hstack((points_total, points))
        occupancy_total = np.hstack((occupancy_total, occupancy))

        print(f'iteration: {i} done')

    # save lhmc
    with open('lhmc.pkl', 'wb') as file:
        pickle.dump(lhmc, file)

    lhmc.evaluate(points_total, occupancy_total)


if __name__ == "__main__":
    main()
