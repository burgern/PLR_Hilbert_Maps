from src.hilbert_map import Square, LocalHilbertMapCollection
from src.models import *
from src.data import DatasetHexagon
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
from config import PATH_LOG, load_config
import pickle


def main():
    # configuration
    exp_name = "lhmc_v100"

    # load config
    config = load_config()
    config_local, config_cell, config_map_manager = config["local"], config["cell"], config["map_manager"]

    # load dataset
    updates = 200
    data = DatasetHexagon(10000, 5, (0, 0))

    # lhmc setup
    cell = Square(center=config_cell["center"],
                  width=config_cell["width"],
                  nx=config_cell["nx"], ny=config_cell["ny"])
    local_model = BaseModel(MLP(), nn.BCELoss(),
                            lr=config_local["lr"],
                            batch_size=config_local["batch_size"],
                            epochs=config_local["epochs"])
    lhmc = LocalHilbertMapCollection(cell, local_model,
                                     x_neighbour_dist=config_map_manager["x_neighbour_dist"],
                                     y_neighbour_dist=config_map_manager["y_neighbour_dist"])

    # run local hilbert map collection
    points_total, occupancy_total = np.empty((2, 0)), np.empty(0)
    path_exp = os.path.join(PATH_LOG, exp_name)

    for i in range(updates):
        # update data
        points, occupancy, _ = data.next()

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
