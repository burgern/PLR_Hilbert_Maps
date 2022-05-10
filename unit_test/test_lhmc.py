from src.hilbert_map import Square, LocalHilbertMapCollection
from src.models import *
from src.data import DatasetHexagon
from torch import nn
import numpy as np
from config import load_config
import pickle
from evaluation import Logger


def main():
    # configuration
    exp_name = "lhmc_v101"

    # load config
    config = load_config()
    config_local, config_cell = config["local"], config["cell"]
    config_map_manager, config_dataset = config["map_manager"], config["dataset"]["dummy"]

    # load dataset
    updates = config_dataset["updates"]
    data = DatasetHexagon(n=config_dataset["points"],
                          size=config_dataset["size"],
                          center=(config_dataset["center_x"], config_dataset["center_y"]))

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

    # logger setup
    logger = Logger(exp_name=exp_name, config=config)

    # run local hilbert map collection
    points_total, occupancy_total = np.empty((2, 0)), np.empty(0)

    for i in range(updates):
        points, occupancy, _ = data.next()  # update data

        lhmc.update(points, occupancy)  # update lhmc

        # update dataset
        points_total = np.hstack((points_total, points))
        occupancy_total = np.hstack((occupancy_total, occupancy))

        print(f'iteration: {i} done')

        # log data, lhm, lhmc and map_manager
        logger.update(data=np.vstack((points, occupancy)),
                      lhm=lhmc.lhm_collection,
                      lhmc=lhmc,
                      map_manager=lhmc.map_manager)

    logger.save_as_pickle(file_name="lhmc.pkl", source=lhmc)


if __name__ == "__main__":
    main()
