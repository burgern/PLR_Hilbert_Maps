from src.hilbert_map import Square, LocalHilbertMapCollection
from src.models import *
from src.data import DatasetHexagon
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
from config import PATH_LOG, load_config
import pickle
from evaluation import Logger


def main():

    points, occupancy, _ = data.next()  # update data

    lhmc.update(points, occupancy)  # update lhmc

    # log data, lhm, lhmc and map_manager
    logger.update(data=np.vstack((points, occupancy)),
                  lhm=lhmc.lhm_collection,
                  lhmc=lhmc,
                  map_manager=lhmc.map_manager)

    logger.save_as_pickle(file_name="lhmc.pkl", source=lhmc)

    # save lhmc
    with open('lhmc.pkl', 'wb') as file:
        pickle.dump(lhmc, file)


if __name__ == "__main__":
    main()
