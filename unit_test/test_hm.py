from src.hilbert_map.hilbert_map import HilbertMap
from src.data import DatasetHexagon
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

def main(args):
    hilbert_map = HilbertMap(args.config)
    n = 100000
    size = 5
    data = DatasetHexagon(n, size, (0, 0), patch_edgecolor="r", patch_linewidth=1,)
    hilbert_map.update(data.points, data.occupancy)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    hilbert_map.plot(ax, resolution=100)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    hilbert_map.local_map_collection.plot(ax, resolution=100)
    #hilbert_map.global_map.model.plot_weights_history()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the replica dataset class")
    parser.add_argument('--config', action="store")

    args = parser.parse_args()
    main(args)
