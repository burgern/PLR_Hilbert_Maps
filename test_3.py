from PLR_Hilbert_Maps.hilbert_map import LocalHilbertMap, Square, Rectangle, Ellipsoid, Circle, Hexagon,\
    LocalHilbertMapCollection
from PLR_Hilbert_Maps.models import *
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import pickle
import torch.nn as nn


def visualize(points, c: np.array = None, xlim: Tuple[float, float] = (-2, 2), ylim: Tuple[float, float] = (-2, 2)):
    plt.scatter(points[0, :], points[1, :], c=c, s=1)
    plt.xlim([xlim[0], xlim[1]])
    plt.ylim([ylim[0], ylim[1]])
    plt.show()


class Data:
    def __init__(self, n, center, size):
        self.points = np.random.uniform(-size/2, size/2, (2, n))
        self.reflectance = np.random.rand(n)
        elipse_out = Ellipsoid(center=center, r1=(2, 2), r2=(-5, 5), nx=0.5, ny=0.5)
        elipse_in = Ellipsoid(center=center, r1=(1.8, 1.8), r2=(-4.8, 4.8), nx=0.5, ny=0.5)
        elipse_out_mask = elipse_out.is_point_in_cell(self.points)
        elipse_in_mask = elipse_in.is_point_in_cell(self.points)
        walls_in = Square(center=(0, 0), width=19.6, nx=0.5, ny=0.5)
        self.occupancy = elipse_out_mask
        self.points = self.points[:, ~elipse_in_mask]
        self.occupancy = self.occupancy[~elipse_in_mask]
        self.occupancy = self.occupancy | ~walls_in.is_point_in_cell(self.points)


def main():
    # configuration
    exp_name = "local_hilbert_map_test_v001"

    # load data
    data = Data(n=100000, center=(2, 2), size=20)
    visualize(data.points, c=data.occupancy, xlim=(-10, 10), ylim=(-10, 10))

    # initialize cell
    cell = Rectangle(center=None, width=1, length=4, nx=0.5, ny=0.5)

    # initialize local model
    local_model = LocalModel(MLP(), nn.BCELoss(), lr=0.005, batch_size=32, epochs=10)

    # initialize local hilbert map collection
    lhmc = LocalHilbertMapCollection(cell, local_model)

    # update local hilbert map collection
    lhmc.update(data.points, data.occupancy)

    # plot
    lhmc.plot(1001)

    # log results

    # save lhm
    pickle.dump(lhmc, open('lhmc.pkl', 'wb'))


if __name__ == "__main__":
    main()
