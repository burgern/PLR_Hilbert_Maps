from PLR_Hilbert_Maps.hilbert_map import Square, LocalHilbertMapCollection
from PLR_Hilbert_Maps.models import *
from PLR_Hilbert_Maps.data import DatasetHexagon
from torch import nn


class LHMCParam:
    def __init__(self, exp_name: str, updates: int, cell_width: float, data_n: int, data_size: float, lr: float, batch_size: int,
                 epochs: int, x_neighbour_dist: float, y_neighbour_dist: float):
        self.exp_name = exp_name
        self.updates = updates
        self.cell_width = cell_width
        self.data_n = data_n
        self.data_size = data_size
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.x_neighbour_dist = x_neighbour_dist
        self.y_neighbour_dist = y_neighbour_dist

        # exp name
        self.exp_name_abs = f'{exp_name}_n_{self.data_n}_size_{self.data_size}_lr_{self.lr}_batchsize_{self.batch_size}_epochs_{self.epochs}'

        # initialize cell
        self.cell = Square(center=None, width=self.cell_width, nx=0.5, ny=0.5)

        # initialize local model
        self.local_model = LocalModel(MLP(), nn.BCELoss(), lr=self.lr, batch_size=self.batch_size, epochs=self.epochs)

        # initialize local hilbert map collection
        self.lhmc = LocalHilbertMapCollection(self.cell, self.local_model, x_neighbour_dist=self.x_neighbour_dist,
                                              y_neighbour_dist=self.y_neighbour_dist)

        # load data
        self.data = DatasetHexagon(self.data_n, self.data_size, (0, 0))
        self.data.plot()

    def run_simulation(self):
        for i in range(0, self.updates):
            self.data.next()
            self.lhmc.update(self.data.points, self.data.occupancy)
            self.lhmc.plot(1001, self.exp_name_abs, f'iteration_{i}.png')
            print(f'{self.exp_name_abs}: finished iteration {i}')
        self.lhmc.plot(1001)
        self.lhmc.log(self.exp_name_abs, f'lhmc.pkl')


def main():
    # configuration
    exp_name = "lhmc_test_v009"
    updates = 20

    # parameters to test
    n = [100000]
    size = 5
    lr = [0.02]
    epochs = [2]

    # initialize and run simulation
    for n_ in n:
        for lr_ in lr:
            for epochs_ in epochs:
                simulation = LHMCParam(exp_name=exp_name, updates=updates, cell_width=2, data_n=n_, data_size=size, lr=lr_,
                                       batch_size=32, epochs=epochs_, x_neighbour_dist=1, y_neighbour_dist=1)
                simulation.run_simulation()


if __name__ == "__main__":
    main()
