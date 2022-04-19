from PLR_Hilbert_Maps.src.hilbert_map import Square, LocalHilbertMapCollection
from PLR_Hilbert_Maps.src.models import *
from PLR_Hilbert_Maps.src.data import DatasetHexagon
from torch import nn
import pickle


def main():
    # configuration
    exp_name = "local_hilbert_map_test_v001"

    # load data
    data = DatasetHexagon(10000, 5, (0, 0))
    data.plot()

    # initialize cell
    cell = Square(center=None, width=0.5, nx=0.5, ny=0.5)

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
