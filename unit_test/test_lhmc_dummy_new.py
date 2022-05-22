from src.hilbert_map import LocalHilbertMapCollection
from src.data import DatasetHexagon
from config import load_config
import matplotlib.pyplot as plt
import pickle


def main():
    # run new experiment or load from pickle
    run_new = True
    if run_new:
        # generate data
        data = DatasetHexagon(n=10000, size=5, center=(0, 0),
                              patch_edgecolor="r", patch_linewidth=1, updates=3)

        # create LHMC from config
        config = load_config()
        lhmc = LocalHilbertMapCollection(config=config, x_neighbour_dist=1,
                                         y_neighbour_dist=1)

        # first step
        # for points, occupancy, _ in data:
        #     lhmc.update(points, occupancy)
        points, occupancy, _ = data.__next__()
        lhmc.update(points, occupancy)

        # save model
        lhmc.save("test_lhmc_dummy.pkl")

    else:
        with open('test_lhmc_dummy.pkl', 'rb') as file:
            lhmc = pickle.load(file)

    # plot LHM
    fig, ax = plt.subplots(nrows=1, ncols=1)
    mapping = lhmc.plot(ax=ax, resolution=100, show_id=True)
    fig.colorbar(mapping)
    fig.show()
    print("hello")


if __name__ == "__main__":
    main()
