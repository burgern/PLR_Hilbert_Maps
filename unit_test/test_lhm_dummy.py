from src.hilbert_map import LocalHilbertMap
from src.data import DatasetHexagon
from config import load_config
import matplotlib.pyplot as plt
import pickle


def main():
    # run new experiment or load from pickle
    run_new = True
    if run_new:
        # generate data
        data = DatasetHexagon(n=100000, size=5, center=(0, 0),
                              patch_edgecolor="r", patch_linewidth=1)

        # create LHM from config
        config = load_config()
        lhm = LocalHilbertMap(config=config, center=(0, 0))

        # first step
        points, occupancy, _ = data.__next__()
        lhm.update(points, occupancy)

        # save model
        lhm.save("test_lhm_dummy.pkl")

    else:
        with open('test_lhm_dummy.pkl', 'rb') as file:
            lhm = pickle.load(file)

    # plot LHM
    fig, ax = plt.subplots(nrows=1, ncols=1)
    mapping = lhm.plot(ax=ax, resolution=100)
    fig.colorbar(mapping)
    fig.show()


if __name__ == "__main__":
    main()
