from src.data.datasets import DataIntelLab, DataFreiburg079, DataFreiburgCampus
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


def test_plt(ax: Axes):
    ax.scatter(np.arange(100), np.arange(100), c="green", s=1)


if __name__ == "__main__":
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4.6))
    data_intel_lab = DataIntelLab()
    data_freiburg_079 = DataFreiburg079()
    data_freburg_campus = DataFreiburgCampus()

    data_intel_lab.visualize(ax=axs[0], step_size=10)
    data_freiburg_079.visualize(ax=axs[1], step_size=10)
    data_freburg_campus.visualize(ax=axs[2], step_size=10)

    axs[0].set_title("IntelLab")
    axs[1].set_title("Freiburg 079")
    axs[2].set_title("Freiburg Campus")

    fig.show()

