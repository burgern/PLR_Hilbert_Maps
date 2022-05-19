from src.data.datasets import DataIntelLab, DataFreiburg079, DataFreiburgCampus
import matplotlib.pyplot as plt
import numpy as np
from config import PATH_LOG_DATA
import os
from typing import Union


def create_video(experiment_name: str, start_ind: int, end_ind: int,
                        dataset: Union[DataIntelLab, DataFreiburg079,
                                       DataFreiburgCampus]):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4.6))
    data = dataset()
    data.visualize(ax=axs[0])
    for i in np.arange(start_ind, end_ind):
        print(f"iteration: {i}")
        axs[2].clear()
        data.visualize_viewpoint(ax=axs[1], index=i)
        data.visualize_viewpoint(ax=axs[2], index=i)
        path = os.path.join(PATH_LOG_DATA, f"{experiment_name}_{i:05}.png")
        fig.savefig(path)  # save the figure to file


if __name__ == "__main__":
    # create_video("intellab_v001", 0, 910, DataIntelLab)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4.6))
    data_intel_lab = DataIntelLab()
    data_freiburg_079 = DataFreiburg079()
    data_freiburg_campus = DataFreiburgCampus()

    data_intel_lab.visualize(ax=axs[0], step_size=10)
    data_freiburg_079.visualize(ax=axs[1], step_size=10)
    data_freiburg_campus.visualize(ax=axs[2], step_size=10)

    axs[0].set_title("IntelLab")
    axs[0].set_aspect('equal', 'box')
    axs[1].set_title("Freiburg 079")
    axs[1].set_aspect('equal', 'box')
    axs[2].set_title("Freiburg Campus")
    axs[2].set_aspect('equal', 'box')

    fig.show()
