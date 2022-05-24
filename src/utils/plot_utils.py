from matplotlib.contour import ContourSet
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


def plot_meshgrid(ax: Axes, x: np.array, y: np.array, zz: np.array) -> \
        ContourSet:
    """ plot composite onto axes of a matplotlib figure
    :param ax is the matplotlib axes to be plotted onto
    :param x linspace in x dir
    :param y linspace in y dir
    :param zz meshgrid of predicted points spanned by x linspace and y linspace
    :return mapping -> colormap to be added to the figure separately
    """
    mapping = ax.contourf(x, y, zz, levels=10, cmap='binary')
    ax.set_facecolor('lightcoral')
    plt.axis('scaled')
    return mapping
