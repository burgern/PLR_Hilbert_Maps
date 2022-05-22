from abc import ABC, abstractmethod
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib.axes import Axes
from matplotlib.contour import ContourSet

from src.utils.math_utils import meshgrid_points


class Component(ABC):
    """
    The base Component class declares common operations for both simple and
    complex objects of a composition.
    In our case, LHM, LHMC, GlobalHM and HM are all components
    """

    @abstractmethod
    def update(self, points: np.array, occupancy: np.array):
        """ update component with new information
        :param points 2d array of n points (2 x n)
        :param occupancy 1d array with information about the points occupancy,
        1 being occupied, 0 being free space
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, points: np.array) -> Tuple[np.array, np.array]:
        """ prediction of component on points
        :param points 2d array of n points to be predicted (2 x n)
        :return predictions, 1d array of the respective predictions
        :return mask, 1d array of points which are used in the resp. component
        """
        raise NotImplementedError

    def evaluate(self, points: np.array, occupancy: np.array) ->\
            Tuple[np.array, np.array, np.array, np.array]:
        """ evaluation of points
        :param points array of n points to be evaluated (2 x n)
        :param occupancy 1d array with information about the points occupancy,
        1 being occupied, 0 being free space
        :return fpr, false positive rate
        :return tpr, true positive rate
        :return prec, precision
        :return recall
        """
        pred, mask = self.predict(points)
        gth = occupancy[mask]

        # extract relevant evaluation metrics
        fpr, tpr, _ = metrics.roc_curve(gth, pred)
        prec, recall, _ = metrics.precision_recall_curve(gth, pred)

        return fpr, tpr, prec, recall


class Composite(Component):
    """
    The Composite class represents the complex components that may have
    children.
    In our case, HilbertMap and LocalHilbertMapCollection are Composites.
    """
    def __init__(self):
        self.x_limits = {"min": 0, "max": 0}
        self.y_limits = {"min": 0, "max": 0}

    @abstractmethod
    def update(self, points: np.array, occupancy: np.array):
        raise NotImplementedError

    @abstractmethod
    def get_lhm_collection(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, points: np.array) -> Tuple[np.array, np.array]:
        raise NotImplementedError

    def predict_lhm(self, points: np.array) -> Tuple[np.array, np.array]:
        """ predictions of all lhm's on points
        :param points 2d array of n points to be predicted (2 x n)
        :return predictions, (#lhm's, n) array of the respective predictions
        :return mask, 1d array info if resp. point predicted by at least one lhm
        """
        pred = np.empty((len(self.get_lhm_collection()), points.shape[1]))
        pred[:, :] = np.nan
        for lhm_idx, lhm in enumerate(self.get_lhm_collection()):
            pred_lhm, mask = lhm.predict(points)
            pred[lhm_idx, mask] = pred_lhm
        mask = ~np.all(np.isnan(pred), axis=0)
        return pred[:, mask], mask

    def predict_weighted(self, points: np.array, weights: np.array) -> \
            Tuple[np.array, np.array]:
        """ weighted prediction of lhm collection on points
        :param points 2d array of n points to be predicted (2 x n)
        :param weights 1d array (size is #lhm's)
        :return predictions, 1d array of the respective predictions
        :return mask, 1d array of points which are used in the resp. component
        """
        pred, mask = self.predict_lhm(points)
        pred_weighted = []
        pred_mask = ~np.isnan(pred)
        for col_idx, (col_pred, col_mask) in enumerate(zip(pred.T,
                                                           pred_mask.T)):
            if not np.isnan(mask[col_idx]):
                pred_weighted.append(np.dot(col_pred[col_mask],
                                            weights[col_mask]) /
                                     np.sum(weights[col_mask]))
        return np.array(pred_weighted), mask

    def plot(self, ax: Axes, resolution: int = 10, show_patch: bool = True,
             show_id: bool = False) -> ContourSet:
        """ plot composite onto axes of a matplotlib figure
        :param ax is the matplotlib axes to be plotted onto
        :param resolution of predicted points per unit length for plotting
        :param show_patch shows the cells shape in the plot
        :param show_id shows the resp. id of each cell
        :return mapping -> colormap to be added to the figure separately
        """
        points, x, y = meshgrid_points(x_start=self.x_limits["min"],
                                       x_end=self.x_limits["max"],
                                       y_start=self.y_limits["min"],
                                       y_end=self.y_limits["max"],
                                       resolution=resolution)

        # compute weighted predictions of relevant points
        pred, mask = self.predict(points=points)
        pred_all = np.empty(points.shape[1])
        pred_all[:] = np.nan
        pred_all[mask] = pred

        # plot onto axes
        mapping = ax.contourf(x, y, pred_all.reshape(len(y), len(x)),
                              levels=10, cmap='binary')
        ax.set_facecolor('lightcoral')
        plt.axis('scaled')
        for lhm in self.get_lhm_collection():
            if show_patch:
                ax.add_patch(lhm.cell.patch())  # add patches
            if show_id:
                ax.text(lhm.cell.center[0], lhm.cell.center[1], str(lhm.id),
                        color="orange", fontsize=12)
        return mapping


class Leaf(Component):
    """
    The Leaf class represents the end objects of a composition. A leaf can't
    have any children.
    In our case, GlobalModel and LocalHilbertMap are Leafs.
    """

    def update(self, points: np.array, occupancy: np.array):
        raise NotImplementedError

    def predict(self, points: np.array) -> Tuple[np.array, np.array]:
        raise NotImplementedError

    def plot(self, ax: Axes, res: int = 100):
        raise NotImplementedError
