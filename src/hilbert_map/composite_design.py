from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import os
from config import PATH_LOG


class Component(ABC):
    """
    The base Component class declares common operations for both simple and
    complex objects of a composition.
    """

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError


class Composite(Component):
    """
    The Composite class represents the complex components that may have
    children.
    In our case, HilbertMap and LocalHilbertMapCollection are Composites.
    """
    def __init__(self):
        self.x_limits = {"min": 0, "max": 0}
        self.y_limits = {"min": 0, "max": 0}

    def update(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self, resolution, exp_name: Optional[str] = None, name: Optional[str] = None, show_patch: bool = True,
             show_id: bool = True):
        # get grid points
        x = np.linspace(self.x_limits["min"], self.x_limits["max"], resolution)
        y = np.linspace(self.y_limits["min"], self.y_limits["max"], resolution)
        xx, yy = np.meshgrid(x, y)
        points = np.concatenate((np.expand_dims(xx.flatten(), axis=0), np.expand_dims(yy.flatten(), axis=0)), axis=0)

        zz = self.predict_meshgrid(points)
        # plot
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor('lightcoral')
        mapping = ax.contourf(x, y, zz, levels=10, cmap='binary')
        fig.colorbar(mapping)
        for lhm in self.get_lhm_collection():
            if show_patch:
                ax.add_patch(lhm.cell.patch())  # add patches
            if show_id:
                ax.text(lhm.cell.center[0], lhm.cell.center[1], str(lhm.id), color="blue", fontsize=12)
        plt.axis('scaled')

        if exp_name is not None:
            path_exp = os.path.join(PATH_LOG, exp_name)
            if not os.path.exists(path_exp):
                os.makedirs(path_exp)
                print("created new experiment log folder")
            path = os.path.join(path_exp, name)
            plt.savefig(path)
        else:
            plt.show()


class Leaf(Component):
    """
    The Leaf class represents the end objects of a composition. A leaf can't
    have any children.
    In our case, GlobalModel and LocalHilbertMap are Leafs.
    """

    def update(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
