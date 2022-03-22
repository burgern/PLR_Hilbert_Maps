from PLR_Hilbert_Maps.data import GridManager
import numpy as np
import matplotlib.pyplot as plt


class VisualizePredictions:
    def __init__(self, gm: GridManager, x_min, x_max, y_min, y_max, resolution):
        # arguments
        self.gm = gm
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution

        # initialize grid points
        self.x_arr, self.y_arr, self.x_np, self.y_np = self.initialize_grid_points()

        # predictions on grid points
        self.pred_np = self.grid_predictions()

    def initialize_grid_points(self):
        x_arr = np.arange(self.x_min, self.x_max, self.resolution)
        y_arr = np.arange(self.y_min, self.y_max, self.resolution)
        x_np = np.zeros((len(x_arr) * len(y_arr), 1))
        y_np = np.zeros((len(x_arr) * len(y_arr), 1))
        return x_arr, y_arr, x_np, y_np

    def grid_predictions(self):
        pred_np = np.zeros((len(self.x_arr) * len(self.y_arr), 1))
        i = 0
        for x in self.x_arr:
            for y in self.y_arr:
                self.x_np[i] = x
                self.y_np[i] = y
                self.pred_np[i] = self.gm.pred(x, y)
                i += 1
        return pred_np

    def visualize(self):
        plt.scatter(self.x_np, self.y_np, s=2, c=self.pred_np, cmap='viridis')
        plt.colorbar()
        plt.show()

    def visualize_cell(self, x, y):
        pos = self.gm.pos[self.gm.cell_mask[x, y, :]]
        occ = self.gm.occ[self.gm.cell_mask[x, y, :]]

        plt.scatter(pos[:, 0], pos[:, 1], s=2, c=occ, cmap='viridis')
        plt.colorbar()
        plt.show()
