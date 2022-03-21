import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from PLR_Hilbert_Maps.training import train


class GridManager:
    """
        Manages Grid Map (GM) framework for:
        - grid map size
        - grid map occupancy map
        - data masks per grid
        - visualization
        Inputs:
            - pos:  <x-pos, y-pos>
            - occ:  <occ>
            - cell_width:  x-dir size of cell
            - cell_length:  y-dir size of cell
            - exp_name:  name of the experiment
    """
    def __init__(self, pos: np.array, occ: np.array, cell_width: float, cell_length: float, exp_name: str):
        # input parameters
        self.pos = pos
        self.occ = occ
        self.cell_width = cell_width
        self.cell_length = cell_length
        self.exp_name = exp_name

        # grid map specs
        self.gm_width, self.gm_length, self.bottom_left_offset = self.gm_size()
        self.gm_shape = self.gm_compute_shape()

        # grid map operations
        self.cell_mask = self.compute_cell_masks()  # iterate over grid whilst creating a cell mask
        self.occupancy_mask = self.compute_occupancy_map()  # iterate over grid to check for non-occupied grids

        # empty model array
        self.grid_models = np.empty(self.gm_shape, dtype=object)

    def gm_size(self):
        x_min = self.pos[:, 0].min()
        x_max = self.pos[:, 0].max()
        y_min = self.pos[:, 1].min()
        y_max = self.pos[:, 1].max()
        return x_max - x_min, y_max - y_min, [x_min, y_min]

    def gm_compute_shape(self):
        x_size = int(self.gm_width // self.cell_width + 1)
        y_size = int(self.gm_length // self.cell_length + 1)
        return x_size, y_size

    def compute_cell_masks(self):
        cell_mask = np.full(self.gm_shape + (self.pos.shape[0],), False)
        for x in range(0, self.gm_shape[0]):
            for y in range(0, self.gm_shape[1]):
                x_interval_min = x * self.cell_width + self.bottom_left_offset[0]
                x_interval_max = x_interval_min + self.cell_width
                y_interval_min = y * self.cell_length + self.bottom_left_offset[1]
                y_interval_max = y_interval_min + self.cell_length

                within_bound = (((self.pos[:, 0] >= x_interval_min) &
                                 (self.pos[:, 0] < x_interval_max)) &
                                ((self.pos[:, 1] >= y_interval_min) &
                                 (self.pos[:, 1] < y_interval_max)))
                cell_mask[x, y, :] = np.array(within_bound, dtype=bool)
        return cell_mask

    def compute_occupancy_map(self):
        occupancy_mask = np.full(self.gm_shape, True)
        for x in range(0, self.gm_shape[0]):
            for y in range(0, self.gm_shape[1]):
                if np.all((self.cell_mask[x, y, :] == 0)):
                    print(f'cell <{x}, {y}> is empty')
                    occupancy_mask[x, y] = False
        return occupancy_mask

    def train_model(self, cell_x, cell_y, device, model, loss_fn, optimizer, batch_size, epochs):
        # add cell data to DataLoader
        pos = self.pos[self.cell_mask[cell_x, cell_y, :]]
        occ = self.occ[self.cell_mask[cell_x, cell_y, :]]
        pos_tensor = torch.Tensor(pos)
        occ_tensor = torch.Tensor(occ)
        data = TensorDataset(pos_tensor, occ_tensor)
        dataloader = DataLoader(data, batch_size=batch_size)

        # train cell model
        for t in range(epochs):
            print(f"Cell: <{cell_x}, {cell_y}> ---- Epoch {t + 1}\n-------------------------------")
            train(dataloader, model, device, loss_fn, optimizer)
        print("Done!")

        # save model
        model_path = f"{self.exp_name}_x_{cell_x}_y_{cell_y}"
        torch.save(model.state_dict(), model_path)
        self.grid_models[cell_x, cell_y] = model


    def visualize_cell(self, x, y):
        pos = self.pos[self.cell_mask[x, y, :]]
        occ = self.occ[self.cell_mask[x, y, :]]

        plt.scatter(pos[:, 0], pos[:, 1], s=2, c=occ, cmap='viridis')
        plt.colorbar()
        plt.show()

    def pred(self, x_pos, y_pos):
        # find the correct cell
        cell_x = int((x_pos - self.bottom_left_offset[0]) // self.cell_width)
        cell_y = int((y_pos - self.bottom_left_offset[1]) // self.cell_length)

        # load local cell model
        model = self.grid_models[cell_x, cell_y]
        model.load_state_dict(torch.load(f"../scripts/{self.exp_name}_x_{cell_x}_y_{cell_y}"))

        # build data
        pos_tensor = torch.Tensor(np.array([x_pos, y_pos]))
        data = TensorDataset(pos_tensor)

        # make prediction
        model.eval()
        with torch.no_grad():
            pred = model(data)
        pred_np = pred.cpu().detach().numpy()

        return pred_np
