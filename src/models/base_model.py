from src.utils.device_setup import device_setup
from .local_model.mlp import MLP

from configparser import ConfigParser
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
import copy


class BaseModel(nn.Module):
    """
    Local Model
    TODO Description
    """
    def __init__(self, model: nn.Module = MLP(), loss_fn: nn.modules.loss = nn.BCELoss(), lr: float = 1e-3,
                 batch_size: int = None, epochs: int = None):
        super().__init__()
        # read inputs
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device_setup()  # device setup (cpu or gpu)

    def predict(self, points: np.array):
        x = torch.Tensor(points.T)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x, inference=True)
        pred_np = pred.cpu().detach().numpy()
        return pred_np

    def evaluate(self):
        raise NotImplementedError

    def train(self, points: np.array, occupancy: np.array, print_loss: bool = False):
        if points.size == 0:
            return
        dataloader = self.get_dataloader(points, occupancy)  # get data in required pytorch format
        model = self.model.to(self.device)

        # train model
        loss_per_epoch = []
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        for t in range(self.epochs):
            print(f'Epoch: {t+1} of {self.epochs}', end='\r')
            model.train()
            loss_per_batch = []
            for batch, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)

                # Compute prediction error
                pred = model(x)
                loss = self.loss_fn(pred, y)
                loss_per_batch.append(loss)

                # Backpropagation
                # We have to reinitialize the optimizer, since we can have new parameters during runtime.
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss_in_epoch = sum(loss_per_batch) / len(loss_per_batch)
            if print_loss:
                print(mean_loss_in_epoch)
            loss_per_epoch.append(mean_loss_in_epoch)

        self.model = model

    def get_dataloader(self, points: np.array, occupancy: np.array):
        occupancy = occupancy.reshape((len(occupancy), 1))
        points_tensor = torch.Tensor(points.T)
        occupancy_tensor = torch.Tensor(occupancy)
        dataset = TensorDataset(points_tensor, occupancy_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def new_model(self):
        new_model = copy.deepcopy(self)
        return new_model
