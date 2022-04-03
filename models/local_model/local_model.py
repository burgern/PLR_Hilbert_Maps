from PLR_Hilbert_Maps.config import PATH_CONFIG_LOCAL_MODEL
from PLR_Hilbert_Maps.utils import device_setup
from .mlp import MLP

from configparser import ConfigParser
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim


class LocalModel:
    """
    Local Model
    TODO Description
    """
    def __init__(self, model: nn.Module = MLP(), loss_fn: nn.modules.loss = nn.BCELoss(), lr: float = 1e-3,
                 batch_size: int = None, epochs: int = None):
        # read config.ini file for default values
        config = ConfigParser()
        config.read(PATH_CONFIG_LOCAL_MODEL)

        # read inputs
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.batch_size = batch_size if batch_size is not None else config.getint('General', 'batch_size')
        self.epochs = epochs if epochs is not None else config.getint('General', 'epochs')

        self.device = device_setup()  # device setup (cpu or gpu)

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def train(self, points: np.array, occupancy: np.array):
        dataloader = self.get_dataloader(points, occupancy)  # get data in required pytorch format
        model = self.model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # train model
        for t in range(self.epochs):
            print(f'Epoch: {t+1} of {self.epochs}', end='\r')
            model.train()
            for batch, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)

                # Compute prediction error
                pred = model(x)
                loss = self.loss_fn(pred, y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def get_dataloader(self, points: np.array, occupancy: np.array):
        occupancy = occupancy.reshape((len(occupancy), 1))
        points_tensor = torch.Tensor(points.T)
        occupancy_tensor = torch.Tensor(occupancy)
        dataset = TensorDataset(points_tensor, occupancy_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader
