import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
import os


class DataGeneratorRoom:
    """
        2D Quadratic Room Random Data Generator
        Inputs:
            length: length of the quadratic room
            n_occ: number of requested occupied sample points
            n_free: number of requested non-occupied sample points
            noise_mean: mean of gaussian noise
            noise_std: standard deviation of gaussian noise
            train_split: fraction of dataset which should be training [0, 1]
    """
    def __init__(self, length, n_occ, n_free, noise_mean, noise_std, train_split):
        # input parameters
        self.length = length
        self.n_occ = n_occ
        self.n_free = n_free
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.train_split = train_split

        # data generation
        self.X_occ, self.y_occ = self.generate_occupied()
        self.X_free, self.y_free = self.generate_free()
        self.X = np.append(self.X_occ, self.X_free, axis=0)
        self.y = np.append(self.y_occ, self.y_free, axis=0)

        # data shuffling
        self.shuffle()

        # split train and evaluation dataset
        train_split_idx = int(self.train_split * self.X.shape[0])
        self.X_train, self.X_test, self.y_train, self.y_test = self.split(train_split_idx)

        # convert to pytorch format for DataLoader
        self.train, self.test = self.pytorch_format()

    def pytorch_format(self):
        tensor_train_x = torch.Tensor(self.X_train)
        tensor_test_x = torch.Tensor(self.X_test)
        tensor_train_y = torch.Tensor(self.y_train).unsqueeze(-1)
        tensor_test_y = torch.Tensor(self.y_test).unsqueeze(-1)
        train = TensorDataset(tensor_train_x, tensor_train_y)
        test = TensorDataset(tensor_test_x, tensor_test_y)
        return train, test

    def split(self, train_split_idx):
        X_train = self.X[:train_split_idx, :]
        X_test = self.X[train_split_idx:, :]
        y_train = self.y[:train_split_idx]
        y_test = self.y[train_split_idx:]
        return X_train, X_test, y_train, y_test

    def shuffle(self):
        assert len(self.X) == len(self.y), "Data generation failed, X and y to not have same size"
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def generate_occupied(self):
        """ x: [x_pos, y_pos] """
        # generate empty x and random vector for generation
        x = np.empty((self.n_occ, 2))
        x[:, :] = np.nan
        x_rand = np.random.rand(self.n_occ) * 4

        # find random point on wall and add gaussian noise
        # wall from <0, 0> to <length, 0>
        mask_1 = np.where(x_rand < 1.0)
        x[mask_1, 0] = self.length * x_rand[mask_1]
        x[mask_1, 1] = np.random.normal(self.noise_mean, self.noise_std, size=mask_1[0].size)

        # wall from <length, 0> to <length, length>
        mask_2 = np.where((x_rand >= 1.0) & (x_rand < 2.0))
        x[mask_2, 0] = np.random.normal(self.noise_mean + self.length, self.noise_std, size=mask_2[0].size)
        x[mask_2, 1] = self.length * (x_rand[mask_2] - 1)

        # wall from <length, length> to <0, length>
        mask_3 = np.where((x_rand >= 2.0) & (x_rand < 3.0))
        x[mask_3, 0] = self.length * (x_rand[mask_3] - 2)
        x[mask_3, 1] = np.random.normal(self.noise_mean + self.length, self.noise_std, size=mask_3[0].size)

        # wall from <0, length> to <0, 0'>
        mask_4 = np.where(x_rand >= 3.0)
        x[mask_4, 0] = np.random.normal(self.noise_mean, self.noise_std, size=mask_4[0].size)
        x[mask_4, 1] = self.length * (x_rand[mask_4] - 3)

        assert not np.isnan(x).any(), "there is a nan value somewhere in the datageneration"

        y = np.ones(self.n_occ)  # generate ground truth vector

        return x, y

    def generate_free(self):
        x = np.random.rand(self.n_free, 2) * self.length
        y = np.zeros(self.n_free)
        return x, y

    def visualize(self):
        plt.scatter(self.X_free[:, 0], self.X_free[:, 1], s=2)
        plt.scatter(self.X_occ[:, 0], self.X_occ[:, 1], s=2)
        plt.show()

    def save_visualization(self, exp_path):
        plt.scatter(self.X_free[:, 0], self.X_free[:, 1], s=2)
        plt.scatter(self.X_occ[:, 0], self.X_occ[:, 1], s=2)
        plt.show()
        plt.savefig(os.path.join(exp_path, "data.png"))
