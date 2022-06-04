import torch.nn as nn
import numpy as np
from torch import randn, ones, cat, matmul, sum, sigmoid, nan_to_num

class MLP_global(nn.Module):
    """
    MLP global
    TODO Description
    """

    def __init__(self):
        super(MLP_global, self).__init__()
        scalar = randn(1)
        self.bias = nn.Parameter(scalar)
        self.weights_pool = nn.Parameter(randn(100000000))
        self.weights_pool_counter = 0
        self.weight_matrix = None

    def forward(self, local_map_predictions: np.array):
        number_of_cells_in_input = local_map_predictions.shape[1]
        self.adapt_weight_matrix(number_of_cells_in_input)
        output = matmul(self.weight_matrix, nan_to_num(local_map_predictions.transpose(1, 0)))
        output = sum(output, dim=0) + self.bias
        return sigmoid(output.unsqueeze(-1))

    def adapt_weight_matrix(self, number_of_cells_in_input: int):
        if self.weight_matrix is None:
            number_of_required_weights = number_of_cells_in_input * number_of_cells_in_input
            weights_to_allocate = self.weights_pool[:number_of_required_weights]
            self.weight_matrix = weights_to_allocate.reshape(number_of_cells_in_input, number_of_cells_in_input)
            self.weights_pool_counter = number_of_cells_in_input * number_of_cells_in_input

        else:
            number_of_rows_cols_to_allocate = number_of_cells_in_input - self.weight_matrix.shape[0]
            if number_of_rows_cols_to_allocate > 0:
                number_of_col_weights = self.weight_matrix.shape[0] * number_of_rows_cols_to_allocate
                cols_to_allocate = self.weights_pool[self.weights_pool_counter:
                                                     self.weights_pool_counter + number_of_col_weights]
                self.weights_pool_counter += number_of_col_weights
                cols_to_allocate = cols_to_allocate.reshape(self.weight_matrix.shape[0], number_of_rows_cols_to_allocate)
                self.weight_matrix = cat((self.weight_matrix, cols_to_allocate), dim=1)
                number_of_row_weights = self.weight_matrix.shape[1] * number_of_rows_cols_to_allocate
                rows_to_allocate = self.weights_pool[self.weights_pool_counter:
                                                     self.weights_pool_counter + number_of_row_weights]
                self.weights_pool_counter += number_of_row_weights
                rows_to_allocate = rows_to_allocate.reshape(number_of_rows_cols_to_allocate, self.weight_matrix.shape[1])
                self.weight_matrix = cat((self.weight_matrix, rows_to_allocate), dim=0)
