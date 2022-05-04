from ..base_model import BaseModel
from torch import nn, zeros, cat, matmul, nan_to_num, randn, sigmoid

class LogisticRegression(nn.Module):
    """
    Logistic Regression
    TODO Description
    """
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.weights = []
        scalar = randn(1)
        self.bias = nn.Parameter(scalar)

    def forward(self, local_map_predictions):
        number_of_cells_in_input = local_map_predictions.shape[1]
        self.allocate_model_weights(number_of_cells_in_input)
        weights = cat(self.weights)
        linear_combination = matmul(nan_to_num(local_map_predictions), weights) + self.bias
        return sigmoid(linear_combination).unsqueeze(-1)

    def allocate_model_weights(self, number_of_cells_in_input):
        number_of_new_cells_to_allocate = number_of_cells_in_input - len(self.weights)
        if number_of_new_cells_to_allocate > 0:
            for _ in range(number_of_new_cells_to_allocate):
                scalar = randn(1)
                self.weights.append(nn.Parameter(scalar))


