from ..base_model import BaseModel
from torch import nn, FloatTensor, randn, cat, matmul

class LogisticRegression(nn.Module):
    """
    Logistic Regression
    TODO Description
    """
    def __init__(self, config, loss):
        self.weights = []

    def forward(self, local_map_predictions):
        number_of_cells = local_map_predictions.shape[0]
        number_of_new_cells_to_allocate = number_of_cells - len(self.weights)
        if number_of_new_cells_to_allocate > 0:
            for _ in range(number_of_new_cells_to_allocate):
                scalar = FloatTensor(1)
                self.weights.append(nn.Parameter(randn(1), out=scalar))
        weights = cat(self.weights)
        return matmul(weights, local_map_predictions)

