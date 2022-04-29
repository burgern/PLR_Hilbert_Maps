from ..base_model import BaseModel
from torch import nn, FloatTensor, randn, cat, matmul


class LogisticRegression(BaseModel):
    """
    Logistic Regression
    TODO Description
    """
    def __init__(self, config, loss):
        super().__init__(self, loss, config["lr"], config["batch_size"], config["epochs"])
        self.weights = []

    def forward(self, local_map_predictions):
        weights = cat(self.weights)
        return matmul(weights, local_map_predictions)

    def update(self, local_map_predicitons, occupancy):
        number_of_cells = local_map_predicitons.shape[0]
        number_of_new_cells_to_allocate = number_of_cells - len(self.weights)
        if number_of_new_cells_to_allocate > 0:
            for _ in range(number_of_new_cells_to_allocate):
                scalar = FloatTensor(1)
                self.weights.append(nn.Parameter(randn(1), out=scalar))
        self.train(local_map_predicitons, occupancy)

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
