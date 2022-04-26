from ..base_model import BaseModel


class LogisticRegression(BaseModel):
    """
    Logistic Regression
    TODO Description
    """
    def __init__(self, config, loss):
        super().__init__(self, loss, config["lr"], config["batch_size"], config["epochs"])
        self.weights = []

    def update(self, local_map_predicitons, occupancy):
        number_of_cells = local_map_predicitons.shape[0]
        # TODO hadzica: Implement dynamic weight allocation and update


    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
