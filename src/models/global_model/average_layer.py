from ..base_model import BaseModel
from torch import nn, zeros, nanmean

class AverageLayer(nn.Module):
    """
    Logistic Regression
    TODO Description
    """

    def __init__(self):
        super(AverageLayer, self).__init__()
        scalar = zeros(1)
        self.bias = nn.Parameter(scalar)

    def forward(self, local_map_predictions, inference=False):
        return nanmean(local_map_predictions, dim=1).unsqueeze(-1) + 0. * self.bias




