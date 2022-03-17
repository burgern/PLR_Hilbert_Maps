# TODO burgern: Implement actual loss from ref paper, which
# inherits from base loss.

from torch import nn
from .base_loss import BaseLoss


class HilbertMapLoss(BaseLoss):
    def __init__(self):
        self.loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss
