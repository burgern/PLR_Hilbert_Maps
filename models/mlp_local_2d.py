# TODO burgern: Implement local MLP based 2D hilbert map here H_L.
# H_L: (x, z) -> P_occupied, where
# (x, z) € L := [x_0, x_1] x [z_0, z_1], where
# x_0, x_1, z_0 and z_1 are boundaries, which
# restrict H_L. Therefore, it holds
#
#                    { H(x, z), if (x, z) € L
# P_occupied(x, y) = {
#                    { 0, otherwise

#            x --------- X y
#                        |
#                        |
#                        | z
#
#   lidar coordinate system

from torch import nn


class MlpLocal2D(nn.Module):
    """
        Neural Net for local map representation within 2 dimensions
    """
    def __init__(self):
        super(MlpLocal2D, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

