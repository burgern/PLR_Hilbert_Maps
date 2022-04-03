from torch import nn


class MLP(nn.Module):
    """
        Neural Net for local map representation within 2 dimensions
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 32, bias=True),
            nn.Tanhshrink(),
            nn.Linear(32, 32, bias=True),
            nn.Tanhshrink(),
            nn.Linear(32, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
