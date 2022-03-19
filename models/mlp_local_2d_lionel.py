import torch


class FCN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.nonlinearity = torch.nn.Tanhshrink

        self._hidden_block = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            self.nonlinearity(),
        )
        self._layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            self.nonlinearity(),
            self._hidden_block,
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        return self._layers(data)