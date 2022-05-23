from torch import nn

class MLP(nn.Module):
    """
        Neural Net for local map representation within 2 dimensions
    """
    def __init__(self, config):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        n_inputs = config["n_inputs"]
        n_outputs = config["n_outputs"]
        w_hidden_layers = config["w_hidden_layers"]
        self.model.append(nn.Linear(n_inputs, w_hidden_layers))
        for _ in range(config["n_hidden_layers"]):
            self.model.append(nn.Linear(w_hidden_layers, w_hidden_layers))
        self.model.append(nn.Linear(w_hidden_layers, n_outputs))

        if config["activation_input_layer"] == "Tanhshrink":
            self.input_activation = nn.Tanhshrink()
        else:
            raise ValueError

        if config["activation_hidden_layers"] == "Tanhshrink":
            self.hidden_activation = nn.Tanhshrink()
        else:
            raise ValueError

        if config["activation_output_layer"] == "Sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            raise ValueError

    def forward(self, x):
        x = self.model[0](x)
        x = self.input_activation(x)
        for hidden_layer in self.model[1:-1]:
            x = hidden_layer(x)
            x = self.hidden_activation(x)
        x = self.model[-1](x)
        x = self.output_activation(x)
        return x
