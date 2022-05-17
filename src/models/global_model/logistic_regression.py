from ..base_model import BaseModel
from torch import nn, zeros, cat, matmul, nan_to_num, randn, sigmoid, ones, sum, isnan, unique
import matplotlib.pyplot as plt

class LogisticRegression(nn.Module):
    """
    Logistic Regression
    TODO Description
    """

    def __init__(self):
        super(LogisticRegression, self).__init__()
        scalar = randn(1)
        self.bias = nn.Parameter(scalar)
        self.weights_pool = nn.Parameter(randn(100))
        self.weights_pool_counter = 0
        self.weights = []
        self.weight_history = []

    def forward(self, local_map_predictions, inference=False):
        number_of_cells_in_input = local_map_predictions.shape[1]
        self.allocate_model_weights(number_of_cells_in_input)
        linear_combination = self.calculate_linear_combination(local_map_predictions)
        if inference:
            return sigmoid(linear_combination)
        else:
            return linear_combination

    def allocate_model_weights(self, number_of_cells_in_input):
        number_of_new_cells_to_allocate = number_of_cells_in_input - len(self.weights)
        if number_of_new_cells_to_allocate > 0:
            for _ in range(number_of_new_cells_to_allocate):
                self.weights.append(self.weights_pool[self.weights_pool_counter].unsqueeze(0))
                self.weights_pool_counter += 1

    def calculate_linear_combination(self, local_map_predictions):
        weights = cat(self.weights)
        self.weight_history.append(weights.unsqueeze(-1))
        number_of_valid_predictions_per_point = sum(isnan(local_map_predictions) == False, dim=1)
        occuring_numbers = unique(number_of_valid_predictions_per_point)
        linear_combinations = zeros(local_map_predictions.shape[0])
        for occuring_number in occuring_numbers:
            idx_considered_points = number_of_valid_predictions_per_point == occuring_number
            considered_predictions = local_map_predictions[idx_considered_points, :]
            number_of_considered_points = considered_predictions.shape[0]
            idx_non_nan_condsidered_points = isnan(considered_predictions) == False
            non_nan_considered_predictions = considered_predictions[idx_non_nan_condsidered_points]\
                .reshape(number_of_considered_points, occuring_number)
            repeated_weights = weights.unsqueeze(0).repeat(number_of_considered_points, 1)
            relevant_weights_for_points_with_complete_set = repeated_weights[idx_non_nan_condsidered_points]\
                .reshape(number_of_considered_points, occuring_number)
            linear_combination = sum(non_nan_considered_predictions * \
                             relevant_weights_for_points_with_complete_set, dim=1) + self.bias
            linear_combinations[idx_considered_points] = linear_combination
        return linear_combinations.unsqueeze(-1)

    def plot_weights_history(self):
        weights_history_tensor = cat(self.weight_history, dim=1)
        number_of_packages_of_ten = weights_history_tensor.shape[0] // 10 + 1
        weight_idx = 0
        for idx_ten in range(number_of_packages_of_ten):
            legend = []
            for weight_history in weights_history_tensor[idx_ten * 10:(idx_ten + 1) * 10]:
                plt.plot(weight_history.detach().numpy())
                legend.append(weight_idx)
                weight_idx += 1
            plt.legend(legend)
            plt.show()



