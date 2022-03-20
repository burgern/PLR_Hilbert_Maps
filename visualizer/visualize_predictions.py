from PLR_Hilbert_Maps.data import DataGeneratorRoom
from PLR_Hilbert_Maps.models import MlpLocal2D, FCN
import torch
import matplotlib.pyplot as plt


class VisualizePredictions:
    def __init__(self):
        # load data
        data_generator = DataGeneratorRoom(length=10, n_occ=1000, n_free=4000,
                                           noise_mean=0, noise_std=0.1, train_split=1.0)
        self.train_data = data_generator.train

        # load model
        model = MlpLocal2D()
        #model = FCN(2, 32, 1)
        model.load_state_dict(torch.load("../scripts/model_mlp"))

        # make predictions
        model.eval()
        X = self.train_data[:][0]
        with torch.no_grad():
            #pred = model(X).type(torch.float)
            pred = model(X)
        X_np = X.cpu().detach().numpy()
        pred_np = pred.cpu().detach().numpy()
        print(pred)
        print(pred_np)

        # visualize
        self.visualize(X_np, pred_np)

    def visualize(self, X, pred):
        plt.scatter(X[:, 0], X[:, 1], s=2, c=pred, cmap='viridis')
        plt.colorbar()
        plt.show()


a = VisualizePredictions()
