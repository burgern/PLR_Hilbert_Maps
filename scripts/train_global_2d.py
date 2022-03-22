import numpy as np
import torch
from PLR_Hilbert_Maps.utils import device_setup
from PLR_Hilbert_Maps.models import MlpLocal2D, FCN
from PLR_Hilbert_Maps.data import DataGeneratorRoom, GridManager
import pickle

torch.autograd.set_detect_anomaly(True)


def main():
    # set up device
    device = device_setup()

    # load/generate data
    data_generator = DataGeneratorRoom(length=10, n_occ=1000, n_free=1000,
                                       noise_mean=0, noise_std=0.1, train_split=1.0)
    X = data_generator.X_train
    y = np.expand_dims(data_generator.y_train, axis=1)

    # grid management
    exp_name = "global_v001"  # experiment name
    cell_width, cell_height = 2, 2
    gm = GridManager(X, y, cell_width, cell_height, exp_name)

    # train all models in grid
    for cell_x in range(0, gm.gm_shape[0]):
        for cell_y in range(0, gm.gm_shape[1]):
            # nn specs
            model = MlpLocal2D().to(device)
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
            batch_size = 32
            epochs = 30

            # train local nn of current cell
            gm.train_model(cell_x, cell_y, device, model, loss_fn,
                           optimizer, batch_size, epochs)

    save_gm = open(f"{gm.exp_name}_gm", 'w')
    pickle.dump(gm, save_gm)


if __name__ == "__main__":
    main()