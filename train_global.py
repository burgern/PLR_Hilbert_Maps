import numpy as np
import torch
from PLR_Hilbert_Maps.utils import device_setup
from PLR_Hilbert_Maps.models import MlpLocal2D, FCN
from PLR_Hilbert_Maps.data import DataGeneratorRoom, GridManager

torch.autograd.set_detect_anomaly(True)


def main():
    # experiment name
    exp_name = "global_occ_1000_free_5000"

    # set up device
    device = device_setup()

    # load/generate data
    data_generator = DataGeneratorRoom(length=10, n_occ=1000, n_free=5000,
                                       noise_mean=0, noise_std=0.1, train_split=1.0)
    X = data_generator.X_train
    y = np.expand_dims(data_generator.y_train, axis=1)

    # grid management
    cell_width, cell_height = 2, 2
    gm = GridManager(X, y, cell_width, cell_height, exp_name)
    data_generator.save_visualization(gm.exp_path)

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

    gm.gm_save()


if __name__ == "__main__":
    main()
