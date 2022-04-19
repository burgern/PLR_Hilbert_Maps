import numpy as np
import torch
from PLR_Hilbert_Maps.src.utils import device_setup
from PLR_Hilbert_Maps.src.models import MlpLocal2D
from PLR_Hilbert_Maps.src.data import DataGeneratorRoom, GridManager

torch.autograd.set_detect_anomaly(True)


def main():
    # experiment name
    exp_name = "global_occ_v005_tanshrink_2000_free_4000"

    # set up device
    device = device_setup()

    # load/generate data
    data_generator = DataGeneratorRoom(length=10, n_occ=2000, n_free=4000,
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
            loss_fn = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
            batch_size = 32
            epochs = 30

            # train local nn of current cell
            gm.train_model(cell_x, cell_y, device, model, loss_fn,
                           optimizer, batch_size, epochs)

    gm.gm_save()


if __name__ == "__main__":
    main()
