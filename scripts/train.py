import torch
from torch.utils.data import DataLoader
from PLR_Hilbert_Maps.utils import device_setup
from PLR_Hilbert_Maps.models import MlpLocal2D
from PLR_Hilbert_Maps.loss import HilbertMapLoss
from PLR_Hilbert_Maps.training import train, test
from PLR_Hilbert_Maps.data import DataGeneratorRoom


torch.autograd.set_detect_anomaly(True)


def main():
    # Set up device
    device = device_setup()

    # Load data
    batch_size = 32
    data_generator = DataGeneratorRoom(length=10, n_occ=1000, n_free=4000,
                                       noise_mean=0, noise_std=0.1, train_split=1.0)
    train_dataloader = DataLoader(data_generator.train, batch_size=batch_size)
    test_dataloader = DataLoader(data_generator.test, batch_size=batch_size)
    data_generator.visualize()  # visualize created data

    # Load model
    model = MlpLocal2D().to(device)
    print(model)

    # Loss and optimizer
    loss_fn = HilbertMapLoss().loss_fn
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train
    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, device, loss_fn, optimizer)
        test(test_dataloader, model, device, loss_fn)
    print("Done!")

    # save model
    model_path = "model_mlp"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to : {model_path}")


if __name__ == "__main__":
    main()
