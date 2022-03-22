import torch
from torch.utils.data import DataLoader
from PLR_Hilbert_Maps.utils import device_setup
from PLR_Hilbert_Maps.models import MlpLocal2D, FCN
from PLR_Hilbert_Maps.training import train, test
from PLR_Hilbert_Maps.data import DataGeneratorRoom


torch.autograd.set_detect_anomaly(True)


def main():
    # Set up device
    device = device_setup()

    # Load data
    batch_size = 32
    data_generator = DataGeneratorRoom(length=10, n_occ=1000, n_free=1000,
                                       noise_mean=0, noise_std=0.1, train_split=0.9)
    train_dataloader = DataLoader(data_generator.train, batch_size=batch_size)
    test_dataloader = DataLoader(data_generator.test, batch_size=batch_size)

    # Load model
    model = MlpLocal2D().to(device)
    print(model)

    # Loss and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    # train
    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, device, loss_fn, optimizer)
        test(test_dataloader, model, device, loss_fn)
    print("Done!")

    # debugger
    # print(model.parameters())
    # print(list(model.parameters())[0].grad)

    # save model
    model_path = "model_mlp_2"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to : {model_path}")


if __name__ == "__main__":
    main()
