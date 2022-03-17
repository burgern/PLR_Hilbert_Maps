import torch
from PLR_Hilbert_Maps.utils import device_setup
from PLR_Hilbert_Maps.models import MlpLocal2D
from PLR_Hilbert_Maps.loss import HilbertMapLoss
from PLR_Hilbert_Maps.training import train, test


def main():
    # Set up device
    device = device_setup()

    # Load data
    # TODO

    # Load model
    model = MlpLocal2D().to(device)
    print(model)

    # Loss and optimizer
    loss_fn = HilbertMapLoss().loss_fn
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train
    # epochs = 5
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train(train_dataloader, model, device, loss_fn, optimizer)
    #     test(test_dataloader, model, device, loss_fn)
    # print("Done!")


if __name__ == "__main__":
    main()
