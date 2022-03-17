from PLR_Hilbert_Maps.utils import device_setup
from PLR_Hilbert_Maps.models import MlpLocal2D


def main():
    device = device_setup()
    model = MlpLocal2D().to(device)
    print(model)


if __name__ == "__main__":
    main()