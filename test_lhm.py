from src.hilbert_map import Square, LocalHilbertMap
from src.models import *
from torch import nn
from src.data.replica_dataset import ReplicaDataSet
from config import PATH_CONFIG
import os
import json
import matplotlib.pyplot as plt


def main():
    # load dataset
    config_path = os.path.join(PATH_CONFIG, 'test_config.json')
    with open(config_path) as f:
        config = json.load(f)
    data = ReplicaDataSet(config)

    # lhmc setup
    cell = Square(center=(8.5, 3.5), width=6, nx=0.5, ny=0.5)
    local_model = BaseModel(MLP(), nn.BCELoss(), lr=0.02, batch_size=32, epochs=32)
    lhm = LocalHilbertMap(cell, local_model)

    # update data
    data_step = data.__getitem__(10)
    points, occupancy = data_step[:2, :], data_step[2, :]

    # update lhmc
    lhm.update(points, occupancy)

    plt.close('all')
    plt.scatter(points[0, :], points[1, :], c=occupancy, s=1, cmap='cool')
    plt.show()
    lhm.plot(10, 10, 1001)

    lhm.evaluate(points, occupancy)


if __name__ == "__main__":
    main()
