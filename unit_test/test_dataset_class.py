import argparse
from src.data.replica_dataset import ReplicaDataSet
import json
import matplotlib.pyplot as plt
from numpy.random import randint
from numpy import concatenate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the replica dataset class")
    parser.add_argument('--config', action="store")
    parser.add_argument('--number_of_viewpoints', action="store")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    dataset = ReplicaDataSet(config)

    idx = randint(dataset.__len__(), size=int(args.number_of_viewpoints))

    pc_room = []
    for id in idx:
        pc_scan = dataset.__getitem__(id)
        pc_room.append(pc_scan)
    pc_room = concatenate(pc_room, axis=1)
    plt.scatter(pc_room[0], pc_room[1], c=pc_room[2], s=0.01)
    plt.show()







