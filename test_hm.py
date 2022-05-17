from src.hilbert_map.hilbert_map import HilbertMap
from src.data.dataset_elements import DatasetHexagon
import argparse

def main(args):
    hilbert_map = HilbertMap(args.config)
    n = 100000
    size = 5
    data = DatasetHexagon(n, size, (0, 0))
    hilbert_map.update(data.points, data.occupancy)
    hilbert_map.plot(1001)
    hilbert_map.local_map_collection.plot(1001)
    hilbert_map.global_map.model.plot_weights_history()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the replica dataset class")
    parser.add_argument('--config', action="store")

    args = parser.parse_args()
    main(args)
