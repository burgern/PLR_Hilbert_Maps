from torch.utils.data import Dataset
import json

class BaseDataSet(Dataset):
    def __init__(self, config):
        with open(config["sensor_config_path"]) as f:
            self.sensor_config = json.load(f)
        with open(config["data_config_path"]) as f:
            self.data_config = json.load(f)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
