from torch.utils.data import Dataset
import json

class BaseDataSet(Dataset):
    def __init__(self, config):
        self.sensor_config = json.load(config["sensor_type_path"])
        self.data_config = json.load(config["data_config_path"])
        self.learning_method_config = json.load(config["learning_method_config_path"])

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
