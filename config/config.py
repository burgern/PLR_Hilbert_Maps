import configparser
from config import PATH_CONFIG
import os
import json


def read_config(config_file, section, key):
    config = configparser.ConfigParser()
    config.read(os.path.join(PATH_CONFIG, config_file))
    try:
        ret = config[section][key]
    except KeyError:
        raise ValueError(f"{config_file} file is missing key '{key}'")
    return ret


def load_config(config_file: str = "test_hilbert_map_config.json"):
    config_file = os.path.join(PATH_CONFIG, config_file)
    with open(config_file) as file:
        config = json.load(file)
    return config
