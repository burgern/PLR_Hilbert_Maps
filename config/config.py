from .paths import PATH_CONFIG
import os
import json


def load_config(config_file: str = "config.json"):
    config_file = os.path.join(PATH_CONFIG, config_file)
    with open(config_file) as file:
        config = json.load(file)
    return config
