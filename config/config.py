import configparser
from config import PATH_CONFIG
import os


def read_config(config_file, section, key):
    config = configparser.ConfigParser()
    config.read(os.path.join(PATH_CONFIG, config_file))
    try:
        ret = config[section][key]
    except KeyError:
        raise ValueError(f"{config_file} file is missing key '{key}'")
    return ret
