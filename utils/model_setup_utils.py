from typing import Dict
from datetime import datetime

from src.hilbert_map.local_hilbert_map import LocalHilbertMap
from src.hilbert_map.local_hilbert_map_collection import \
    LocalHilbertMapCollection
from src.hilbert_map.hilbert_map import HilbertMap
from src.data import DatasetHexagon, DataIntelLab, DataFreiburg079,\
    DataFreiburgCampus
from evaluation.logger import Logger


def generate_data(config: Dict):
    if config["dataset"]["dataset"] == "dummy":
        data = DatasetHexagon(config=config)
    elif config["dataset"]["dataset"] == "intel_lab":
        data = DataIntelLab(config=config)
    elif config["dataset"]["dataset"] == "freiburg_079":
        data = DataFreiburg079(config=config)
    elif config["dataset"]["dataset"] == "freiburg_campus":
        data = DataFreiburgCampus(config=config)
    else:
        raise ValueError
    return data


def create_model(config: Dict):
    if config["model"]["model"] == "lhm":
        model = LocalHilbertMap(config=config,
                                center=(config["model"]["lhm"]["center_x"],
                                        config["model"]["lhm"]["center_y"]))
    elif config["model"]["model"] == "lhmc":
        model = LocalHilbertMapCollection(config=config)
    elif config["model"]["model"] == "hm":
        model = HilbertMap(config)
    else:
        raise ValueError
    return model


def set_up_logger(exp_name: str, config: Dict):
    if not exp_name:
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        flag = input("[LOGGER] do you want to log your experiment?\n" +
                     f"[LOGGER] it would be logged in log/{date}\n" +
                     "[LOGGER] YES: hit enter\n" +
                     "[LOGGER] NO:  enter anything")
        if flag:
            print("[LOGGER] no experiment name given, " +
                  "hence nothing will be logged")
            return None
        else:
            print("[LOGGER] experiment will be logged - cheers")
            exp_name = date

    return Logger(exp_name=exp_name, config=config)
