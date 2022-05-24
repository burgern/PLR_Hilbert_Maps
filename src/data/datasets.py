import os
from config import PATH_DATA
from src.data.dataset import Dataset
from src.utils.data_utils import parse_carmen_log
from typing import Optional, Dict


def load_from_config(config: Dict):
    shuffle_seed = config["shuffle_seed"]
    skip_vp = config["skip_vp"]
    tot_viewpoints = config["tot_viewpoints"]
    invalid_scan_dist = config["invalid_scan_dist"]
    return shuffle_seed, skip_vp, tot_viewpoints, invalid_scan_dist


class DataIntelLab(Dataset):
    def __init__(self, config: Optional[Dict] = None,
                 shuffle_seed: Optional[int] = None, skip_vp: int = 1,
                 tot_viewpoints: Optional[int] = None,
                 invalid_scan_dist: float = 20):
        if config is not None:
            config_intel = config["dataset"]["intel_lab"]
            shuffle_seed, skip_vp, tot_viewpoints, invalid_scan_dist = \
                load_from_config(config_intel)
        self.data_path = os.path.join(PATH_DATA, "raddish", "intel.gfs.log")
        fov_deg = 180
        poses, scans = parse_carmen_log(self.data_path)
        super().__init__(poses=poses, scans=scans, fov_deg=fov_deg,
                         invalid_scan_dist=invalid_scan_dist,
                         shuffle_seed=shuffle_seed, skip_vp=skip_vp,
                         tot_viewpoints=tot_viewpoints)


class DataFreiburg079(Dataset):
    def __init__(self, config: Optional[Dict] = None,
                 shuffle_seed: Optional[int] = None, skip_vp: int = 1,
                 tot_viewpoints: Optional[int] = None,
                 invalid_scan_dist: float = 10):
        if config is not None:
            config_intel = config["dataset"]["freiburg_079"]
            shuffle_seed, skip_vp, tot_viewpoints, invalid_scan_dist = \
                load_from_config(config_intel)
        self.data_path = os.path.join(PATH_DATA, "raddish",
                                      "fr079-complete.gfs.log")
        fov_deg = 180
        poses, scans = parse_carmen_log(self.data_path)
        super().__init__(poses=poses, scans=scans, fov_deg=fov_deg,
                         invalid_scan_dist=invalid_scan_dist,
                         shuffle_seed=shuffle_seed, skip_vp=skip_vp,
                         tot_viewpoints=tot_viewpoints)


class DataFreiburgCampus(Dataset):
    def __init__(self, config: Optional[Dict] = None,
                 shuffle_seed: Optional[int] = None, skip_vp: int = 1,
                 tot_viewpoints: Optional[int] = None,
                 invalid_scan_dist: float = 70):
        if config is not None:
            config_intel = config["dataset"]["freiburg_campus"]
            shuffle_seed, skip_vp, tot_viewpoints, invalid_scan_dist = \
                load_from_config(config_intel)
        self.data_path = os.path.join(PATH_DATA, "raddish",
                                      "fr-campus-20040714.carmen.gfs.log")
        fov_deg = 180
        poses, scans = parse_carmen_log(self.data_path)
        super().__init__(poses=poses, scans=scans, fov_deg=fov_deg,
                         invalid_scan_dist=invalid_scan_dist,
                         shuffle_seed=shuffle_seed, skip_vp=skip_vp,
                         tot_viewpoints=tot_viewpoints)