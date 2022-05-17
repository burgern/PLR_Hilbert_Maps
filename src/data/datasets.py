import os
from config import PATH_DATA
from src.data.dataset import Dataset
from src.utils.data_utils import parse_carmen_log


class DataIntelLab(Dataset):
    def __init__(self):
        self.data_path = os.path.join(PATH_DATA, "raddish", "intel.gfs.log")
        fov_deg = 180
        poses, scans = parse_carmen_log(self.data_path)
        invalid_scan_dist = 20
        super().__init__(poses=poses, scans=scans, fov_deg=fov_deg,
                         invalid_scan_dist=invalid_scan_dist)


class DataFreiburg079(Dataset):
    def __init__(self):
        self.data_path = os.path.join(PATH_DATA, "raddish",
                                      "fr079-complete.gfs.log")
        fov_deg = 180
        poses, scans = parse_carmen_log(self.data_path)
        invalid_scan_dist = 10
        super().__init__(poses=poses, scans=scans, fov_deg=fov_deg,
                         invalid_scan_dist=invalid_scan_dist)


class DataFreiburgCampus(Dataset):
    def __init__(self):
        self.data_path = os.path.join(PATH_DATA, "raddish",
                                      "fr-campus-20040714.carmen.gfs.log")
        fov_deg = 180
        poses, scans = parse_carmen_log(self.data_path)
        invalid_scan_dist = 70
        super().__init__(poses=poses, scans=scans, fov_deg=fov_deg,
                         invalid_scan_dist=invalid_scan_dist)