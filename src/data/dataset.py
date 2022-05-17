from typing import List, Tuple
import numpy as np
import math
from matplotlib.axes import Axes


class Dataset:
    """ Dataset class generates required assets
    :param poses the poses (x, y, theta)
    :param scans the distance for each scan per viewpoint
    :param fov the field of view
    """
    def __init__(self, poses: List[List], scans: List[List], fov_deg: float,
                 invalid_scan_dist: float):
        # read inputs
        assert len(poses) == len(scans), \
            "[INPUT ERROR] poses and scans must have same length"
        self.poses = np.array(poses)
        self.scans = np.array(scans)
        self.fov_deg = fov_deg
        self.invalid_scan_dist = invalid_scan_dist

        # required variables
        self.fov_rad = math.radians(fov_deg)
        self.scans_per_vp = len(scans[0])
        self.nr_viewpoints = len(poses)
        self.current_viewpoint = 0
        self.data = []

        # generate data
        #   get occupancy list from scans
        #   generate free spaces
        # preprocess data
        #   shuffle data
        #   requested number of viewpoints
        #   train / test split
        # logger
        #   log info about requested data

        self.generate_data()

    def generate_data(self):
        """ Generate required data from inputs (occlusions and free spaces) """
        angles_rad = np.linspace(- self.fov_rad / 2, self.fov_rad / 2,
                                 self.scans_per_vp)
        for index, (pose, scan) in enumerate(zip(self.poses, self.scans)):
            # compute occlusion points from scan and pose
            points_occ_x = scan * np.cos(angles_rad + pose[2]) + pose[0]
            points_occ_y = scan * np.sin(angles_rad + pose[2]) + pose[1]
            occupancy = np.ones(self.scans_per_vp)
            points_occ = np.vstack((points_occ_x, points_occ_y))

            # generate free space

            # remove scans out of range
            valid_scans_mask = scan <= self.invalid_scan_dist
            points_occ = points_occ[:, valid_scans_mask]
            occupancy = occupancy[valid_scans_mask]

            data_step = {"points": points_occ,
                         "occupancy": occupancy,
                         "pose": {"position": (pose[0], pose[1]),
                                  "angle": pose[2]}}

            self.data.append(data_step)

    def visualize(self, ax: Axes, step_size: int = 1):
        for i in np.arange(0, self.nr_viewpoints, step_size):
            ax.scatter(self[i]["points"][0, :], self[i]["points"][1, :],
                       s=1, c="blue")
            ax.scatter(self[i]["pose"]["position"][0],
                       self[i]["pose"]["position"][1], s=2, c="green")

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> List[Tuple]:
        if self.current_viewpoint >= self.nr_viewpoints:
            self.current_viewpoint = 0
            raise StopIteration
        return self.data[self.current_viewpoint]
