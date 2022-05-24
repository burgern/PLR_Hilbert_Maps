import random
from typing import List, Tuple, Optional
import numpy as np
import math
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from src.utils.evaluation_utils import timeit
import time


class Dataset:
    """ Dataset class generates required assets
    :param poses the poses (x, y, theta)
    :param scans the distance for each scan per viewpoint
    :param fov the field of view
    :param invalid_scan_dist cutoff distance for scan rays out of range
    :param shuffle_seed given a seed, it will shuffle the viewpoints
    :param skip_vp every x (e.g. 2nd/3rd/...) viewpoint is loaded
    :param tot_viewpoints if given a number, only first x viewpoints are loaded
    """
    def __init__(self, poses: List[List], scans: List[List], fov_deg: float,
                 invalid_scan_dist: float, shuffle_seed: Optional[int] = None,
                 skip_vp: int = 1, tot_viewpoints: Optional[int] = None):
        # read inputs
        assert len(poses) == len(scans), \
            "[INPUT ERROR] poses and scans must have same length"
        self.poses = np.array(poses)
        self.scans = np.array(scans)
        self.fov_deg = fov_deg
        self.invalid_scan_dist = invalid_scan_dist
        self.shuffle_seed = shuffle_seed
        self.tot_viewpoints = tot_viewpoints
        self.skip_vp = skip_vp

        # useful variables
        self.fov_rad = math.radians(fov_deg)
        self.scans_per_vp = len(scans[0])
        self.current_viewpoint = 0
        self.data = []
        self.nr_occ_points, self.nr_free_points, self.total_points = 0, 0, 0
        self.generation_runtime = 0
        self.angles_rad = np.linspace(- self.fov_rad / 2, self.fov_rad / 2,
                                      self.scans_per_vp)

        self.generate_data()  # generate occupancy and free spaces from scan
        self.preprocess_data()  # preprocessing step on generated data
        self.evaluate_data()  # evaluate generated and preprocessed data

    def data_concatenated(self):
        points = np.empty((2, 0))
        occupancy = np.empty(0)
        for vp in self.data:
            points = np.hstack((points, vp["points"]))
            occupancy = np.concatenate((occupancy, vp["occupancy"]))
        return points, occupancy

    def preprocess_data(self):
        """ preprocess loaded data
                -> shuffle
                -> skip viewpoints
                -> crop number of total viewpoints
        """
        if self.shuffle_seed is not None:
            random.Random(self.shuffle_seed).shuffle(self.data)
        if self.skip_vp != 1:
            assert self.skip_vp >= 1, "skip_vp argument must be larger than 1"
            self.data = self.data[::self.skip_vp]
        if self.tot_viewpoints is not None:
            assert len(self.data) >= self.tot_viewpoints, \
                "tot_viewpoints set too large"
            self.data = self.data[0: self.tot_viewpoints]

    @timeit
    def generate_data(self):
        """ Generate required data from inputs (occlusions and free spaces) """
        start_time = time.perf_counter()
        for index, (pose, scan) in enumerate(zip(self.poses, self.scans)):
            # compute occlusion points from scan and pose
            points_occ_x = scan * np.cos(self.angles_rad + pose[2]) + pose[0]
            points_occ_y = scan * np.sin(self.angles_rad + pose[2]) + pose[1]
            points_occ = np.vstack((points_occ_x, points_occ_y))
            occupancy_occ = np.ones(self.scans_per_vp)

            # remove scans out of range
            valid_scans_mask = scan <= self.invalid_scan_dist
            points_occ = points_occ[:, valid_scans_mask]
            occupancy_occ = occupancy_occ[valid_scans_mask]

            # generate free space
            points_free, occupancy_free = \
                self.create_free_space(pose, scan, valid_scans_mask)

            # combine and format
            points = np.hstack((points_occ, points_free))
            occupancy = np.hstack((occupancy_occ, occupancy_free))
            data_step = {"points": points,
                         "occupancy": occupancy,
                         "pose": {"position": (pose[0], pose[1]),
                                  "angle": pose[2]}}
            self.nr_occ_points += len(occupancy_occ)
            self.nr_free_points += len(occupancy_free)
            self.data.append(data_step)
        end_time = time.perf_counter()
        self.generation_runtime = end_time - start_time

    def create_free_space(self, pose: np.array, scan: np.array, mask: np.array):
        """ Generates free spaces for each ray in every scan
        :param pose is given as [x_pos, y_pos, ...]
        :param scan are the scanned ray distances of the selected viewpoint
        :param mask gives us information about which rays to use / not to use
        :return numpy array (2, len(scan)) of free points generated
        """
        points_free_x, points_free_y = np.empty(0), np.empty(0)
        counts = np.ones(self.scans_per_vp, dtype=np.int)
        counts[scan.astype(np.int) >= 1] = \
            scan[scan.astype(np.int) >= 1].astype(np.int)

        # generate free spaces for valid rays
        for index, (dist, count, valid) in enumerate(zip(scan, counts, mask)):
            if valid:
                r = np.sqrt(np.random.random(count)) * dist
                points_free_x = np.append(points_free_x,
                                          r * np.cos(self.angles_rad[index] + pose[2])
                                          + pose[0])
                points_free_y = np.append(points_free_y,
                                          r * np.sin(self.angles_rad[index] + pose[2])
                                          + pose[1])
        points_free = np.vstack((points_free_x, points_free_y))
        occupancy = np.zeros(points_free.shape[1])

        return points_free, occupancy

    def visualize_viewpoint(self, index: int, ax: Optional[Axes] = None):
        if ax is None:
            plt.scatter(self[index]["points"][0, :], self[index]["points"][1, :],
                       c=self[index]["occupancy"], cmap="coolwarm", s=1)
            plt.scatter(self[index]["pose"]["position"][0],
                       self[index]["pose"]["position"][1], s=4, c="green")
            plt.show()
        else:
            ax.scatter(self[index]["points"][0, :], self[index]["points"][1, :],
                       c=self[index]["occupancy"], cmap="coolwarm", s=1)
            ax.scatter(self[index]["pose"]["position"][0],
                       self[index]["pose"]["position"][1], s=4, c="green")

    def visualize(self, ax: Axes, step_size: Optional[int] = 1):
        for i in np.arange(0, len(self.data), step_size):
            self.visualize_viewpoint(ax=ax, index=i)

    def evaluate_data(self):
        # TODO not correct evaluation when using flags for preprocessing data
        self.total_points = self.nr_occ_points + self.nr_free_points
        nr_viewpoints = len(self.data)
        occ_points = self.nr_occ_points
        free_points = self.nr_free_points
        total_points = self.total_points
        avg_occ_per_vp = occ_points / nr_viewpoints
        avg_free_per_vp = free_points / nr_viewpoints
        avg_points_per_vp = total_points / nr_viewpoints
        free_to_occ_ratio = free_points / occ_points
        print("/"*80)
        print(f"Dataset:\t{type(self).__name__}")
        print(f"Generation:\t{self.generation_runtime:.3} s")
        print(f"Viewpoints:\t{len(self.data)}")
        print(f"# Occ:\t{occ_points}")
        print(f"# Free:\t{free_points}")
        print(f"# Points:\t{total_points}")
        print(f"Av. Occ / VP:\t{avg_occ_per_vp:.3}")
        print(f"Av. Free / VP:\t{avg_free_per_vp:.3}")
        print(f"Av. Points / VP:\t{avg_points_per_vp:.3}")
        print(f"Free : Occ ratio:\t{free_to_occ_ratio:.3}")
        print("/" * 80)

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.array, np.array, None]:
        if self.current_viewpoint >= len(self.data):
            self.current_viewpoint = 0
            raise StopIteration
        data = self.data[self.current_viewpoint]
        self.current_viewpoint += 1
        return data["points"], data["occupancy"], None
