from data.base_dataset import BaseDataSet
from utils.data_utils import get_all_files_from_directory, get_depth_samples_and_pose_data_from_replica_data_folder
from os.path import join
from pickle import load
from numpy import arange, cos, sin, loadtxt, ones, vstack
from math import pi

class ReplicaDataSet(BaseDataSet):
    def __init__(self, config):
        super.__init__(config)
        self.sensor_type = self.sensor_config["type"]
        self.sensor_fov = self.sensor_config["fov"]
        self.data_type = self.data_config["type"]
        self.data_folder_path = self.data_type["path"]
        files = get_all_files_from_directory(self.data_folder_path)
        if self.data_type == "2d_replica":
            self.depth_samples, pose_data_path = get_depth_samples_and_pose_data_from_replica_data_folder(files)
            self.pose_data = loadtxt(join(self.data_folder_path, pose_data_path))
            self.square_unit_m = self.data_config["square_unit_m"]
        else:
            raise NotImplementedError

    def __len__(self):
        if self.data_type == "2d_replica":
            return len(self.depth_samples)
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        if self.data_type == "2d_replica":
            depth_sample_path = join(self.data_folder_path, self.depth_samples[item])
            depth_data = load(depth_sample_path)
            sensor_angular_resolution = self.sensor_fov["horizontal"] / depth_data.shape[1]
            sensor_sampling_angles = arange((pi + self.sensor_fov["horizontal"]) / 2,
                                            (pi - self.sensor_fov["horizontal"]) / 2, -sensor_angular_resolution)
            x = depth_data * cos(sensor_sampling_angles)
            y = depth_data * sin(sensor_sampling_angles)
            pose_data = self.pose_data[item, :] * self.square_unit_m
            x += pose_data[1]
            y += pose_data[0]
            p = ones(x.shape[0])
            point_cloud_data = vstack([x, y, p])
        else:
            raise NotImplementedError
        # TODO hadzcamir: Sample and add non occupied points.
        return point_cloud_data
