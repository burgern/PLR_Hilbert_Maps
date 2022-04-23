from src.data.base_dataset import BaseDataSet
from src.utils.data_utils import get_all_files_from_directory, get_depth_samples_and_pose_data_from_replica_data_folder
from os.path import join
from numpy import arange, loadtxt, ones, vstack, load, sin, cos, concatenate
from src.utils.math_utils import sample_angles_around_optical_axis, sample_points_in_freespace_for_viewpoint, \
    point_cloud_WRT_World, deg_to_rad

class ReplicaDataSet(BaseDataSet):
    def __init__(self, config):
        super().__init__(config)
        self.sensor_fov = self.sensor_config["theta_fov_deg"]
        self.sensor_W = self.sensor_config["W_px"]
        self.sensor_angular_resolution = self.sensor_fov / self.sensor_W
        self.data_type = self.data_config["type"]
        self.data_folder_path = self.data_config["path"]
        files = get_all_files_from_directory(self.data_folder_path)
        if self.data_type == "2d_replica":
            self.depth_samples, pose_data_path = get_depth_samples_and_pose_data_from_replica_data_folder(files)
            self.pose_data = loadtxt(join(self.data_folder_path, pose_data_path), delimiter=",")
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
            with open(depth_sample_path, "rb") as f:
                depth_data = load(f, allow_pickle=True)
            sensor_pixel_angles = sample_angles_around_optical_axis(self.sensor_fov, self.sensor_angular_resolution)
            sensor_pixel_angles = deg_to_rad(sensor_pixel_angles)
            x = depth_data * cos(sensor_pixel_angles)
            y = depth_data * sin(sensor_pixel_angles)
            points_in_free_space = sample_points_in_freespace_for_viewpoint([x, y])
            pose_data = self.pose_data[item, :] * self.square_unit_m
            p = ones(x.shape)
            point_cloud_data = vstack([x, y, p])
            point_cloud_data = point_cloud_WRT_World([pose_data[1], pose_data[0]], point_cloud_data)
            points_in_free_space = point_cloud_WRT_World([pose_data[1], pose_data[0]], points_in_free_space)
            point_cloud_data = concatenate((point_cloud_data, points_in_free_space), axis=1)
        else:
            raise NotImplementedError
        # TODO hadzcamir: Sample and add non occupied points.
        return point_cloud_data
