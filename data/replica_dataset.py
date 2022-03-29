from data.base_dataset import BaseDataSet
from utils.data_utils import get_all_files_from_directory, get_depth_samples_and_pose_data_from_replica_data_folder
from os.path import join
from numpy import arange, tan, loadtxt, ones, vstack, load
from utils.math_utils import sample_sensor_row_indices

class ReplicaDataSet(BaseDataSet):
    def __init__(self, config):
        super().__init__(config)
        self.sensor_fov = self.sensor_config["theta_fov_deg"]
        self.sensor_W = self.sensor_config["W_px"]
        self.sensor_focal_length = self.sensor_W / (2 * tan(self.sensor_fov / 2))
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
            sensor_row_indices = sample_sensor_row_indices(self.sensor_W)
            x = depth_data * sensor_row_indices / self.sensor_focal_length
            y = depth_data
            pose_data = self.pose_data[item, :] * self.square_unit_m
            x = x + pose_data[1]
            y = y + pose_data[0]
            p = ones(x.shape)
            point_cloud_data = vstack([x, y, p])
        else:
            raise NotImplementedError
        # TODO hadzcamir: Sample and add non occupied points.
        return point_cloud_data
