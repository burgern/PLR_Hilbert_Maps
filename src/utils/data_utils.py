from os import listdir
from os.path import isfile, join
import numpy as np

def get_all_files_from_directory(path_to_directory):
    files = [f for f in listdir(path_to_directory) if isfile(join(path_to_directory, f))]
    files.sort()
    return files

def get_depth_samples_and_pose_data_from_replica_data_folder(files_of_replica_data_folder):
    pose_data_file = None
    depth_samples = []
    for file in files_of_replica_data_folder:
        if "txt" in file:
            pose_data_file = file
        elif "pkl" in file:
            depth_samples.append(file)
    return depth_samples, pose_data_file

def concatenate_ones(x: np.array, axis: int) -> np.array:
    if axis == 0:
        return np.concatenate((x, np.ones((1, x.shape[1]), dtype=x.dtype)), axis=axis)
    elif axis == 1:
        return np.concatenate((x, np.ones((x.shape[0], 1), dtype=x.dtype)), axis=axis)
    else:
        raise ValueError

def concatenate_zeros(x: np.array, axis: int) -> np.array:
    if axis == 0:
        return np.concatenate((x, np.zeros((1, x.shape[1]), dtype=x.dtype)), axis=axis)
    elif axis == 1:
        return np.concatenate((x, np.zeros((x.shape[0], 1), dtype=x.dtype)), axis=axis)
    else:
        raise ValueError


def parse_carmen_log(fname):
    """Parses a CARMEN log file and extracts poses and laser scans.

    :param fname the path to the log file to parse
    :return poses and scans extracted from the log file
    """
    poses = []
    scans = []
    for line in open(fname):
        if line.startswith("FLASER"):
            arr = line.split()
            count = int(arr[1])
            poses.append([float(v) for v in arr[-9:-6]])
            scans.append([float(v) for v in arr[2:2 + count]])

    return poses, scans
