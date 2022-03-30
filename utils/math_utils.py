import math
from numpy import arange, array, transpose, sum, round, arctan2, sqrt, cos, sin, empty, concatenate, zeros, vstack
from numpy.random import uniform

def deg_to_rad(angle_deg):
    return angle_deg / 180 * math.pi

def rad_to_deg(angle_rad):
    return angle_rad / math.pi * 180

def sample_angles_around_optical_axis(angle_fov, angular_resolution):
    return arange((90 + angle_fov / 2),
           (90 - angle_fov / 2), -angular_resolution)

def sample_sensor_row_indices(width):
    if width % 2 == 0:
        return arange(-width / 2 + 0.5, width / 2 + 0.5, 1)
    else:
        return arange(-width / 2, width / 2 + 0.5, 1)

def sample_points_in_freespace_for_viewpoint(point_cloud_wrt_sensor_frame):
    r_squarred = point_cloud_wrt_sensor_frame[0] ** 2 + point_cloud_wrt_sensor_frame[1] ** 2
    sum_r_squarred = sum(r_squarred)
    number_of_points = round(r_squarred / sum_r_squarred * point_cloud_wrt_sensor_frame[0].shape[1])
    angles = arctan2(point_cloud_wrt_sensor_frame[1], point_cloud_wrt_sensor_frame[0])
    free_points_x = []
    free_points_y = []
    for r_squarred_point, number_of_points_angle in zip(r_squarred[0], zip(number_of_points[0], angles[0])):
        number_of_points_for_point, angle = number_of_points_angle
        radial_distances = sqrt(uniform(0, 1, int(number_of_points_for_point))) * sqrt(r_squarred_point)
        x = radial_distances * cos(angle)
        y = radial_distances * sin(angle)
        free_points_x.append(x)
        free_points_y.append(y)

    free_points_x = concatenate(free_points_x)
    free_points_y = concatenate(free_points_y)
    free_points_occupancy = zeros(free_points_x.shape)
    free_points = vstack([free_points_x, free_points_y, free_points_occupancy])
    return free_points

def point_cloud_WRT_World(pose, point_cloud):
    point_cloud[0] += pose[0]
    point_cloud[1] += pose[1]
    return point_cloud
