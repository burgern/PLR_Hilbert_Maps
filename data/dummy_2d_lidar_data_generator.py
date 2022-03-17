import numpy as np
import math

lidar_fov = (1 + 1 / 3) * math.pi  # 240 degrees
half_fov_behind = (lidar_fov - math.pi) / 2  # 30 degrees
lidar_end = (1 + half_fov_behind) * math.pi  # why rad * rad?
lidar_start = - half_fov_behind  # - 30 degrees
lidar_angular_resolution = 0.5 / 180 * math.pi
number_of_samples = int(lidar_fov / lidar_angular_resolution)  # 480 samples
max_view_distance = 2

# Sample occupied points
phi_lidar_angles = np.arange(lidar_start, lidar_end, lidar_angular_resolution)
# min = 1m, max = 3m
random_radial_distances = max_view_distance * np.random.rand(number_of_samples) + 1
x_occupied = random_radial_distances * np.cos(phi_lidar_angles)  # OUTPUTS ERROR
z_occupied = random_radial_distances * np.sin(phi_lidar_angles)
p_occupied = np.ones(number_of_samples)

occupied = np.array([[x_occupied], [z_occupied], [p_occupied]])
# Sample empty points
# min = 0, max = 1m
random_radial_distances = np.random.rand(number_of_samples)
x_free = random_radial_distances * np.cos(phi_lidar_angles)
z_free = random_radial_distances * np.sin(phi_lidar_angles)
p_free = np.zeros(number_of_samples)
free = np.array([[x_free], [z_free], [p_free]])
# Final data
data = np.cat((occupied, free), axis=1)

print(data)