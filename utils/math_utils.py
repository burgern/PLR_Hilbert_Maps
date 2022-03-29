import math
from numpy import arange, array

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

def bresenham(start_point, end_point):
    """Returns the points on the line from start to end point.

    The points are assumed to be integers, i.e. cell coordinates.

    Args:
        start_point: the start point coordinates
        end_point: the end point coordinates

    Returns:
        list of coordinates on the line from start to end
    """
    coords = []

    dx = abs(end_point[0] - start_point[0])
    dy = abs(end_point[1] - start_point[1])
    x, y = start_point[0], start_point[1]
    sx = -1 if start_point[0] > end_point[0] else 1
    sy = -1 if start_point[1] > end_point[1] else 1
    if dx > dy:
        err = dx / 2.0
        while x != end_point[0]:
            coords.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != end_point[1]:
            coords.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    coords.append((x, y))

    return array(coords)
