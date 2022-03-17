import numpy as np
import math


class DataGenerator:
    """
        2D Lidar Data Generator
        We assume a lidar sitting in a quadratic room (height x width, center at <0,0>) which takes n samples
        Inputs:
            width: width of quadratic room (x-dir)
            height: height of quadratic room (y-dir)
            x_pos: x position of the lidar
            y_pos: y position of the lidar
            angular_res: lidars angular resolution in degrees
            n_occ: number of requested occupied sample points
            free_occupied_ratio: free to occupied ratio
    """
    def __init__(self, width, height, x_pos, y_pos, angular_res, n_occ, free_occupied_ratio):
        # input parameters
        self.width = width
        self.height = height
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.angular_res = math.radians(angular_res)
        self.n_occ = n_occ

        # check input boundaries
        assert abs(x_pos) < width / 2 and abs(y_pos) < height / 2, "lidar positions are outside of the room"

        # data generation
        # TODO

        # convert to pytorch format
        # TODO
