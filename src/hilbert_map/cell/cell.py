from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from src.utils.data_utils import concatenate_ones
import matplotlib.patches as patches
from typing import Optional
import copy


class Cell(ABC):
    """
    Cell
    TODO Description
    """
    def __init__(self, center: Optional[Tuple[float, float]],
                 r1: Tuple[float, float], r2: Tuple[float, float], nx: float,
                 ny: float, patch_edgecolor: str, patch_linewidth: float):
        # read inputs
        self.center = np.array([center[0], center[1]]) if center is not None \
            else None
        self.r1 = np.array([r1[0], r1[1]])
        self.r2 = np.array([r2[0], r2[1]])
        self.nx = nx
        self.ny = ny
        self.patch_edgecolor = patch_edgecolor
        self.patch_linewidth = patch_linewidth

        # compute magnitudes
        self.r1_mag = np.linalg.norm(self.r1)
        self.r2_mag = np.linalg.norm(self.r2)

        if center is not None:
            self.normalization_mat, self.normalization_mat_inv = \
                self.non_template_operations()


    @abstractmethod
    def is_point_in_cell(self, points: np.array) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def patch(self) -> patches:
        raise NotImplementedError

    def non_template_operations(self):
        # compute matrices for switching between original and normalized points
        normalization_mat = self.compute_normalization_mat()
        normalization_mat_inv = np.linalg.inv(normalization_mat)
        return normalization_mat, normalization_mat_inv

    def new_cell(self, center: Tuple[float, float]):
        new_cell = copy.deepcopy(self)
        new_cell.center = self.center = np.array([center[0], center[1]])
        new_cell.normalization_mat, new_cell.normalization_mat_inv = \
            new_cell.non_template_operations()
        return new_cell

    def compute_normalization_mat(self):
        # translation matrix
        translation_mat = np.eye(3)
        translation_mat[:2, 2] = - self.center

        # rotation matrix
        rotation_mat = np.eye(3)
        rotation_mat[0, :2] = self.r1 / self.r1_mag
        rotation_mat[1, :2] = self.r2 / self.r2_mag

        # scaling matrix
        scaling_mat = np.eye(3)
        scaling_mat[0, 0] = self.nx / self.r1_mag
        scaling_mat[1, 1] = self.ny / self.r2_mag

        # normalization matrix
        normalization_mat = np.matmul(scaling_mat, np.matmul(rotation_mat,
                                                             translation_mat))
        return normalization_mat

    def original_to_normalized(self, points_original: np.array):
        x = concatenate_ones(points_original, 0)
        return self.normalization_mat.dot(x)[:2, :]

    def normalized_to_original(self, points_normalized: np.array):
        x = concatenate_ones(points_normalized, 0)
        return self.normalization_mat_inv.dot(x)[:2, :]
