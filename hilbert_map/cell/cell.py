from abc import ABC, abstractmethod


class Cell(ABC):
    """
    Cell
    TODO Description
    """
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def is_point_in_cell(self):
        raise NotImplementedError
