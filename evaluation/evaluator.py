from config import PATH_LOG
import os


class Evaluator:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.exp_path = os.path.join(PATH_LOG, exp_name)

    def visualize