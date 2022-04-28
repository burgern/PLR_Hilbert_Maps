import os
from pathlib import Path
from .local_config import LOCAL_PLR

PATH_PLR = os.path.join(Path.home(), LOCAL_PLR)

# log path
PATH_LOG = os.path.join(PATH_PLR, "log")
if not os.path.exists(PATH_LOG):
    os.makedirs(PATH_LOG)
    print("Required models/log folder created for future model logs")

# config paths
PATH_CONFIG = os.path.join(PATH_PLR, 'config')
PATH_CONFIG_LOCAL_MODEL = os.path.join(PATH_CONFIG, 'local_model.ini')
PATH_CONFIG_LOCAL_HILBERT_MAP = os.path.join(PATH_CONFIG, 'local_hilbert_map.ini')
PATH_CONFIG_CELL = os.path.join(PATH_CONFIG, 'cell.ini')

GRID_MANAGER_NAME = "grid_manager.p"
