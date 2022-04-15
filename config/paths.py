import os
from pathlib import Path

LOCAL_PLR = "/home/amirhadzic/PLR_Hilbert_Maps"
PATH_PLR = os.path.join(Path.home(), LOCAL_PLR)
PATH_LOG = os.path.join(PATH_PLR, "log")

# check whether PATH_MODELS already exists
if not os.path.exists(PATH_LOG):
    os.makedirs(PATH_LOG)
    print("Required models/log folder created for future model logs")

GRID_MANAGER_NAME = "grid_manager.p"
