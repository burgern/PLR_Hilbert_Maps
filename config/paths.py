import os
from pathlib import Path


def create_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# get local path to PLR project
LOCAL_PLR = Path(__file__).resolve().parent.parent
PATH_PLR = os.path.join(Path.home(), LOCAL_PLR)

# useful paths for reference
PATH_LOG = os.path.join(PATH_PLR, "log")
PATH_LOG_DATA = os.path.join(PATH_LOG, "dataset")
PATH_LOG_DATASET = os.path.join(PATH_LOG, "dataset.pkl")
PATH_LOG_EXP = os.path.join(PATH_LOG, "experiments")
PATH_DATA = os.path.join(PATH_PLR, "dataset")
PATH_CONFIG = os.path.join(PATH_PLR, 'config')

# create required folders
req_folders = [PATH_LOG, PATH_LOG_DATA, PATH_LOG_EXP]
for folder_path in req_folders:
    create_folder(folder_path)
