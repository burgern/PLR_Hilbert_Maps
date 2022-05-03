from .paths import *
from .local_config import *
from .config import read_config

__project_path__ = read_config('config.ini', 'Setup', 'project_path')
__dataset_path__ = read_config('config.ini', 'Setup', 'dataset_path')
