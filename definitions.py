import os
from dataclasses import dataclass

@dataclass(frozen=True)
class PROJECT_VARS:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    print('root dir : ', ROOT_DIR)
    CONFIG_PATH = os.path.join(ROOT_DIR, 'configuration.conf') 