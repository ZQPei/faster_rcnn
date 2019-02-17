import os
import os.path as osp
from easydict import EasyDict

cfg = EasyDict()

# Directories of project
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
cfg.DATA_DIR = osp.join(cfg.ROOT_DIR, 'data')

# 



if __name__ == "__main__":
    import ipdb; ipdb.set_trace()