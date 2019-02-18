import os
import os.path as osp
import numpy as np
from easydict import EasyDict

cfg = EasyDict()

cfg.SEED = 1024

# Directories of project
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
cfg.DATA_DIR = osp.join(cfg.ROOT_DIR, 'data')

# Dataset
cfg.DATASET_NAME = 'Pascal_VOC'

# Preprocess =================================================================================
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# During train ================================================================================
cfg.TRAIN = EasyDict() 

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
cfg.TRAIN.SCALE = 600
# Max pixel size of the longest side of a scaled input image
cfg.TRAIN.MAX_SIZE = 1000
# Use RPN to detect objects, of course
cfg.TRAIN.HAS_RPN = True
# Images to use per minibatch. if using RPN, then IMS_PER_BATCH has to be set to 1
cfg.TRAIN.IMS_PER_BATCH = 1
assert cfg.TRAIN.HAS_RPN and cfg.TRAIN.IMS_PER_BATCH is 1, "Single batch only when has RPN"
# Minibatch size (number of regions of interest [ROIs])  --> minibatch is a batch contained of RoIs
cfg.TRAIN.BATCH_SIZE = 128

# During test =================================================================================



if __name__ == "__main__":
    import ipdb; ipdb.set_trace()