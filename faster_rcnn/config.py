import os
import os.path as osp
import numpy as np
from easydict import EasyDict

cfg = EasyDict()

cfg.SEED = 1024

# CUDA config
cfg.USE_CUDA = True
cfg.CUDA_VISIBLE_DEVICES = 0

# Directories of project
cfg.SPECIFIC_NAME = "voc_resnet_0218"
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
cfg.DATA_DIR = osp.join(cfg.ROOT_DIR, 'data')
cfg.SAVE_MODEL_DIR = osp.join(cfg.ROOT_DIR, 'models', cfg.SPECIFIC_NAME)
if not osp.exists(cfg.SAVE_MODEL_DIR):
    os.makedirs(cfg.SAVE_MODEL_DIR)

# Dataset
cfg.DATASET = EasyDict()

cfg.DATASET.NAME = 'Pascal_VOC'
# cfg.DATASET.NUM_CLASSES = 20
# cfg.DATASET.CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
#                 'bottle', 'bus', 'car', 'cat', 'chair',
#                 'cow', 'diningtable', 'dog', 'horse',
#                 'motorbike', 'person', 'pottedplant',
#                 'sheep', 'sofa', 'train', 'tvmonitor')

# Preprocess =================================================================================
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]]) # <== This is BGR, set by Ross Girshick in his code
# To use pytorch official pretrained models, we should do the normalization by its mean and std used in training pretrained models
cfg.MEAN = [0.485, 0.456, 0.406]
cfg.STD  = [0.229, 0.224, 0.225]

# Network config ==============================================================================
cfg.NETWORK = EasyDict()

# Basic Network to get feature map
cfg.NETWORK.BASIC_NETWORK = "vgg16"
cfg.NETWORK.BASIC_NETWORK_OUTCHANNELS = 512
cfg.NETWORK.RPN_CONV_OUTCHANNELS = 512
cfg.NETWORK.RCNN_FC_OUTCHANNELS = 4096

# Set anchor scales and ratios
cfg.NETWORK.NUM_ANCHORS = 9
cfg.NETWORK.NUM_ANCHOR_SCALES = 3
cfg.NETWORK.NUM_ANCHOR_RATIOS = 3
# This is the feature stride size of a feature map. Feature stride doubled for each maxpooling layer
# e.g. VGG16's feature stride is 16 (2^4) for it's 4 maxpooling layer
cfg.NETWORK.FEATURE_STRIDE = 16 
cfg.NETWORK.ANCHOR_SCALES = [8, 16, 32]
cfg.NETWORK.ANCHOR_RATIOS = [0.5, 1., 2.]
# With feature stride 16 and anchor scales [8, 16, 32] and anchor ratios [0.5, 1, 2], the basic anchors should be as follows. 
# See generate_anchors.py for more detail.
#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

# RoI pooling layer config
cfg.NETWORK.ROI_POOLED_SIZE = 7

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