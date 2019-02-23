import os
import os.path as osp
import numpy as np
from easydict import EasyDict

cfg = EasyDict()

cfg.SEED = 1024

cfg.DEBUG = False
cfg.VERBOSE = True



# CUDA config
cfg.USE_CUDA = True
cfg.CUDA_VISIBLE_DEVICES = 0

# import torch
# cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Directories of project
cfg.SPECIFIC_NAME = "vgg16_0223_without_init"
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
cfg.DATA_DIR = osp.join(cfg.ROOT_DIR, 'data')
cfg.SAVE_MODEL_DIR = osp.join(cfg.ROOT_DIR, 'models', cfg.SPECIFIC_NAME)
if not osp.exists(cfg.SAVE_MODEL_DIR):
    os.makedirs(cfg.SAVE_MODEL_DIR)

cfg.DEMO_IMAGE_DIR = osp.join(cfg.ROOT_DIR, 'img/demo')
cfg.ROUND = 80000
cfg.MODEL_NAME = 'voc_resnet_0218/faster_rcnn_{}.pkl'.format(cfg.ROUND)
cfg.DEMO_MODEL_FILE = osp.join(cfg.ROOT_DIR, 'models', cfg.MODEL_NAME)
cfg.DEMO_THRESH = 0.3

# Dataset
cfg.DATASET = EasyDict()

cfg.DATASET.NAME = 'Pascal_VOC'
cfg.DATASET.NUM_CLASSES = 20
cfg.DATASET.CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')

cfg.OFFICIAL = True
# Preprocess =================================================================================
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]]) # <== This is BGR, set by Ross Girshick in his code
# To use pytorch official pretrained models, we should do the normalization by its mean and std used in training pretrained models
if cfg.OFFICIAL:
    cfg.MEAN = [122.7717, 115.9465, 102.9801]
    cfg.STD  = [1.0, 1.0, 1.0]
else:
    cfg.MEAN = [0.485, 0.456, 0.406]
    cfg.STD  = [0.229, 0.224, 0.225]


# Network config ==============================================================================
cfg.NETWORK = EasyDict()


# Basic Network to get feature map
# cfg.NETWORK.BASIC_NETWORK = "vgg16"
# cfg.NETWORK.BASIC_NETWORK_OUTCHANNELS = 512
# cfg.NETWORK.RPN_CONV_OUTCHANNELS = 512
# cfg.NETWORK.RCNN_FC_OUTCHANNELS = 4096
cfg.NETWORK.BASIC_NETWORK = "resnet18"
cfg.NETWORK.BASIC_NETWORK_OUTCHANNELS = 256
cfg.NETWORK.RPN_CONV_OUTCHANNELS = 512
cfg.NETWORK.RCNN_FC_OUTCHANNELS = 4096

# Set anchor scales and ratios
cfg.NETWORK.NUM_ANCHORS = 9
cfg.NETWORK.NUM_ANCHOR_SCALES = 3
cfg.NETWORK.NUM_ANCHOR_RATIOS = 3
# This is the feature stride size of a feature map. Feature stride is the downsampling ratio of feature map to the original input image.
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


# During train phase ================================================================================
cfg.TRAIN = EasyDict() 

# Config
cfg.TRAIN.START_STEP = 0
cfg.TRAIN.END_STEP = 100001
cfg.TRAIN.MILESTONE = [40000, 60000, 80000]
cfg.TRAIN.OPTIMIZER = 'SGD'
cfg.TRAIN.LEARNING_RATE = 0.001
cfg.TRAIN.LEARNING_RATE_DECAY = 0.1
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 5e-4
cfg.TRAIN.DAMPENING = 0
cfg.TRAIN.LOG_INTERVAL = 10
cfg.TRAIN.GAMMA = 0.2



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
cfg.TRAIN.BATCH_SIZE = 300
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
cfg.TRAIN.FG_FRACTION = 0.3
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
cfg.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
cfg.TRAIN.BG_THRESH_HI = 0.5
cfg.TRAIN.BG_THRESH_LO = 0.0

cfg.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
cfg.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
cfg.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)



# RPN 
cfg.TRAIN.RPN_BATCH_SIZE = 256

# NMS config during train phase
# NMS threshold used on RPN proposals
cfg.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
cfg.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
cfg.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
cfg.TRAIN.RPN_MIN_SIZE = 16

cfg.TRAIN.PRECLUDE_HARD_SAMPLES = True
# IOU >= thresh: positive example
cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
cfg.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# Max number of foreground examples
cfg.TRAIN.RPN_FG_FRACTION = 0.5
cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# Set to -1.0 to use uniform example weighting
cfg.TRAIN.RPN_POSITIVE_WEIGHT = -1.0


# During test phase =================================================================================
cfg.TEST = EasyDict()

# NMS config during test phase
# NMS threshold used on RPN proposals
cfg.TEST.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
cfg.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
cfg.TEST.RPN_MIN_SIZE = 16

# Overlap threshold used for non-maximum suppression in RCNN in test phase
cfg.TEST.RCNN_NMS_THRESH = 0.3


if __name__ == "__main__":
    import ipdb; ipdb.set_trace()