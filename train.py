import numpy as np

from faster_rcnn.datasets.pascal_voc import Pascal_VOC
from faster_rcnn.roi_data_layer.roidb import prepare_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer

from faster_rcnn.network.faster_rcnn import FasterRCNN

from faster_rcnn.config import cfg
np.random.seed(cfg.SEED)

# data_layer
if cfg.DATASET.NAME == 'Pascal_VOC':
    imdb = Pascal_VOC("trainval", "2007")
prepare_roidb(imdb)
data_layer = RoIDataLayer(imdb.roidb)

# network definition
net = FasterRCNN(imdb.num_classes)


from IPython import embed; embed()