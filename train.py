import numpy as np
import torch

from faster_rcnn.datasets.pascal_voc import Pascal_VOC
from faster_rcnn.roi_data_layer.roidb import prepare_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.roi_data_layer.minibatch import preprocess

from faster_rcnn.network.faster_rcnn import FasterRCNN

from faster_rcnn.config import cfg
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# data_layer
if cfg.DATASET.NAME == 'Pascal_VOC':
    imdb = Pascal_VOC("trainval", "2007")
prepare_roidb(imdb)
data_layer = RoIDataLayer(imdb.roidb)

# network definition
net = FasterRCNN(imdb.num_classes)
net.train()
if cfg.USE_CUDA:
    net.cuda()

inputs = data_layer.forward()
im_data = inputs['im_data']
im_info = inputs['im_info']
gt_boxes = inputs['gt_boxes']
gt_ishard = inputs['gt_ishard']

rcnn_cls_prob, rcnn_bbox_pred, rois = net(im_data, im_info, gt_boxes, gt_ishard)

import cv2
im = cv2.imread("img/test.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_data, im_scale_ratio = preprocess(im)
im_info = np.array([*im_data.shape[:2], im_scale_ratio])
net.eval()
res = net.detect(im_data, im_info)

from IPython import embed; embed()