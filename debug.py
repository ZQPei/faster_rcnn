import numpy as np
import torch

from faster_rcnn.datasets.pascal_voc import Pascal_VOC
from faster_rcnn.data_layer.roidb import prepare_roidb
from faster_rcnn.data_layer.layer import DataLayer
from faster_rcnn.data_layer.minibatch import preprocess

from faster_rcnn.network.faster_rcnn import FasterRCNN

from faster_rcnn.config import cfg
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# data_layer
if cfg.DATASET.NAME == 'Pascal_VOC':
    imdb = Pascal_VOC("trainval", "2007")
prepare_roidb(imdb)
data_layer = DataLayer(imdb.roidb)

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

# net(im_data, im_info, gt_boxes, gt_ishard)

import cv2
im = cv2.imread("img/test.jpg")
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_data, im_scale_ratio = preprocess(im_rgb)
im_info = np.array([*im_data.shape[:2], im_scale_ratio])
# net.eval()
dets, scores, classes = net.detect(im_data, im_info)

print(dets)

for i, det in enumerate(dets):
    det = tuple(int(x) for x in det)
    cv2.rectangle(im, det[0:2], det[2:4], (255, 205, 51), 2)
    cv2.putText(im, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                1.0, (0, 0, 255), thickness=1)
cv2.imwrite('img/out.jpg',im)
cv2.imshow('img', im)
cv2.waitKey(0)


from IPython import embed; embed()