import os
import cv2

import numpy as np
import torch

from lib.datasets.pascal_voc import Pascal_VOC
from lib.data_layer.minibatch import preprocess
from lib.data_layer.roidb import prepare_roidb
from lib.data_layer.layer import DataLayer

from lib.network.faster_rcnn import FasterRCNN
from lib.network.modules import save_net, load_net
from lib.network.proposals.bbox_transform import bbox_transform_inv, clip_boxes
from lib.network.nms import nms

from lib.utils.log_print import log_print
from lib.utils.draw_bbox import draw_bbox
from lib.utils.timer import Timer
from lib.config import cfg

rand_seed = cfg.SEED
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)


verbose = cfg.VERBOSE

#config
save_model_dir = cfg.SAVE_MODEL_DIR
model_path = cfg.DEMO_MODEL_FILE
output_dir = cfg.TEST_OUTPUT_DIR
output_file = cfg.TEST_OUTPUT_FILE

test_prob_thresh = cfg.TEST.PROB_THRESH
test_nms_thresh = cfg.TEST.NMS_THRESH

max_per_image = 300

# Dataset imdb
dataset_name = cfg.DATASET.NAME

# data_layer
if dataset_name == 'Pascal_VOC':
    imdb = Pascal_VOC("test", "2007")
prepare_roidb(imdb)
num_classes = imdb.num_classes
num_images = imdb.num_images

# network definition
net = FasterRCNN(imdb.num_classes)
net.eval()
if cfg.USE_CUDA:
    net.cuda()
if verbose:
    # print(net)
    pass

load_net(net, model_path)

# Dont uncomment this line!!!
# cudnn.benchmark = True



if verbose:
    pass


# Start training
t = Timer()
t.tic()

all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]


for i in range(num_images):
    im_path = imdb.image_path_at(i)

    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_data, im_scale = preprocess(im)
    im_info = np.array([im_data.shape[0], im_data.shape[1], im_scale], dtype=np.float32)
    im2show = im

    # forward
    t.tic()
    rcnn_cls_prob, rcnn_bbox_pred, rois = net(im_data, im_info)
    rcnn_cls_prob, rcnn_bbox_pred, rois = rcnn_cls_prob.data.cpu().numpy(), rcnn_bbox_pred.data.cpu().numpy(), rois.data.cpu().numpy()
    # Apply bounding-box regression deltas
    scores = rcnn_cls_prob
    box_deltas = rcnn_bbox_pred
    boxes = rois/im_scale
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_data.shape[1], im_data.shape[0])
    detect_time = t.toc(average=False)

    # nms
    t.tic()
    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > test_prob_thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        if cls_dets.shape[0] == 0:
            all_boxes[j][i] = cls_dets
            continue
        keep = nms(cls_dets, test_nms_thresh)
        cls_dets = cls_dets[keep, :]
        if verbose:
            cls_str = [cfg.DATASET.CLASSES[j-1]]*cls_dets.shape[0]
            im2show = draw_bbox(im2show, cls_dets[:,:4], cls_dets[:,4], cls_str)
        all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]
    nms_time = t.toc(average=False)

    # if verbose:
    #     cv2.imshow("test", im2show)
    #     cv2.waitKey(1)
    
    print('process: {:d}/{:d} image: {} detect: {:.3f}s nms: {:.3f}s'.format( i+1, num_images, os.path.basename(im_path), detect_time, nms_time))


torch.save(all_boxes, output_file)
    
print('Evaluating detections')
imdb.evaluate_detections(all_boxes, output_dir)

