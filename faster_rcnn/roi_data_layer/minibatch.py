"""Compute minibatch blob for training a Faster R-CNN network."""
import cv2
import os

import numpy as np
from ..config import cfg

def get_minibatch(minibatch_db):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(minibatch_db)
    assert num_images is 1, "Single image in a minibatch only"

    roidb = minibatch_db[0]
    im = cv2.imread(roidb['image'])
    im_data, im_scale = preprocess(im, cfg.PIXEL_MEANS, cfg.TRAIN.SCALE, cfg.TRAIN.MAX_SIZE)

    blob = {'im_data': im_data}
    # gt boxes: (x1, y1, x2, y2, cls)
    gt_boxes = np.empty((roidb['num_objs'], 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb['boxes']*im_scale
    gt_boxes[:, 4] = roidb['gt_classes']
    blob['gt_boxes'] = gt_boxes

    blob['gt_ishard'] = roidb['gt_ishard']
    blob['im_info'] = np.array([im_data.shape[1], im_data.shape[2], im_scale], dtype=np.float32)
    blob['im_name'] = os.path.basename(roidb['image'])

    return blob

def preprocess(im, pixel_means, target_size, max_size):
    """Scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    # im -= pixel_means
    im_height, im_width = im.shape[:2]
    im_size_max, im_size_min = (im_height, im_width) if im_height>im_width else (im_width, im_height)
    im_scale = float(target_size) / im_size_min
    # Pervent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale*im_size_max) > max_size:
        im_scale = float(max_size) / im_size_max
    im_data = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR) 
    # im_data = np.expand_dims(im_data, axis=0)
    return im_data, im_scale