"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""
import numpy as np
import torch

from .generate_anchors import generate_anchors
from .nms import nms
from ..config import cfg

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, is_train, feat_stride, anchor_scales):
    """
    Parameters
    ----------
    rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: [image_height, image_width, scale_ratios]
    cfg_key: 'TRAIN' or 'TEST'
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)

    """
    _anchors = generate_anchors(base_size=feat_stride, scales=np.array(anchor_scales, dtype=np.int64))
    _num_anchors = _anchors.shape[0]

    phase = 'TRAIN' if is_train else 'TEST'
    pre_nms_topN = cfg[phase].RPN_PRE_NMS_TOP_N
    post_nms_topN= cfg[phase].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[phase].RPN_NMS_THRESH
    min_size = cfg[phase].RPN_MIN_SIZE

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    scores = rpn_cls_prob[:, _num_anchors:, :, :]
    bbox_deltas = rpn_bbox_pred

    height, width = 

