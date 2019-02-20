"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""
import numpy as np
import torch

from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, filter_boxes
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

    # 1. Generate proposals from bbox deltas and shifted anchors
    im_height, im_width, im_scale_ratio = im_info.data
    height, width = scores.shape[-2:]

    # Enumerate all shifts
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # for a feature map of size (37, 56)
    # shifts is like
    # shifts
    # array([[  0,   0,   0,   0],
    #        [ 16,   0,  16,   0],
    #        [ 32,   0,  32,   0],
    #        ...,
    #        [848, 576, 848, 576],
    #        [864, 576, 864, 576],
    #        [880, 576, 880, 576]])


    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))
    # anchors
    # array([[ -84.,  -40.,   99.,   55.],
    #        [-176.,  -88.,  191.,  103.],
    #        [-360., -184.,  375.,  199.],
    #        ...,
    #        [ 844.,  496.,  931.,  671.],
    #        [ 800.,  408.,  975.,  759.],
    #        [ 712.,  232., 1063.,  935.]])
    # Now convert anchors from numpy to tensor in gpu
    anchors = torch.from_numpy(anchors).float().cuda()

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    import ipdb; ipdb.set_trace()
    bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous().view(-1, 4)

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.permute(0, 2, 3, 1).contiguous().view(-1, 1)

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_width, im_height)

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    proposals, scores = filter_boxes(proposals, scores, min_size*im_scale_ratio)

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.squeeze(1).sort(descending=True)[1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order, :]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    mask = nms(torch.cat([proposals,scores], dim=1), nms_thresh)
    proposals = proposals[mask, :]
    # scores = scores[mask, :]

    return proposals



