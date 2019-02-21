import torch
import numpy as np

from .bbox_transform import bbox_transform_torch as bbox_transform
from .bbox import bbox_overlaps_torch as bbox_overlaps

from ...config import cfg

def proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_rois:  (1 x H x W x A, 4) [x1, y1, x2, y2]
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    gt_ishard: (G, 1) {0 | 1} 1 indicates hard
    dontcare_areas: (D, 4) [ x1, y1, x2, y2]
    _num_classes
    ----------
    Returns
    ----------
    rois: (1 x H x W x A, 4) [x1, y1, x2, y2]
    labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
    bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
    bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
    bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
    """

    gt_boxes = torch.from_numpy(gt_boxes).float()
    gt_ishard = torch.from_numpy(gt_ishard).long()
    if cfg.USE_CUDA:
        gt_boxes, gt_ishard = gt_boxes.cuda(), gt_ishard.cuda()

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format

    # Include ground-truth boxes in the set of candidate rois
    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
        assert gt_ishard.shape[0] == gt_boxes.shape[0]
        gt_easyboxes = gt_boxes[gt_ishard.ne(1), :]
    else:
        gt_easyboxes = gt_boxes

    """
    add the ground-truth to rois will cause zero loss! not good for visuallization
    """
    

    return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights