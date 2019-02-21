import torch
import numpy as np

import random

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

    gt_boxes = torch.from_numpy(gt_boxes[:,:4]).float()
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
    jittered_gt_boxes = _jitter_gt_boxes(gt_easyboxes)
    zeros = torch.zeros((gt_easyboxes.shape[0] * 2, 1), dtype=gt_easyboxes.dtype)
    if cfg.USE_CUDA:
        zeros = zeros.cuda()
    all_rois = torch.cat([all_rois, gt_easyboxes, jittered_gt_boxes], dim=0)

    rois_per_image = cfg.TRAIN.BATCH_SIZE
    fg_rois_per_image = int(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, gt_boxes, gt_ishard, fg_rois_per_image, rois_per_image, num_classes)

    rois = rois.view(-1, 4)
    labels = labels.view(-1, 1)
    bbox_targets = bbox_targets.view(-1, num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.view(-1, num_classes * 4)

    bbox_outside_weights = torch.tensor(bbox_inside_weights > 0).float()

    return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _sample_rois(all_rois, gt_boxes, gt_ishard, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: R x G
    overlaps = bbox_overlaps(all_rois, gt_boxes)
    max_overlaps, gt_assignment = overlaps.max(dim=1)  # R
    labels = gt_boxes[gt_assignment, 4]

    import ipdb; ipdb.set_trace()
    # preclude hard samples
    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
        gt_ishard = gt_ishard.long()
        gt_hardboxes = gt_boxes[gt_ishard.eq(1), :]
        if gt_hardboxes.shape[0] > 0:
            # R x H
            hard_overlaps = bbox_overlaps(all_rois, gt_hardboxes)
            hard_max_overlaps, _ = hard_overlaps.max(dim=1)  # R x 1
            # hard_gt_assignment = hard_overlaps.argmax(axis=0)  # H
            ignore_mask = (hard_max_overlaps >= cfg.TRAIN.FG_THRESH)

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_mask = (max_overlaps >= cfg.TRAIN.FG_THRESH)
    fg_mask = fg_mask - ignore_mask
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    return labels, rois, bbox_targets, bbox_inside_weights


def _jitter_gt_boxes(gt_boxes, jitter=0.05):
    """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    """
    jittered_boxes = torch.empty_like(gt_boxes).copy_(gt_boxes)
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    if cfg.USE_CUDA:
        width_offset = (torch.rand(jittered_boxes.shape[0]).cuda() - 0.5) * jitter * ws
        height_offset = (torch.rand(jittered_boxes.shape[0]).cuda() - 0.5) * jitter * hs
    else:
        width_offset = (torch.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
        height_offset = (torch.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes