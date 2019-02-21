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
    jittered_gt_boxes = _jitter_gt_boxes(gt_easyboxes)
    zeros = torch.zeros((gt_easyboxes.shape[0] * 2, 1), dtype=gt_easyboxes.dtype)
    if cfg.USE_CUDA:
        zeros = zeros.cuda()
    all_rois = torch.cat([all_rois, gt_easyboxes[:,:4], jittered_gt_boxes[:,:4]], dim=0)

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
    overlaps = bbox_overlaps(all_rois, gt_boxes[:,:4])
    max_overlaps, gt_assignment = overlaps.max(dim=1)  # R
    labels = gt_boxes[gt_assignment, 4]
    inds = torch.arange(labels.shape[0]).long()
    if cfg.USE_CUDA:
        inds = inds.cuda()

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

    # import ipdb; ipdb.set_trace()
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_mask = (max_overlaps >= cfg.TRAIN.FG_THRESH)
    fg_mask = (torch.max(fg_mask,ignore_mask) - torch.min(fg_mask,ignore_mask))
    fg_num = fg_mask.sum().item()
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_num)
    fg_inds = inds[fg_mask]
    if cfg.USE_CUDA:
        rand_inds = torch.randperm(fg_inds.shape[0]).cuda()
    fg_inds = fg_inds[rand_inds][:fg_rois_per_this_image]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_mask = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) * (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))
    bg_mask = (torch.max(bg_mask,ignore_mask) - torch.min(bg_mask,ignore_mask))
    bg_num = bg_mask.sum().item()

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_num)
    bg_inds = inds[bg_mask]
    if cfg.USE_CUDA:
        rand_inds = torch.randperm(bg_inds.shape[0]).cuda()
    bg_inds = bg_inds[rand_inds][:bg_rois_per_this_image]

    # Select sampled values from various arrays:
    keep_inds = torch.cat([fg_inds, bg_inds])
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    import ipdb; ipdb.set_trace()
    bbox_target_data = _compute_targets( \
        rois, gt_boxes[gt_assignment[keep_inds], :4], labels)
    
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        mean = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).float()
        std  = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        if cfg.USE_CUDA:
            mean, std = mean.cuda(), std.cuda()
        targets = ((targets - mean) / std)
    return torch.cat([labels.view(-1,1), targets], dim=1)

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = torch.zeros((clss.size, 4 * num_classes)).float()
    bbox_inside_weights = torch.zeros(bbox_targets.shape).float()
    if cfg.USE_CUDA:
        bbox_targets, bbox_inside_weights = bbox_targets.cuda(), bbox_inside_weights.cuda()
    inds = torch.arange(clss.shape[0]).long()[clss > 0]
    for ind in inds:
        cls = int(clss[ind.item()])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        if cfg.USE_CUDA:
            bbox_inside_weights[ind, start:end] = torch.tensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).float().cuda()
        else:
            bbox_inside_weights[ind, start:end] = torch.tensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).float()

    return bbox_targets, bbox_inside_weights

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