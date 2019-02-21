import numpy as np
import torch

def bbox_transform_np(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_torch(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 torch tensor, given boxes
    :param gt_rois: n * 4 torch tensor, ground-truth boxes
    :return: deltas: n * 4 torch tensor, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh), dim=1).contiguous()
    return targets

def bbox_transform_inv_torch(boxes, deltas):
    """
    Input:
        boxes shape: (N, 4) float32
        deltas shape: (N, 4) float32

        pred_cx = cx + dx*w
        pred_cy = cy + dy*h
        pred_w  = w + exp(dw)
        pred_h  = h + exp(dh)
    Return:
        pred boxes
    """
    box_width = (boxes[:,2] - boxes[:,0] + 1.)
    box_height= (boxes[:,3] - boxes[:,1] + 1.)
    box_cx = (boxes[:,0] + 0.5*box_width)
    box_cy = (boxes[:,1] + 0.5*box_height)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_cx = dx.mul(box_width).add(box_cx)
    pred_cy = dy.mul(box_height).add(box_cy)
    pred_w = torch.exp(dw).mul(box_width)
    pred_h = torch.exp(dh).mul(box_height)

    pred_x1 = pred_cx - 0.5*pred_w
    pred_y1 = pred_cy - 0.5*pred_h
    pred_x2 = pred_cx + 0.5*pred_w
    pred_y2 = pred_cy + 0.5*pred_h
    pred_boxes = torch.stack([pred_x1,pred_y1,pred_x2,pred_y2], dim=1)

    return pred_boxes

bbox_transform_inv = bbox_transform_inv_torch

def clip_boxes(boxes, im_width, im_height):
    """
    Clip boxes to image boundaries.
    """
    boxes[boxes<0] = 0
    boxes[:,0][boxes[:,0]>im_width] = im_width
    boxes[:,1][boxes[:,1]>im_height] = im_height
    boxes[:,2][boxes[:,2]>im_width] = im_width
    boxes[:,3][boxes[:,3]>im_height] = im_height
    return boxes

def filter_boxes(boxes, scores, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1

    mask = ((ws >= min_size)*(hs >= min_size))
    boxes = boxes[mask, :]
    scores = scores[mask, :]
    return boxes, scores