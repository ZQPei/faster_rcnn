import torch
import torch.nn.functional as F

def build_rcnn_loss(rcnn_cls_score, rcnn_bbox_pred, rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    # classification loss
    fg_cnt = torch.sum(labels.data.ne(0)).item()
    bg_cnt = labels.data.numel() - fg_cnt
    import ipdb; ipdb.set_trace()
    ce_weights = torch.ones_like(rcnn_cls_score[0]).float()
    ce_weights[0] = 1. *fg_cnt / bg_cnt
    
    labels = labels.squeeze()
    rcnn_cross_entropy = F.cross_entropy(rcnn_cls_score, labels, weight=ce_weights.detach())

    # bounding box regression L1 loss
    bbox_targets = bbox_targets.mul(bbox_inside_weights)
    rcnn_bbox_pred = rcnn_bbox_pred.mul(bbox_inside_weights)

    rcnn_box_loss = F.smooth_l1_loss(rcnn_bbox_pred, bbox_targets, reduction='sum') / (fg_cnt + 1e-4)

    return rcnn_cross_entropy, rcnn_box_loss
