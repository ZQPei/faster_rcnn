import torch
import torch.nn.functional as F

def build_rpn_loss(rpn_cls_score, rpn_bbox_pred, rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights):
    # classification loss
    rpn_cls_score = rpn_cls_score.permute(0,2,3,1).contiguous().view(-1,2)
    rpn_labels = rpn_labels.view(-1)

    mask = rpn_labels.ne(-1)
    rpn_cls_score = rpn_cls_score[mask, :]
    rpn_labels = rpn_labels[mask]

    fg_cnt = rpn_labels.data.sum().item()
    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_labels)

    # box loss
    rpn_bbox_targets = rpn_bbox_targets.mul(rpn_bbox_inside_weights)
    rpn_bbox_pred = rpn_bbox_pred.mul(rpn_bbox_inside_weights)

    rpn_box_loss = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, reduction='sum') / (fg_cnt + 1e-4)

    return rpn_cross_entropy, rpn_box_loss
