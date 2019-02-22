import torch
import torch.nn as nn

import numpy as np

from .modules import *

from .proposals.proposal_layer import proposal_layer
from .proposals.anchor_target_layer import anchor_target_layer

from .nms import nms

from ..loss_function.rpn_loss import build_rpn_loss

from ..utils.timer import Timer, tic, toc
from ..config import cfg


class RPN(nn.Module):
    """RPN network
    """
    def __init__(self, in_channels, out_channels, sliding_window_size=3):
        super(RPN, self).__init__()

        self.feature_stride = cfg.NETWORK.FEATURE_STRIDE
        self.anchor_scales = cfg.NETWORK.ANCHOR_SCALES
        self.anchor_ratios = cfg.NETWORK.ANCHOR_RATIOS

        self.conv = Conv2d(in_channels, out_channels, kernel_size=sliding_window_size, stride=1, same_padding=True)
        self.num_anchors = cfg.NETWORK.NUM_ANCHORS
        self.cls_conv = Conv2d(out_channels, self.num_anchors*2, 1, relu=False, same_padding=False)
        self.bbox_conv  = Conv2d(out_channels, self.num_anchors*4, 1, relu=False, same_padding=False)
        # weight_init(self.conv)
        # weight_init(self.cls_conv)
        # weight_init(self.bbox_conv)

        self.use_cuda = cfg.USE_CUDA

        # loss
        self.rpn_cls_loss = None
        self.rpn_box_loss = None

    @property
    def loss(self):
        return self.rpn_cls_loss + self.rpn_box_loss*10
        
    def forward(self, feature_map, im_info, gt_boxes=None, gt_ishard=None):
        
        x = self.conv(feature_map)

        # rpn cls prob
        rpn_cls_score = self.cls_conv(x)
        rpn_cls_prob = self.rpn_score_to_prob_softmax(rpn_cls_score)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(x)
        
        # proposal layer to RCNN network as input
        # import ipdb; ipdb.set_trace()
        rois = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info)

        # generating training labels and build the rpn loss
        if self.training:
            feature_map_size = list(feature_map.data.shape[-2:])
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                self.anchor_target_layer(feature_map_size, gt_boxes, gt_ishard, im_info)
            
            self.rpn_cls_loss, self.rpn_box_loss = \
                build_rpn_loss(rpn_cls_score, rpn_bbox_pred, rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)

        return rois

    @staticmethod
    def rpn_score_to_prob_softmax(rpn_cls_score):
        b, c, h, w = rpn_cls_score.shape
        d = 2
        rpn_cls_score = rpn_cls_score.view(b, d, c//d, h, w)
        rpn_cls_prob = F.softmax(rpn_cls_score, dim=1)
        rpn_cls_prob = rpn_cls_prob.view(b, c, h, w)
        return rpn_cls_prob

    def proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, im_info):
        rpn_cls_prob = tensor_to_array(rpn_cls_prob, dtype=np.float32)
        rpn_bbox_pred = tensor_to_array(rpn_bbox_pred, dtype=np.float32)
        proposals = proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, self.training, self.feature_stride, self.anchor_scales)
        proposals = np.hstack([np.zeros((proposals.shape[0],1),dtype=np.float32), proposals])
        proposals = np.ascontiguousarray(proposals)
        rois = torch.from_numpy(proposals).float().cuda()
        return rois.view(-1, 5)

    def anchor_target_layer(self, feature_map_size, gt_boxes, gt_ishard, im_info):
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                anchor_target_layer(feature_map_size, gt_boxes, gt_ishard, im_info, self.feature_stride, self.anchor_scales)
                
        rpn_labels = array_to_tensor(rpn_labels, is_cuda=self.use_cuda, dtype=torch.long)
        rpn_bbox_targets = array_to_tensor(rpn_bbox_targets, is_cuda=self.use_cuda, dtype=torch.float32)
        rpn_bbox_inside_weights = array_to_tensor(rpn_bbox_inside_weights, is_cuda=self.use_cuda, dtype=torch.float32)
        rpn_bbox_outside_weights = array_to_tensor(rpn_bbox_outside_weights, is_cuda=self.use_cuda, dtype=torch.float32)
        
        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


        