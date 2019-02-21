import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np

from .modules import *
from .vgg import vgg16_bn as vgg16
from .resnet import resnet50

from .rpn import RPN

from .proposals.proposal_target_layer import proposal_target_layer
from .proposals.bbox_transform import bbox_transform_inv, clip_boxes

from .nms import nms

from .roi_pooling.modules.roi_pool_py import RoIPool as RoIPool_py
from .roi_pooling.modules.roi_pool import RoIPool

from ..loss_function.rcnn_loss import build_rcnn_loss

from ..utils.timer import Timer
from ..config import cfg

class BasicNetwork(nn.Module):
    """Basic Network to get feature map
    """
    def __init__(self, net_name=None):
        super(BasicNetwork, self).__init__()
        if net_name is None:
            self.conv = vgg16(pretrained=True)
            del self.conv.classifier
        else:
            eval(cfg.NETWORK.BASIC_NETWORK)()

        if net_name is None:
            self.out_channels = 512
        else:
            self.out_channels = cfg.NETWORK.BASIC_NETWORK_OUTCHANNELS
        
    def forward(self, x):
        return self.conv(x)

class FasterRCNN(nn.Module):
    """Faster RCNN network
    """

    def __init__(self, num_classes, is_train=False):
        super(FasterRCNN, self).__init__()
        
        self.num_classes = num_classes
        mean = cfg.MEAN
        std  = cfg.STD
        self._normalize = transforms.Normalize(mean, std)

        self.features = BasicNetwork()
        self.rpn = RPN(self.features.out_channels, cfg.NETWORK.RPN_CONV_OUTCHANNELS)
        feature_stride = cfg.NETWORK.FEATURE_STRIDE # <== feature stride
        roi_pooled_size = cfg.NETWORK.ROI_POOLED_SIZE
        self.roipool_layer = RoIPool(roi_pooled_size, roi_pooled_size, 1./feature_stride)
        self.rcnn_fc = nn.Sequential(
            FC(cfg.NETWORK.RPN_CONV_OUTCHANNELS*roi_pooled_size*roi_pooled_size, cfg.NETWORK.RCNN_FC_OUTCHANNELS, dropout=True),
            FC(cfg.NETWORK.RCNN_FC_OUTCHANNELS, cfg.NETWORK.RCNN_FC_OUTCHANNELS, dropout=True)
        )
        self.rcnn_cls_fc = FC(cfg.NETWORK.RCNN_FC_OUTCHANNELS, self.num_classes, relu=False)   # ==> 21
        self.rcnn_bbox_fc = FC(cfg.NETWORK.RCNN_FC_OUTCHANNELS, self.num_classes * 4, relu=False)# ==> 21*4 = 84
        # weight_init(self.rcnn_fc)
        # weight_init(self.rcnn_cls_fc)
        # weight_init(self.rcnn_bbox_fc)

        self.use_cuda = cfg.USE_CUDA

        # loss
        self.rcnn_cls_loss = None
        self.rcnn_box_loss = None

    @property
    def loss(self):
        return self.rcnn_cls_loss + self.rcnn_box_loss*10

    @staticmethod
    def preprocess(im_data, transform=None, is_cuda=False):
        """
        Input:
            im_data: HxWxC, numpy.ndarray, [0,255], RGB color
        Return:
            Normalized im_data: 1xCxHxW, torch.tensor
        """
        im_data = (im_data).astype(np.float32)/255
        im_data = torch.from_numpy(im_data).permute(2,0,1)
        if transform is not None:
            im_data = transform(im_data)
        im_data = im_data.unsqueeze(0)
        if is_cuda:
            im_data = im_data.cuda()
        return im_data

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None):
        im_data = self.preprocess(im_data, transform=self._normalize, is_cuda=self.use_cuda)

        feature_map = self.features(im_data)

        rois = self.rpn(feature_map, im_info, gt_boxes, gt_ishard)
        if self.training:
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
                proposal_target_layer(rois, gt_boxes, gt_ishard, self.num_classes)

        # roi pooling
        roi_pooled_features = self.roipool_layer(feature_map, rois)
        x = roi_pooled_features.view(roi_pooled_features.size(0), -1)
        x = self.rcnn_fc(x)

        # rcnn cls prob
        rcnn_cls_score = self.rcnn_cls_fc(x)
        rcnn_cls_prob  = F.softmax(rcnn_cls_score, dim=1)
        # rcnn bboxes
        rcnn_bbox_pred  = self.rcnn_bbox_fc(x)

        if self.training:
            self.rcnn_cls_loss, self.rcnn_box_loss = \
                build_rcnn_loss(rcnn_cls_score, rcnn_bbox_pred, rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        return rcnn_cls_prob, rcnn_bbox_pred, rois

    @staticmethod
    def interpret_faster_rcnn(rcnn_cls_prob, rcnn_bbox_pred, rois, im_info, min_score, use_nms=True, use_clip=True):
        """
        Say N is the number of rois
        rcnn_cls_prob  (N, 21)
        rcnn_bbox_pred  (N, 84)
        rois  (N, 4)
        """
        # import ipdb; ipdb.set_trace()
        # filter bg cls and scores smaller than min_score
        scores, cls_inds = rcnn_cls_prob.max(1)
        mask = (cls_inds>0)*(scores>=min_score)
        scores = scores[mask]
        cls_inds = cls_inds[mask]
        box_deltas = rcnn_bbox_pred[mask, :]
        im_height, im_width, im_scale_ratio = im_info.data
        boxes = rois[mask, :]/im_scale_ratio

        if cls_inds.shape[0]==0:
            return boxes, scores, cls_inds
        # do bbox transform
        box_deltas = torch.stack([
            box_deltas[i, cls_inds[i]*4 : cls_inds[i]*4+4] for i in range(cls_inds.shape[0])
        ], dim=0)
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if use_clip:
            pred_boxes = clip_boxes(pred_boxes, im_width, im_height)

        if use_nms and pred_boxes.shape[0]>0:
            nms_mask = nms(torch.cat([pred_boxes, scores.view(-1,1)], dim=1), threshold=cfg.TEST.RCNN_NMS_THRESH)
            pred_boxes = pred_boxes[nms_mask,:]
            scores = scores[nms_mask]
            cls_inds = cls_inds[nms_mask]

        return pred_boxes, scores, cls_inds

    
    def detect(self, im_data, im_info, score_thresh=0.0):
        """
        Input:
            im_data: numpy.ndarray
                    RGB
                    scaled shorter side to 600
        Return:
            prob_boxes, scores, cls_inds
        """
        rcnn_cls_prob, rcnn_bbox_pred, rois = self(im_data, im_info)
        rcnn_cls_prob, rcnn_bbox_pred, rois = rcnn_cls_prob.data.cpu(), rcnn_bbox_pred.data.cpu(), rois.data.cpu()
        prob_boxes, scores, cls_inds = self.interpret_faster_rcnn(rcnn_cls_prob, rcnn_bbox_pred, rois, im_info, score_thresh)
        prob_boxes, scores, cls_inds = (prob_boxes/im_info[2]).numpy(), scores.numpy(), cls_inds.numpy()
        return prob_boxes, scores, cls_inds

        
