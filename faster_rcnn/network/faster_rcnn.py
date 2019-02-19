import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .modules import *
from .vgg import vgg16_bn as vgg16
from .resnet import resnet50

from .roi_pooling.modules.roi_pool_py import RoIPool as RoIPool_py
from .roi_pooling.modules.roi_pool import RoIPool

from ..utils.timer import Timer
from ..config import cfg

class BasicNetwork(nn.Module):
    """Basic Network to get feature map
    """
    def __init__(self, net_name=None):
        if net_name is None:
            self.conv = vgg16(pretrained=True)
            del self.conv.classifier.fc
        else:
            eval(cfg.NETWORK.BASIC_NETWORK)()

        if net_name is None:
            self.out_channels = 512
        else:
            self.out_channels = cfg.NETWORK.BASIC_NETWORK_OUTCHANNELS
        
    def forward(self, x):
        return self.conv(x)


class RPN(nn.Module):
    """RPN network
    """
    def __init__(self, in_channels, out_channels, sliding_window_size=3):
        super(RPN, self).__init__()

        self.anchor_scales = cfg.NETWORK.ANCHOR_SCALES
        self.anchor_ratios = cfg.NETWORK.ANCHOR_RATIOS

        self.conv = Conv2d(in_channels, out_channels, kernel_size=sliding_window_size, stride=1, same_padding=True)
        self.num_anchors = cfg.NETWORK.NUM_ANCHORS
        self.score_conv = Conv2d(out_channels, self.num_anchors*2, 1, relu=False, same_padding=False)
        self.bbox_conv  = Conv2d(out_channels, self.num_anchors*4, 1, relu=False, same_padding=False)

        # loss
        self.rpn_cls_loss = None
        self.rpn_box_loss = None
        
    def build_rpn_loss(self):
        return self.rpn_cls_loss + self.rpn_bbox_loss

    def forward(self, x):
        x = self.conv(x)

        # rpn score
        rpn_cls_score = self.score_conv(x)
        rpn_cls_prob = self.rpn_score_to_prob_softmax(rpn_cls_score)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(x)

        # proposal layer
        rois = self.rpn_get_proposal(rpn_cls_score, rpn_bbox_pred, im_info)

        return rois

    @staticmethod
    def rpn_score_to_prob_softmax(rpn_cls_score):
        b, c, h, w = rpn_cls_score.shape
        rpn_cls_score = rpn_cls_score.resize(b, d, c//d, h, w)
        rpn_cls_prob = F.softmax(rpn_cls_score, dim=1)
        rpn_cls_prob = rpn_cls_prob.resize(b, c, h, w)
        return rpn_cls_prob
        
    @staticmethod
    def rpn_get_proposal():
        pass

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
        self.rpn = RPN(self.basic_network.out_channels, cfg.NETWORK.RPN_CONV_OUTCHANNELS)
        feature_stride = cfg.NETWORK.FEATURE_STRIDE # <== feature stride
        roi_pooled_size = cfg.NETWORK.ROI_POOLED_SIZE
        self.roipool_layer = RoIPool(roi_pooled_size, roi_pooled_size, 1./feature_stride)
        self.rcnn_fc = nn.Sequential([
            FC(cfg.NETWORK.RPN_CONV_OUTCHANNELS*roi_pooled_size*roi_pooled_size, cfg.NETWORK.RCNN_FC_OUTCHANNELS, dropout=True),
            FC(cfg.NETWORK.RCNN_FC_OUTCHANNELS, cfg.NETWORK.RCNN_FC_OUTCHANNELS, dropout=True)
        ])
        self.rcnn_cls_fc = FC(cfg.NETWORK.RCNN_FC_OUTCHANNELS, self.num_classes+1, relu=False)
        self.rcnn_box_fc = FC(cfg.NETWORK.RCNN_FC_OUTCHANNELS, 4, relu=False)

        self.use_cuda = cfg.USE_CUDA

        # loss
        self.rcnn_cls_loss = None
        self.rcnn_box_loss = None

    def preprocess(self, im_data):
        """
        Input:
            im_data: HxWxC, numpy.ndarray, [0,255]
        Return:
            Normalized im_data
        """
        im_data = (im_data).astype(np.float32)/255
        im_data = torch.from_numpy(im_data).permute(2,0,1)
        im_data = self._normalize(im_data)
        im_data = im_data.unsqueeze(0)
        return im_data

    def forward(self, im_blob):
        """A im_blob is a dict with these keys:
            im_data,
            gt_boxes,
            gt_ishard,
            im_info,
            im_name
        """
        im_data = self.preprocess(im_blob['im_data'], is_cuda=self.use_cuda)
        feature_map = self.features(im_data)

        rois = self.rpn(feature_map)
        # roi pooling
        roi_pooled_features = self.roipool_layer(feature_map, rois)
        x = roi_pooled_features.view(roi_pooled_features.size(0), -1)
        x = self.rcnn_fc(x)
        rcnn_cls_score = self.rcnn_cls_fc(x)
        rcnn_cls_prob  = F.softmax(rcnn_cls_score, dim=1)
        rcnn_box_pred  = self.rcnn_box_fc(x)

        if self.training:
            self.rcnn_cls_loss, self.rcnn_box_loss = self.build_rcnn_loss(rcnn_cls_prob, rcnn_box_pred, im_blob['gt_boxes'])
            pass

        return rcnn_cls_prob, rcnn_box_pred, rois

    @staticmethod
    def build_rcnn_loss(rcnn_cls_prob, rcnn_box_pred, gt_boxes): 
        pass



    
    def detect(self, x):
        return bboxes
