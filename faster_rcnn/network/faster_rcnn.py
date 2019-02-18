import torch
import torch.nn as nn

from .modules import *
from .vgg import vgg16_bn as vgg16
from .resnet import resnet50

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
        self.conv = Conv2d(in_channels, out_channels, kernel_size=sliding_window_size, stride=1, same_padding=True)
        self.num_anchors = cfg.NETWORK.NUM_ANCHORS
        self.score_conv = Conv2d(out_channels, self.num_anchors*2, 1, relu=False, same_padding=False)
        self.bbox_conv  = Conv2d(out_channels, self.num_anchors*4, 1, relu=False, same_padding=False)

        # loss
        

    def forward(self, x):
        x = self.conv(x)

        # rpn score
        rpn_cls_score = self.score_conv(x)
        rpn_cls_prob = self.softmax_rpn_cls_score(rpn_cls_score)

        # rpn boxes
        rpn_bbox_reg = self.bbox_conv(x)

        

        return rpn_score, rpn_bbox

    @staticmethod
    def softmax_rpn_cls(rpn_cls_score):
        b, c, h, w = rpn_cls_score.shape
        rpn_cls_score = rpn_cls_score.resize(b, d, c//d, h, w)
        rpn_cls_prob = F.softmax(rpn_cls_score, dim=1)
        rpn_cls_prob = rpn_cls_prob.resize(b, c, h, w)
        return rpn_cls_prob

class FasterRCNN(nn.Module):
    """Faster RCNN network
    """

    def __init__(self, num_classes, is_train=False):
        super(FasterRCNN, self).__init__()
        
        self.num_classes = num_classes
        self.features = BasicNetwork()
        self.rpn = RPN(self.basic_network.out_channels, cfg.NETWORK.RPN_CONV_OUTCHANNELS)
        self.fast_rcnn = 

        self.use_cuda = cfg.USE_CUDA


    def forward(self, blob):
        """A blob is a dict contains these keys:
            im_data,
            gt_boxes,
            gt_ishard,
            im_info,
            im_name
        """
        im_data = array_to_tensor(blob['im_data'], is_cuda=self.use_cuda)
        feature_map = self.features(im_data)

        rpn_conv1 = self.rpn(feature_map)



    
    def detect(self, x):
        pass
