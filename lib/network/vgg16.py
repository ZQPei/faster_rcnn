import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# from .utils.blob import im_list_to_blob
from .modules import Conv2d, set_trainable


class VGG16(nn.Module):
    def __init__(self, bn=False):
        super(VGG16, self).__init__()

        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        set_trainable(self.conv1, requires_grad=False)
        set_trainable(self.conv2, requires_grad=False)

        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn))
        #set_trainable(self.conv3, requires_grad=False)
        #set_trainable(self.conv4, requires_grad=False)
        #set_trainable(self.conv5, requires_grad=False)

    def forward(self, x):
        # im_data, im_scales = get_blobs(image)
        # im_info = np.array(
        #     [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        #     dtype=np.float32)
        # data = Variable(torch.from_numpy(im_data)).cuda()
        # x = data.permute(0, 3, 1, 2)
        #import ipdb;ipdb.set_trace()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x#.data

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        own_dict = self.state_dict()
        for name, val in own_dict.items():
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}/{}:0'.format(i, j, ptype)
            param = torch.from_numpy(params[key])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)

def load_pretrained_npy(vgg16, fname):
    params = np.load(fname, encoding='bytes').item()
    # vgg16
    vgg16_dict = vgg16.state_dict()
    for name, val in vgg16_dict.items():
        if name.find('bn.') >= 0:
            continue
        i, j = int(name[4]), int(name[6]) + 1
        ptype = b'weights' if name[-1] == 't' else b'biases'
        key = b'conv%d_%d'%(i, j)
        param = torch.from_numpy(params[key][ptype])

        if ptype == b'weights':
            param = param.permute(3, 2, 0, 1)

        val.copy_(param)

    # def load_from_npy_file(self, fname):
    #     own_dict = self.state_dict()
    #     params = np.load(fname).item()
    #     for name, val in own_dict.items():
    #         # # print name
    #         # # print val.size()
    #         # # print param.size()
    #         # if name.find('bn.') >= 0:
    #         #     continue
    #
    #         i, j = int(name[4]), int(name[6]) + 1
    #         ptype = 'weights' if name[-1] == 't' else 'biases'
    #         key = 'conv{}_{}'.format(i, j)
    #         param = torch.from_numpy(params[key][ptype])
    #
    #         if ptype == 'weights':
    #             param = param.permute(3, 2, 0, 1)
    #
    #         val.copy_(param)


if __name__ == '__main__':
    vgg = VGG16()
    vgg.cuda()

    #vgg.load_from_npy_file('/media/longc/Data/models/VGG_imagenet.npy')
    for i in range (1000):
        im_data = torch.randn(1,3,600,600).cuda()

        features = vgg(im_data)
        print(features.shape)
