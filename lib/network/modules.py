import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size-1)/2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True, dropout=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.dropout = nn.Dropout(p=0.5, inplace=False) if dropout else None # inplace must be False Here!!!

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x

def save_net(net, fname):
    torch.save(net.state_dict(), fname)

def load_net(net, fname):
    state_dict = torch.load(fname)
    net.load_state_dict(state_dict)
    print("load model {} successfully!".format(fname))

def array_to_tensor(x, is_cuda=True, dtype=torch.float32):
    x = torch.from_numpy(x).type(dtype).data
    x.requires_grad = False
    if is_cuda:
        x = x.cuda()
    return x

def tensor_to_array(x, dtype=np.float32):
    x = x.data.cpu().numpy().astype(dtype, copy=False)
    return x

def weights_normal_init(model, devilation=0.01):
    """init the conv and bn and fc layers weights by standard devilation"""
    import math
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, devilation)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, devilation)
                if m.bias is not None:
                    m.bias.data.zero_()

def weights_normal_init_kaiming(model, devilation=0.01):
    """init the conv and bn and fc layers weights by standard devilation"""
    if isinstance(model, list):
        for m in model:
            weights_normal_init_kaiming(m, devilation)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


def weights_normal_init_2(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


weight_init = weights_normal_init_2

def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            if torch.__version__[2]=='4' or torch.__version__[0]=='1':
                p.grad.mul_(float(norm))
            else:
                p.grad.mul_(norm)