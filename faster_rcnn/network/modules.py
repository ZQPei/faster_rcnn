import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.dropout = nn.Dropout(p=0.5, inplace=True) if dropout else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def save_net(net, fname):
    torch.save(net, fname)

def load_net(net, fname):
    torch.load(net, fname)

