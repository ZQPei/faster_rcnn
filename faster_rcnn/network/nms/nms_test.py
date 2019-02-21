import torch

import _ext.nms as nms
from nms_gpu_torch import NMS

bbox = [[[3, 3, 6, 6, 0.7, 0.8, 0.2], 
        [4, 4, 6, 6, 0.8,0.7, 0.3], 
        [8.5, 8.5, 7, 7, 0.5, 0.6, 0.4],
        [3, 3, 6, 6, 0.2, 0.2, 0.8], 
        [4, 4, 6, 6, 0.8, 0.3, 0.7], 
        [8.5, 8.5, 7, 7, 0.5, 0.4, 0.6]],
        [[3, 3, 6, 6, 0.8, 0.8, 0.2], 
        [3, 4, 6, 6, 0.4, 0.8, 0.2], 
        [0, 0, 0, 0, 0.1, 0.1, 0.2],
         [3, 3, 6, 6, 0.2, 0.2, 0.8], 
         [3, 4, 6, 6, 0.8, 0.2, 0.8], 
         [0, 0, 0, 0, 0.1, 0.1, 0.2]]]
bbox = torch.Tensor(bbox).cuda()

target_bbox =  [[0,  1,  1,  7,  7,  0.8,  0.7,  1],
                [0,  1,  1,  7,  7,  0.8,  0.7,  0],
                [0,  5,  5, 12, 12,  0.5,  0.6,  1],
                [0,  5,  5, 12, 12,  0.5,  0.6,  0],
                [1,  0,  1,  6,  7,  0.8,  0.8,  1],
                [1,  0,  0,  6,  6,  0.8,  0.8,  0]]
target_bbox = torch.Tensor(target_bbox).cuda()

dets = torch.tensor([[3, 3, 6, 6, 0.7, 0.8, 0.2], 
        [4, 4, 6, 6, 0.8,0.7, 0.3], 
        [8.5, 8.5, 7, 7, 0.5, 0.6, 0.4],
        [3, 3, 6, 6, 0.2, 0.2, 0.8], 
        [4, 4, 6, 6, 0.8, 0.3, 0.7], 
        [8.5, 8.5, 7, 7, 0.5, 0.4, 0.6]]).cuda()

mask = torch.zeros_like(dets[:,:5]).cuda()
nms.nms(dets[:,:5], mask, 0.5)

from IPython import embed; embed()



