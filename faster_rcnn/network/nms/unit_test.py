import sys

import nms
import numpy as np
import torch


def test_nms():
    bbox = [[
        [3, 3, 6, 6, 0.7, 0.8, 0.2], [4, 4, 6, 6, 0.8,
                                      0.7, 0.3], [8.5, 8.5, 7, 7, 0.5, 0.6, 0.4],
        [3, 3, 6, 6, 0.2, 0.2, 0.8], [4, 4, 6, 6, 0.8, 0.3, 0.7], [8.5, 8.5, 7, 7, 0.5, 0.4, 0.6]],
        [[3, 3, 6, 6, 0.8, 0.8, 0.2], [3, 4, 6, 6, 0.4, 0.8, 0.2], [0, 0, 0, 0, 0.1, 0.1, 0.2],
         [3, 3, 6, 6, 0.2, 0.2, 0.8], [3, 4, 6, 6, 0.8, 0.2, 0.8], [0, 0, 0, 0, 0.1, 0.1, 0.2]]]
    bbox = torch.Tensor(bbox).cuda()

    confidence = 0.15

    target_bbox = [[0,  1,  1,  7,  7,  0.8,  0.7,  1],
                   [0,  1,  1,  7,  7,  0.8,  0.7,  0],
                   [0,  5,  5, 12, 12,  0.5,  0.6,  1],
                   [0,  5,  5, 12, 12,  0.5,  0.6,  0],
                   [1,  0,  1,  6,  7,  0.8,  0.8,  1],
                   [1,  0,  0,  6,  6,  0.8,  0.8,  0]]
    target_bbox = torch.Tensor(target_bbox).cuda()

    result = nms.write_results(bbox, confidence, 2)

    if(result.size()[0] != target_bbox.size()[0]):
        print('fail the test')
        print(result)
        sys.exit()

    if(torch.sum((target_bbox-result)) == 0):
        print('test succeed in nms bbox')
    else:
        print('test fail in nms bbox')
        print(result)


if __name__ == '__main__':
    test_nms()
