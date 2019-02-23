import os

import torch
import numpy as np
import PIL.Image
import cv2

from faster_rcnn.data_layer.minibatch import preprocess

from faster_rcnn.network.faster_rcnn import FasterRCNN
from faster_rcnn.network.modules import load_net

from faster_rcnn.utils.timer import Timer, tic, toc
from faster_rcnn.utils.draw_bbox import draw_bbox
from faster_rcnn.config import cfg

demo_image_dir = cfg.DEMO_IMAGE_DIR
demo_model_file = cfg.DEMO_MODEL_FILE
demo_thresh = cfg.DEMO_THRESH

net = load_net(demo_model_file)
net.eval()
net.cuda()

t = Timer()

image_files = [os.path.join(demo_image_dir,x) 
                for x in os.listdir(demo_image_dir) if x[-4:] is '.jpg']

for image_name in image_files:
    im = PIL.Image.open(image_name)
    im = np.array(im)
    im_data, im_scale = preprocess(im)
    
    t.tic()
    bboxes, scores, cls_inds = net.detect(im_data, demo_thresh)
    runtime = t.toc(average=False)

    print('total spend: {}s'.format(runtime), dets)
    cls_str = [cfg.DATASET.CLASSES[x] for x in cls_inds]
    im2show = draw_bbox(im, bboxes, scores, cls_str, im_color_mode='RGB')

    cv2.imshow(image_name, im2show)
    cv2.waitKey(0)



