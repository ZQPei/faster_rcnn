import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from faster_rcnn.datasets.pascal_voc import Pascal_VOC
from faster_rcnn.data_layer.roidb import prepare_roidb
from faster_rcnn.data_layer.layer import DataLayer

from faster_rcnn.network.faster_rcnn import FasterRCNN
from faster_rcnn.network.modules import save_net, load_net, clip_gradient

from faster_rcnn.utils.log_print import log_print
from faster_rcnn.utils.timer import Timer
from faster_rcnn.config import cfg

rand_seed = cfg.SEED
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

save_model_dir = cfg.SAVE_MODEL_DIR

verbose = cfg.VERBOSE

# Dataset imdb
dataset_name = cfg.DATASET.NAME

# data_layer
if dataset_name == 'Pascal_VOC':
    imdb = Pascal_VOC("trainval", "2007")
prepare_roidb(imdb)
data_layer = DataLayer(imdb.roidb)

# network definition
net = FasterRCNN(imdb.num_classes)
net.train()
if cfg.USE_CUDA:
    net.cuda()
if verbose:
    print(net)

# Dont uncomment this line!!!
# cudnn.benchmark = True

# Iteration config
start_step = cfg.TRAIN.START_STEP
end_step = cfg.TRAIN.END_STEP
lr = cfg.TRAIN.LEARNING_RATE
lr_decay = cfg.TRAIN.LEARNING_RATE_DECAY
milestones = cfg.TRAIN.MILESTONE
gamma = cfg.TRAIN.GAMMA
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
dampening = cfg.TRAIN.DAMPENING
log_interval = cfg.TRAIN.LOG_INTERVAL

# Optimizer
params = list(net.parameters())[8:]
optimizer = torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening)
lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=start_step-1)

# Start training
t = Timer()
t.tic()
train_loss = 0
step_cnt = 0
if verbose:
    tp, tf, fg, bg = 0., 0., 0, 0
for step in range(start_step, end_step):
    inputs = data_layer.forward()
    im_data = inputs['im_data']
    im_info = inputs['im_info']
    gt_boxes = inputs['gt_boxes']
    gt_ishard = inputs['gt_ishard']

    # forward
    net(im_data, im_info, gt_boxes, gt_ishard)

    loss = net.loss + net.rpn.loss
    train_loss += loss.data.item()

    if verbose:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    # bachward
    optimizer.zero_grad()
    loss.backward()
    clip_gradient(net, 10.)
    optimizer.step()
    lr_schedular.step()

    step_cnt += 1
    if step % log_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, inputs['im_name'], train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])

        if verbose:
            #print(tp,fg,tf,bg,step_cnt,sep='\n')
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            #import ipdb; ipdb.set_trace()
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.rpn_cls_loss.data.cpu().numpy(), net.rpn.rpn_box_loss.data.cpu().numpy(),
                net.rcnn_cls_loss.data.cpu().numpy(), net.rcnn_box_loss.data.cpu().numpy())
            )
        re_cnt = True

    if (step % 10000 == 0) and step > 0:
        save_name = os.path.join(save_model_dir, 'faster_rcnn_{}.pkl'.format(step))
        save_net(net, save_name)
        print('save model: {}'.format(save_name))

    if re_cnt:
        if verbose:
            tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

from IPython import embed; embed()