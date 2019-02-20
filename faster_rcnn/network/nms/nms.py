# coding=utf-8
import _ext.nms as nms_c
import numpy as np
import torch


def bbox_point_trans(prediction):
    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, 0:4] = box_a[:, :, 0:4]
    return prediction


def softmax_to_class_label(prediction, classes_number):
    new_prediction = torch.zeros(
        (prediction.size()[0], prediction.size()[1], 7)).cuda()
    max_score, max_index = torch.max(prediction[:, :, 5:5+classes_number], 2)
    max_index = max_index.float()
    new_prediction[:, :, 0:5] = prediction[:, :, 0:5]
    new_prediction[:, :, 5] = max_score
    new_prediction[:, :, 6] = max_index
    return new_prediction


def sort_predition(prediction):
    _, index = torch.sort(prediction[:, :, 4], dim=1, descending=True)
    for i in range(0, prediction.size()[0]):
        prediction[i, :, :] = prediction[i, index[i], :]
    return prediction


def make_mask(prediction, thresh, classes_number):
    mask = torch.zeros((prediction.size()[0], prediction.size()[1])).cuda()
    mask_thresh = (prediction[:, :, 4] > thresh).float().cuda()
    return mask_thresh


def make_result(prediction, mask):
    output = 0
    has_value = 0
    for i in range(0, mask.size()[0]):
        select_index = torch.nonzero(mask[i, :])
        select_index = select_index.view(-1)
        batch_output = prediction[i, select_index, :]
        batch_number = torch.zeros((batch_output.size()[0], 1)).cuda()
        batch_number[:, :] = i
        batch_output = torch.cat((batch_number, batch_output), 1)
        if(has_value == 0):
            output = batch_output
            has_value = 1
        else:
            output = torch.cat((output, batch_output), 0)
    return output


def nms(prediction, mask, thresh):
    nms_c.nms(prediction, mask, thresh)
    return mask


def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    prediction = bbox_point_trans(prediction)
    prediction = softmax_to_class_label(prediction, num_classes)
    prediction = sort_predition(prediction)
    mask = make_mask(prediction, confidence, num_classes)
    if nms:
        nms_c.nms(prediction, mask, nms_conf)
    result = make_result(prediction, mask)
    return result
