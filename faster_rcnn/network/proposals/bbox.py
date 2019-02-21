import torch

def bbox_overlaps_torch(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) torch tensor of float
    query_boxes: (K, 4) torch tensor of float
    Returns
    -------
    overlaps: (N, K) torch tensor of overlap between boxes and query_boxes
    """
    overlaps = []
    areas = (boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    for query_box in query_boxes:
        xx1 = torch.max(query_box[0], boxes[:,0])
        yy1 = torch.max(query_box[1], boxes[:,1])
        xx2 = torch.min(query_box[2], boxes[:,2])
        yy2 = torch.min(query_box[3], boxes[:,3])
        w = xx2-xx1+1
        h = yy2-yy1+1
        inter = w*h
        query_box_area = (query_box[2]-query_box[0]+1)*(query_box[3]-query_box[1]+1)
        overlaps.append( inter/(query_box_area+areas-inter) )
    overlaps = torch.stack(overlaps, dim=0).t_().contiguous()
    return overlaps

import numpy as np

def bbox_overlaps_np(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    overlaps = []
    areas = (boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    for query_box in query_boxes:
        xx1 = np.maximum(query_box[0], boxes[:,0])
        yy1 = np.maximum(query_box[1], boxes[:,1])
        xx2 = np.maximum(query_box[2], boxes[:,2])
        yy2 = np.maximum(query_box[3], boxes[:,3])
        w = xx2-xx1+1
        h = yy2-yy1+1
        inter = w*h
        query_box_area = (query_box[2]-query_box[0]+1)*(query_box[3]-query_box[1]+1)
        overlaps.append( inter/(query_box_area+areas-inter) )
    overlaps = np.vstack(overlaps).T
    return overlaps