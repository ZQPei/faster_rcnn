import numpy as np

def NMS(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# def NMS(dets, threshold):
#     """python nms baseline using numpy in cpu
#     Input:
#         dets: [[x1,y1,x2,y2, score_fg]]
#     Return:
#         dets selected by nms
#     """
#     assert dets.ndim == 2 and dets.shape[1] == 5, "input error of dets"

#     x1 = dets[:,0]
#     y1 = dets[:,1]
#     x2 = dets[:,2]
#     y2 = dets[:,3]
#     score = dets[:,4]

#     # 1 compute areas
#     areas = (x2-x1+1) * (y2-y1+1)

#     # 2 sort score
#     order = score.argsort()[::-1]

#     # 3 del bbox of those IoU greater than threshold
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         # compute IoU
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.maximum(x2[i], x2[order[1:]])
#         yy2 = np.maximum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter_area = w * h
#         IoU = inter_area/(areas[i]+areas[order[1:]]-inter_area)

#         inds = np.where(IoU<threshold)[0]
#         order = order[inds+1]

#     mask = np.array(keep, dtype=np.int64)
#     return mask
        


