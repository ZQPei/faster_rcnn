import torch

def NMS(dets, threshold):
    """python nms baseline using torch tensor in gpu
    Input:
        dets: [[x1,y1,x2,y2, score_fg]]
    Return:
        dets selected by nms
    """
    assert dets.dim() == 2 and dets.size(1) == 5, "input error of dets"

    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    score = dets[:,4]

    # 1 compute areas
    areas = (x2-x1+1) * (y2-y1+1)

    # 2 sort score 
    order = score.sort(dim=0,descending=True)[1]

    # 3 del bbox of those IoU greater than threshold
    # import ipdb; ipdb.set_trace()
    mask = torch.zeros_like(order).long().cuda()
    while order.numel() > 0:
        i = order[0]
        mask[i] = 1
        # compute IoU
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = xx2 - xx1 + 1
        h = yy2 - yy1 +1
        w[w<0] = 0
        h[h<0] = 0
        inter_area = w*h
        IoU = inter_area/(areas[i]+areas[order[1:]]-inter_area)

        order = order[1:][IoU<=threshold]

    return mask
        


