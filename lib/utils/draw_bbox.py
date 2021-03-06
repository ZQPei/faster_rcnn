import cv2
import numpy as np

def draw_bbox(im, bboxes, scores, cls_str, im_color_mode='RGB'):
    assert bboxes.ndim == 2
    assert bboxes.shape[0] == scores.shape[0]

    if im_color_mode == 'RGB':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    for i, bbox in enumerate(bboxes):
        bbox = tuple(np.round(bbox).astype(np.int))
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (255, 205, 51), 2)
        cv2.putText(im, '%s: %.3f' % (cls_str[i], scores[i]), 
                    (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)

    return im
