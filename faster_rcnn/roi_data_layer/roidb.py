import PIL
import numpy as np

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in range(imdb.num_images)]
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   num_objs
    #   gt_overlaps
    #   gt_classes
    #   flipped
    # Now add the following keys:
    #   image
    #   width
    #   height
    #   max_classes
    #   max_overlaps
    roidb = imdb.roidb
    for i in range(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps

        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (all the rois must be foreground classes)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

    # To get a better understanding of roidb prepared by this function
    # imdb.roidb[0]  --> pascal voc 2007 trainval set 000005.jpg
    # {'boxes': array([[262, 210, 323, 338],
    #    [164, 263, 252, 371],
    #    [  4, 243,  66, 373],
    #    [240, 193, 294, 298],
    #    [276, 185, 311, 219]], dtype=uint16), 
    #    'gt_classes': array([9, 9, 9, 9, 9], dtype=int32), 
    #    'gt_ishard': array([0, 0, 1, 0, 1], dtype=int32), 
    #    'gt_overlaps': <5x21 sparse matrix of type '<class 'numpy.float32'>'
    #     with 5 stored elements in Compressed Sparse Row format>, 
    #     'flipped': False, 
    #     'seg_areas': array([7998., 9701., 8253., 5830., 1260.], dtype=float32), 
    #     'image': '/home/pzq/Desktop/faster_rcnn_pytorch/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg', 
    #     'width': 500, 'height': 375, 'max_classes': array([9, 9, 9, 9, 9]), 
    #     'max_overlaps': array([1., 1., 1., 1., 1.], dtype=float32)}
