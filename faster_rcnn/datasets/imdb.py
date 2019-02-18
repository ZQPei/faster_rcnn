import os
import os.path as osp

from ..config import cfg
class Imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._class_to_ind = {}
        self._image_index = []
        self._num_images = 0

        self._roidb = None
        self._roidb_handler = None

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   num_objs
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is None:
            assert callable(self._roidb_handler), "define _roidb_handler"
            return self._roidb_handler()

        return self._roidb

    @property
    def name(self):
        return self._name
    @property
    def classes(self):
        return self._classes
    @property
    def num_classes(self):
        return self._num_classes
    @property
    def class_to_ind(self):
        return self._class_to_ind
    @property
    def image_index(self):
        return self._image_index
    @property
    def num_images(self):
        return self._num_images

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path