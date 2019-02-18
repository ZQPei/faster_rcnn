"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import numpy as np

from .minibatch import get_minibatch
from ..config import cfg


class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb's indices."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()
        
        db_inds = self._perm[self._cur : self._cur+cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs