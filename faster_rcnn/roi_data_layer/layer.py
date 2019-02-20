"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import numpy as np

from .minibatch import get_minibatch
from ..config import cfg


class RoIDataLayer(object):
    """Fast R-CNN data layer used for training.
        A blob is a dict with these keys:
            im_data,
            gt_boxes,
            gt_ishard,
            im_info,
            im_name
    """

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
        """Return the blob to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db)

    def forward(self):
        """Get blob and copy them into this layer's top blob vector."""
        blob = self._get_next_minibatch()
        return blob


    # To get a better understanding of roidb prepared by this function
    # 'im_data': array([[[23.       , 36.       , 68.       ],
    #     [23.665    , 36.665    , 68.665    ],
    #     [24.775    , 37.774998 , 69.775    ],
    #     ...,
    #     [23.832489 , 31.832489 , 48.83249  ],
    #     [23.277496 , 31.277496 , 48.277496 ],
    #     [23.       , 31.       , 48.       ]],

    #     [[23.       , 36.       , 68.       ],
    #     [23.665    , 36.665    , 68.665    ],
    #     [24.775002 , 37.774998 , 69.775    ],
    #     ...,
    #     [25.16249  , 33.16249  , 50.16249  ],
    #     [24.607498 , 32.6075   , 49.6075   ],
    #     [24.330002 , 32.33     , 49.33     ]],

    #     [[23.       , 36.       , 68.       ],
    #     [23.665    , 36.665    , 68.665    ],
    #     [24.775    , 37.774998 , 69.775    ],
    #     ...,
    #     [27.382488 , 35.38249  , 52.38249  ],
    #     [26.827497 , 34.827496 , 51.8275   ],
    #     [26.55     , 34.55     , 51.55     ]],

    #     ...,

    #     [[10.       ,  6.       , 11.       ],
    #     [10.       ,  6.       , 11.       ],
    #     [10.       ,  6.       , 11.       ],
    #     ...,
    #     [11.832489 ,  6.832489 ,  8.720001 ],
    #     [11.277496 ,  6.2774963,  8.165009 ],
    #     [11.       ,  6.       ,  7.887512 ]],

    #     [[10.       ,  6.       , 11.       ],
    #     [10.       ,  6.       , 11.       ],
    #     [10.       ,  6.       , 11.       ],
    #     ...,
    #     [11.832489 ,  6.832489 ,  8.164978 ],
    #     [11.277496 ,  6.2774963,  7.6099854],
    #     [11.       ,  6.       ,  7.332489 ]],

    #     [[10.       ,  6.       , 11.       ],
    #     [10.       ,  6.       , 11.       ],
    #     [10.       ,  6.       , 11.       ],
    #     ...,
    #     [11.832489 ,  6.832489 ,  7.832489 ],
    #     [11.277496 ,  6.2774963,  7.2774963],
    #     [11.       ,  6.       ,  7.       ]]], dtype=float32),
    # {'gt_boxes': array([[153.15315 , 268.46848 , 899.0991  , 598.1982  ,  11.      ],
    #     [ 52.25225 , 158.55856 ,  82.88288 , 201.8018  ,   5.      ],
    #     [ 75.675674, 189.1892  , 102.702705, 263.06305 ,   5.      ],
    #     [  9.009009, 183.78378 ,  32.432434, 254.05405 ,   5.      ],
    #     [290.0901  , 230.63063 , 401.8018  , 331.53152 ,   9.      ],
    #     [459.45947 , 218.01802 , 560.36035 , 309.9099  ,   9.      ],
    #     [776.5766  , 207.20721 , 890.0901  , 318.9189  ,   9.      ],
    #     [657.65765 , 266.66666 , 861.2613  , 598.1982  ,   9.      ],
    #     [414.41443 , 291.8919  , 704.5045  , 598.1982  ,   9.      ],
    #     [  0.      , 295.49548 , 290.0901  , 598.1982  ,   9.      ]],
    # dtype=float32),
    # 'gt_ishard': array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int32),
    # 'im_info': array([901.       ,   3.       ,   1.8018018], dtype=float32),
    # 'im_name': '000564.jpg'}


