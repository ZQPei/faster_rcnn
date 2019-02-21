from .nms_cpu_np import NMS as nms_cpu
from .nms_gpu_torch import NMS as nms_gpu

from ...config import cfg

nms = nms_gpu if cfg.USE_CUDA else nms_cpu



