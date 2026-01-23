from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv import Config
from sam import *

cfg = Config.fromfile('configs/sam/sam_NIHLN.py')
cfg.model

model = build_detector(cfg.model)

