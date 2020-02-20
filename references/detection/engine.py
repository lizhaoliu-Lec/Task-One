import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from references.detection.coco_utils import get_coco_api_from_dataset
from references.detection.coco_eval import CocoEvaluator
from references.detection import utils


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types
