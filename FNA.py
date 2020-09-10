#import numpy as np
#from mmdet.apis import inference_detector, show_result
#import mmcv
#import cv2
#from mmcv.image import imread, imwrite
#import argparse
#import os
#import torch
import sys
import os.path as osp
sys.path.append(osp.join(sys.path[0], '..'))
#import time
import mmcv
import models
#from mmcv.parallel import MMDataParallel, collate, scatter
#from mmcv.runner import load_checkpoint, obj_from_dict
#from mmdet import datasets
#from mmdet.core import coco_eval, results2json
#from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
#from tools import utils
#from ..configs import fna_retinanet_fpn_retrain.model

from quiver_engine import server1
from quiver_engine.model_utils import register_hook
from torchvision import  models

config_file = './fna_retinanet_fpn_retrain.py'
#checkpoint_file = '../output/retrain/output/epoch_12.pth'
#img_path = './imgs_test/imgs_raw/000000011760.jpg'
img_path = './data/Cat/0.jpg'

cfg = mmcv.Config.fromfile(config_file)
cfg.model.pretrained = None
model = build_detector(
    cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
print(model)
#_ = load_checkpoint(model, checkpoint_file)

# 测试单张图片并展示结果
#img = mmcv.imread(img_path)
#infer_start = time.time()
#result = inference_detector(model, img, cfg)
#infer_time = time.time() - infer_start
#print('the infer time is {}'.format(infer_time))
#show_result(img, result)

# hook_list = register_hook(model)

hook_list = register_hook(model)
# items = list(model.named_modules())
# print (model.named_modules)
server1.launch(model, hook_list, input_folder="./data/Cat", image_size=[224,224], use_gpu=True)