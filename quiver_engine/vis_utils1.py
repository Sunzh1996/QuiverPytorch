import numpy as np
import  cv2, torch
from os.path import abspath, dirname, join
from quiver_engine.file_utils import save_layer_img

# from quiver_engine.layer_result_generators import get_outputs_generator

import mmcv
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
def _prepare_data(img, img_transform, cfg, device='cuda:0'):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,scale=(1333,800),flip=False,keep_ratio=True)
        #scale=cfg.data.test.img_scale)
        #keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])

def save_layer_outputs(model, hooks, graph, layer_name, input_folder, input_name, out_folder, use_gpu, image_size):

    img_cv = cv2.imread(join(abspath(input_folder), input_name))
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]

    if image_size is not None:
        width = image_size[-1]
        height = image_size[-2]

    img = cv2.resize(img, (width, height))

    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    img_tensor = torch.tensor(img, dtype=torch.float32)

    img_fna = mmcv.imread(join(abspath(input_folder), input_name))
    config_file = './fna_retinanet_fpn_retrain.py'
    cfg = mmcv.Config.fromfile(config_file)
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    data = _prepare_data(img_fna, img_transform, cfg, device='cuda:0')
    outputs = model(return_loss=False, rescale=True, **data)

    #outputs = model(img_tensor)

    layers = graph["config"]["layers"]
    layer_id = None
    for layer in layers:
        if layer["name"] == layer_name:
            config =  layer["config"]
            if config !="None" and "layer_id" in config:
                layer_id = config["layer_id"]
                break
    
    results = []
    if layer_id != None:
        for hook in hooks:
            if hook.layer_id == layer_id:
                channel = np.shape(hook.output)[1]
                max_channel = min([channel, channel])
                for channel in range(max_channel):
                    filename = save_layer_img(hook.output[0,channel,:,:], layer_name, channel, out_folder, input_name)
                    results.append(filename)
                break
    
    return results
