from __future__ import print_function

import json

import platform

import os, cv2
import  numpy as np
from os.path import abspath, dirname, join
import webbrowser

from flask import Flask, send_from_directory
from flask.json import jsonify
from flask_cors import CORS
import torch

try:
    from gevent.wsgi import WSGIServer
except ImportError:
    from gevent.pywsgi import WSGIServer


from quiver_engine.util import (
    load_img, safe_jsonify,
    validate_launch
)

from quiver_engine.model_utils import make_dot

from quiver_engine.file_utils import list_img_files, save_layer_img
from quiver_engine.vis_utils1 import save_layer_outputs

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

def get_app(model, hooks, classes, top, html_base_dir, use_gpu, image_size, temp_folder='./tmp', input_folder='./',
            mean=None, std=None):
    '''
    The base of the Flask application to be run
    :param model: the model to show
    :param classes: list of names of output classes to show in the GUI.
        if None passed - ImageNet classes will be used
    :param top: number of top predictions to show in the GUI
    :param html_base_dir: the directory for the HTML (usually inside the
        packages, quiverboard/dist must be a subdirectory)
    :param temp_folder: where the temporary image data should be saved
    :param input_folder: the image directory for the raw data
    :param mean: list of float mean values
    :param std: lost of float std values
    :return:
    '''

    # single_input_shape, input_channels = get_input_config(model)
    app = Flask(__name__)
    app.threaded = True
    CORS(app)

    '''
        prepare model
    '''
    width = 224
    height = 224
    if image_size is not None:
        width = image_size[-1]
        height = image_size[-2]
    #x = torch.zeros(1, 3, width, height, dtype=torch.float, requires_grad=False).cuda()
    x = torch.zeros(375, 500, 3, dtype=torch.float, requires_grad=False).cuda()

    if use_gpu and torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()

    config_file = './fna_retinanet_fpn_retrain.py'
    cfg = mmcv.Config.fromfile(config_file)
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    img=mmcv.imread('./data/Cat/0.jpg')

    data = _prepare_data(img.astype(np.float32), img_transform, cfg, device='cuda:0')
    out = model(return_loss=False, rescale=True, **data)
    #print(out[0][0])
    graph = make_dot(out[0][0], params = dict(model.named_parameters()))

    '''
        Static Routes
    '''
    @app.route('/')
    def home():
        return send_from_directory(
            join(html_base_dir, 'quiverboard/dist'),
            'index.html'
        )

    @app.route('/<path>')
    def get_board_files(path):
        return send_from_directory(
            join(html_base_dir, 'quiverboard/dist'),
            path
        )

    @app.route('/temp-file/<path>')
    def get_temp_file(path):
        return send_from_directory(abspath(temp_folder), path)

    @app.route('/input-file/<path>')
    def get_input_file(path):
        return send_from_directory(abspath(input_folder), path)

    '''
        Computations
    '''
    @app.route('/model')
    def get_config():
        # print (jsonify(json.loads(model.to_json())))
        # print("test-------------")

        # model_file =  "/home/user/ANS/QuiverTest/model.json"
        # model_file = "/home/user/ANS/pytorch_model_vis/model_1.json"
        # with open(model_file, "r") as f:
        #     return jsonify(json.loads(f.read()))
        return jsonify(graph)

    @app.route('/inputs')
    def get_inputs():
        return jsonify(list_img_files(input_folder))

    @app.route('/layer/<layer_name>/<input_path>')
    def get_layer_outputs(layer_name, input_path):
        print (layer_name, input_path)
        
        results = save_layer_outputs(model, hooks, graph, 
                                    layer_name, input_folder,
                                    input_path, temp_folder, use_gpu, image_size)
        return jsonify(results)
        

    @app.route('/predict/<input_path>')
    def get_prediction(input_path):
        # print ("prediction", input_path)
        results = [[("sa","bot_34", 0.2)],[("sa","bot_35", 0.6)]]
        return safe_jsonify(results)

    return app


def run_app(app, port=5000):
    http_server = WSGIServer(('', port), app)
    # webbrowser.open_new('http://localhost:' + str(port)) #
    http_server.serve_forever()


def launch(model, hooks, input_folder='./', use_gpu=False, image_size=None, classes=None, top=5, temp_folder='./tmp', 
           port=5000, html_base_dir=None, mean=None, std=None):
    if platform.system() is 'Windows':
        temp_folder = '.\\tmp'
        os.system('mkdir %s' % temp_folder)
    else:
        os.system('mkdir -p %s' % temp_folder)


    html_base_dir = html_base_dir if html_base_dir is not None else dirname(abspath(__file__))
    # print(html_base_dir)
    validate_launch(html_base_dir)

    return run_app(
        get_app(
            model, hooks, classes, top,
            html_base_dir=html_base_dir,
            use_gpu=use_gpu,
            image_size=image_size,
            temp_folder=temp_folder,
            input_folder=input_folder,
            mean=mean, std=std
        ),
        port
    )
