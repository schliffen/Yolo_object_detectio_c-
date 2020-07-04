#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: can
"""

# import cv2
# numpy and matplotlib
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
import tvm
from tvm import relay
import os
from time import time

from tvm import autotvm
from tuner import TUNER


######################################################################
# Initials
# -----------------------
CALIBRATION_SAMPLES = 16
num_threads = 8
os.environ["TVM_NUM_THREADS"] = str(num_threads)

net_props = {
    'model_name': 'fd_light',
    'framework': 'darknet',
    'model_dir': 'models',
    'batch_size': 1,
    'width': 416,
    'height': 416,
    'channel': 3,
    'dtype': 'float32',
    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)),
    'transfer_learning': True,
    'quantize': False,
    'try_winograd': True}

######################################################################
# Prepare the environment
# -----------------------
target = tvm.target.create('cuda')
tuner = TUNER(**net_props)
net, mod, params = tuner.get_network.from_darknet()
# tuner.get_network.from_mxnet()
# tuner.get_network.from_torch()

if tuner.quantize:
    mod = tuner.quantize_model(mod, params)

# print("Extract tasks...")
# tasks = tuner.extract_tasks(mod, params, target)

# print("Start tuning...")
# tuner.tune_tasks(tasks)

print("Export library...")
tuner.export_library(mod, params, target)


# tuner.evaluate()

# img_path = "./66429ef6e83191b4.jpg"
# print("test the model on an image...")
# tuner.test_darknet_yolo(img_path)