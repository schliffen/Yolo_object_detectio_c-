#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: can
"""


import numpy as np
import matplotlib.pyplot as plt
import tvm
import sys
from time import time
import os
import multiprocessing as mp

# from ctypes import *
from tvm import relay
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
# from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
import torch
import torchvision
import mxnet as mx


class TUNER(object):
    
    def __init__(self, **kwargs):
        self.batch_size = 1
        self.dtype = 'float32'
        self.model_dir = 'models'
        self.transfer_learning = True
        self.quantize = False
        self.__dict__.update(kwargs)
        self.shape = (self.batch_size, self.channel, self.height, self.width)
        
        self.backup_dir = os.path.join(self.model_dir,self.framework,self.model_name)
        
        self.log_file = "./logs/"+self.framework+'/' + self.model_name + '/' + self.model_name + ".tune.log"
        if not os.path.exists("./logs/"+self.framework):
            os.mkdir("./logs/"+self.framework)
        if not os.path.exists("./logs/"+self.framework+'/' + self.model_name):
            os.mkdir("./logs/"+self.framework+'/' + self.model_name)
        if not os.path.exists(self.log_file):
            os.mknod(self.log_file)
        
        self.out_dir = "./output/" + self.framework+'/' + self.model_name + '/'
        if not os.path.exists("./output/"+self.framework):
            os.mkdir("./output/"+self.framework)
        if not os.path.exists("./output/"+self.framework+'/' + self.model_name):
            os.mkdir("./output/"+self.framework+'/' + self.model_name)
        
        mydict = self.__dict__
        self.get_network = self.get_network(**mydict)
    
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
    
    class get_network():
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            
        def from_darknet(self):
            ''' Generate TVM Module and Parameters for Darknet models '''
            cfg_path = os.path.join(self.backup_dir, self.model_name + '.cfg')
            weights_path = os.path.join(self.backup_dir, self.model_name + '.weights')
            DARKNET_LIB = __darknetffi__.dlopen(os.path.join(self.model_dir, self.framework,'libdarknet2.0.so'))
            
            net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
            
            # input_shape = [self.batch_size, net.c, net.h, net.w]
            data = np.empty(self.shape, self.dtype)
            print('\nData Shape: ', data.shape, '\n')
            
            print("Converting darknet to relay functions...")
            mod, params = relay.frontend.from_darknet(net, dtype=self.dtype, shape=data.shape)
            
            return net, mod, params
            
    
    
        def from_torch(self):
            weights_path = os.path.join(self.backup_dir, self.model_name + '.pth')
            weights = torch.load(weights_path)
            model = getattr(torchvision.models, self.model_name)(pretrained=False)
            model.load_state_dict(weights)
            model = model.eval()
            
            # We grab the TorchScripted model via tracing
            input_data = torch.randn(self.shape, dtype=torch.float32)
            scripted_model = torch.jit.trace(model, input_data).eval()
            
            shape_list = [('data', self.shape)]
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
            
            return mod, params
            
        
        def from_mxnet(self):
            weights_path = os.path.join(self.backup_dir, self.model_name)
            data = np.random.uniform(size=self.shape).astype(self.dtype)
            shape_dict = shape_dict = {'data': data.shape}
            sym, args, auxs = mx.model.load_checkpoint(weights_path, 0)
            mod, params = relay.frontend.from_mxnet(sym, shape_dict, arg_params=args, aux_params=auxs)
            
            return mod, params
            
    
    
    def quantize_model(self, mod, params):
        with relay.quantize.qconfig(calibrate_mode='global_scale', 
                                    global_scale=8.0,
                                    store_lowbit_output=False):
            mod = relay.quantize.quantize(mod, params=params)
        return mod


    def extract_tasks(self, mod, params, target):
        tasks = autotvm.task.extract_from_program(mod, target=target,
                                                      target_host='llvm',
                                                      params=params)
        if self.try_winograd:
            for i in range(len(tasks)):
                try:  # try winograd template
                    tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                              tasks[i].target, tasks[i].target_host)
                    input_channel = tsk.workload[1][1][1]
                    if input_channel >= 64:
                        tasks[i] = tsk
                except Exception:
                    pass
        return tasks
    

    def tune_tasks(self, tasks):
        log_filename = self.log_file
        tuner = self.tuner
        
        # create tmp log file
        tmp_log_file = log_filename + ".tmp"
        if not self.transfer_learning:
            if os.path.exists(tmp_log_file):
                os.remove(tmp_log_file)
        else:
            # select actual best logs
            if not os.path.exists(tmp_log_file):
                os.mknod(tmp_log_file)
    
        for i, tsk in enumerate(reversed(tasks)):
            prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
    
            # create tuner
            if tuner == 'xgb' or tuner == 'xgb-rank':
                tuner_obj = XGBTuner(tsk, loss_type='rank')
            elif tuner == 'ga':
                tuner_obj = GATuner(tsk, pop_size=100)
            elif tuner == 'random':
                tuner_obj = RandomTuner(tsk)
            elif tuner == 'gridsearch':
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)
    
            if self.transfer_learning:
                if os.path.isfile(tmp_log_file):
                    tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
    
            # do tuning
            tuner_obj.tune(n_trial=min(self.n_trial, len(tsk.config_space)),
                           early_stopping=self.early_stopping,
                           measure_option=self.measure_option,
                           callbacks=[
                               autotvm.callback.progress_bar(self.n_trial, prefix=prefix),
                               autotvm.callback.log_to_file(tmp_log_file)])
    
        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)
    
    
    def build_relay(self, mod, params, target):
        with autotvm.apply_history_best(self.log_file):
            print("Compiling with the best configuration logged...")
            with relay.build_config(opt_level=3):
                tuned_graph, tuned_lib, tuned_params = relay.build_module.build(
                    mod, target=target, params=params)
        return tuned_graph, tuned_lib, tuned_params
    
    
    def export_library(self, mod, params, target):
        tuned_graph, tuned_lib, tuned_params = self.build_relay(mod, params, target)
        print("exporting tuned libraries...")
        tuned_lib.export_library(self.out_dir+self.model_name+'.so')
        with open(self.out_dir+self.model_name+'.json', 'w') as f:
            f.write(tuned_graph)
        with open(self.out_dir+self.model_name+'.params', 'wb') as f:
            f.write(relay.save_param_dict(tuned_params))
    
    
    def evaluate(self, mod, params, target):
        tuned_graph, tuned_lib, tuned_params = self.build_relay(mod, params, target)

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(tuned_graph, tuned_lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=self.shape)).astype(self.dtype))
        module.set_input('data', data_tvm)
        module.set_input(**tuned_params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=20, repeat=500)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
    
    
    def test_darknet_yolo(self, img_path, net, mod, params, target):
        tuned_graph, tuned_lib, tuned_params = self.build_relay()
        # load parameters
        ctx = tvm.context(str(target), 0)
        [neth, netw] = self.shape[2:]
        data = tvm.relay.testing.darknet.load_image(img_path, netw, neth)
        font_path = "arial.ttf"
        
        # load original image
        img = tvm.relay.testing.darknet.load_image_color(img_path)
        _, im_h, im_w = img.shape
        
        # create runtime module
        module = runtime.create(tuned_graph, tuned_lib, ctx)
        
        # set inputs
        tvm_input = tvm.nd.array(data.astype(self.dtype), ctx)
        module.set_input('data', tvm_input)
        module.set_input(**params)
        
        thresh = 0.6
        nms_thresh = 0.45
        
        # execute the module
        module.run()
        # get outputs
        tvm_out = []
        for i in range(2):
            layer_out = {}
            layer_out['type'] = 'Yolo'
            # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
            layer_attr = module.get_output(i*4+3).asnumpy()
            layer_out['biases'] = module.get_output(i*4+2).asnumpy()
            layer_out['mask'] = module.get_output(i*4+1).asnumpy()
            out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                          layer_attr[2], layer_attr[3])
            layer_out['output'] = module.get_output(i*4).asnumpy().reshape(out_shape)
            layer_out['classes'] = layer_attr[4]
            tvm_out.append(layer_out)
        
        dets = tvm.relay.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh, 1, tvm_out)
        last_layer = net.layers[net.n - 1]
        tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)
        
        
        label_path = os.path.join(self.backup_dir, self.model_name + ".names")
        with open(label_path) as f:
            content = f.readlines()
        names = [x.strip() for x in content]
            
        tvm.relay.testing.yolo_detection.draw_detections(font_path, img, dets, thresh, names, last_layer.classes)
        plt.imshow(img.transpose(1, 2, 0))
        plt.show()


##############################################################################
# def calibrate_on_dataset(qmodel, params, input_dtype, input_shape):

#     profile_graph = relay.quantize.collect_stats(qmodel)

#     # init input data
#     data = np.empty(input_shape, input_dtype)


#     with relay.build_config(opt_level=3):
#         graph, lib, params = relay.build(relay.IRModule.from_expr(profile_graph), target=target)

#     [neth, netw] = input_shape[2:]

#     ######################################################################
#     # Execute on TVM Runtime
#     # ----------------------
#     # The process is no different from other examples.
#     from tvm.contrib import graph_runtime
#     module = graph_runtime.create(graph, lib, ctx)
#     module.set_input('data', tvm.nd.array(data.astype(input_dtype)))
#     module.set_input(**params)

#     num_outputs = module.get_num_outputs()
#     outputs = [[] for i in range(num_outputs)]
#     print("Network output nodes = %i" % num_outputs)

#     vcap = cv2.VideoCapture(VIDEO_FILE)
#     fps = int(vcap.get(cv2.CAP_PROP_FPS))
#     nframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

#     print("Using video '%s' frames=%i fps=%i time=%i (seconds)" % (VIDEO_FILE, nframes, fps, nframes//fps))

#     # random sample
#     indexes = np.random.randint(0, nframes-1, CALIBRATION_SAMPLES)

#     for idx, fpos in enumerate(indexes):

#         vcap.set(cv2.CAP_PROP_POS_FRAMES, fpos)

#         ret, frame = vcap.read()

#         # prepare frame
#         img = np.array(frame)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         data = cv2.resize(img, (neth, netw), interpolation = cv2.INTER_AREA)
#         data = data.transpose((2, 0, 1))
#         data = np.divide(data, 255.0)
#         data = np.flip(data, 0)

#         sys.stdout.write("\r")
#         sys.stdout.write("Extracting random frame %i out of %i (#%i total)" % (idx+1, len(indexes), nframes))
#         sys.stdout.flush()

#         # set inputs
#         module.set_input('data', tvm.nd.array(data.astype(input_dtype)))

#         # execute
#         module.run()

#         for i in range(num_outputs):
#             output = module.get_output(i).asnumpy()
#             outputs[i].append(output)

#     sys.stdout.write("\n")
#     sys.stdout.flush()

#     print("Concatenate stats...")
#     for i in range(num_outputs):
#         outputs[i] = np.concatenate(outputs[i]).reshape(-1)

#     print("Compute final KL stats...")
#     with mp.Pool() as pool:
#         scales = list(pool.map(relay.quantize.kl_divergence.kl_divergence_scale, outputs))

#     return scales


# def quantize_model(model, params, input_dtype, input_shape, qeval='power2'):

#     skip_conv_layers = [0]
# #    with relay.quantize.qconfig(store_lowbit_output=False, skip_conv_layers=skip_conv_layers):
#     with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0, weight_scale='power2', skip_conv_layers=skip_conv_layers):
#         mod = relay.quantize.quantize(model, params)
#         from tvm.relay.quantize.quantize import _bind_params
#         graph = _bind_params(model['main'], params)
#         mod = tvm.IRModule.from_expr(graph)
#         optimize = _transform.Sequential([_transform.SimplifyInference(),
#                                           _transform.FoldConstant(),
#                                           _transform.FoldScaleAxis(),
#                                           _transform.CanonicalizeOps(),
#                                           _transform.FoldConstant()])



#         with relay.build_config(opt_level=4):
#             mod = optimize(mod)
#             mod = relay.quantize.annotate()(mod)

#             # find scale
#             cache_file = '%s_%s_scales.pkl' % (VIDEO_FILE, MODEL_NAME)
#             if os.path.exists(cache_file):
#                 print("Using cached layer statistics...")
#                 with open(cache_file, 'rb') as f:
#                     scales = pickle.load(f)
#             else:
#                 print("Compute layer statistics...")
#                 scales = calibrate_on_dataset(mod['main'], params, input_dtype, input_shape)
#                 with open(cache_file, 'wb') as f:
#                     pickle.dump(scales, f)

#             if qeval == 'power2':
#                 scales = list(map(lambda scale: 2**np.math.ceil(np.math.log(scale, 2)) if scale > 0 else 1.0, scales))
#                 weight_scales = 'power2'
#             elif qeval == 'max':
#                 weight_scales = 'max'
#             else:
#                 raise ValueError("Invalid quantiziation eval: " + qeval)

#             mod['main'] = relay.quantize.calibrate(mod['main'], weight_scales=weight_scales,
#                                                   scales=scales)
#             mod = relay.quantize.realize()(mod)
#             mod = relay.transform.FoldConstant()(mod)
#     return mod