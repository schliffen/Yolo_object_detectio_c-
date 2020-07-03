/*
 all configurations here!
*/

#ifndef CONFIG_H_
#define CONFIG_H_

#include <iostream>
#include <stdio.h>
#include <fstream>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc.hpp>


#include "tvm/runtime/module.h"
#include "tvm/runtime/registry.h"
#include "tvm/runtime/packed_func.h"
#include "dlpack/dlpack.h"



// tvm configuration!
constexpr int dtype_code = kDLFloat;
constexpr int dtype_bits = 32;
constexpr int dtype_lanes = 1;
constexpr int device_type = kDLGPU;
constexpr int device_id = 0;
constexpr int in_ndim = 4;

const std::string modelFolder= "../fd_tvm/output/darknet/fd_light/";
const std::string detectionMDLname = "fd_light";



const int img_h = 416;
const int img_w = 416;
const float scale = 1/255.F;

const bool swapRB = false;



#endif
