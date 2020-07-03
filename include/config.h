/*
 all configurations here!
*/

#ifndef CONFIG_H_
#define CONFIG_H_

#include <iostream>
#include <stdio.h>
#include <fstream>

const std::string modelFolder= "/home/ali/ProjLAB/yoloTvm/cpp/Yolo_object_detectio_c-/fd_tvm/output/darknet/fd_light/";
const std::string detectionMDLname = "fd_light";



const int img_h = 416;
const int img_w = 416;
const float scale = 1/255.F;

const bool swapRB = false;



#endif
