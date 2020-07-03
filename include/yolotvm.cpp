#include <cstddef>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include "yolotvm.h"


TVM_Inference::TVM_Inference(  std::string modelFolder,  std::string detectionMDLname, std::tuple<int, int, int, int> mdlSize ) {
        //  loading the model
        this->inpSize = mdlSize;
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile( modelFolder + detectionMDLname + ".so" );
        // modelFolder + format("/deploy_lib_%s.so", arcface_model)
        //load graph
        std::ifstream json_in( modelFolder + detectionMDLname + ".json" );
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                              device_type, device_id);
        this->handle.reset(new tvm::runtime::Module(mod));
        //load param
        std::ifstream params_in( modelFolder + detectionMDLname + ".params" , std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    };
//
tvm::runtime::PackedFunc TVM_Inference::forward(cv::Mat tensor ) {
        /* inputs:
         tensor: image blob with inverted channels
         input: tvm DL tensor
        */

        const int64_t in_shape[in_ndim] = {std::get<0>(this->inpSize), std::get<1>(this->inpSize), std::get<2>(this->inpSize), std::get<3>(this->inpSize)};
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &this->input);//
        TVMArrayCopyFromBytes(this->input, tensor.data, std::get<1>(this->inpSize) * std::get<2>(this->inpSize) * std::get<3>(this->inpSize) * sizeof(float));
        tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", this->input);
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        TVMArrayFree(this->input);
        return get_output;
    };


void yoloinference::ydetector(cv::Mat frame){
    static cv::Mat tensor;
    std::cout<< "inference 1 \n";
    // Create a 4D blob from a frame.
    // if (inpSize.width <= 0) inpSize.width = frame.cols;
    // if (inpSize.height <= 0) inpSize.height = frame.rows;
    cv::dnn::blobFromImage(frame, tensor, scale, cv::Size(img_w, img_h), cv::Scalar(), swapRB, false);

    tvm::runtime::PackedFunc tvmout = this->dethandler.forward( tensor );

    // tvm::runtime::NDArray Y;
    tvm::runtime::NDArray res_1 = tvmout(3);
    // tvmout(0, Y); // this is for DLtensor

    // const tvm::runtime::NDArray res = tvmout(0);
    // std::cout<< tvmout(0)->size() << std::endl;
    std::cout<< res_1->shape[1] <<" " << res_1->shape[2] << "  " << res_1->shape[3] << " " << res_1->shape[4] << std::endl;

    // int64_t out_shape[1, this->]


    std::cout<< "inference pased \n";



}


    // RetinaOutput detect(cv::Mat image) {
    //     constexpr int dtype_code = kDLFloat;
    //     constexpr int dtype_bits = 32;
    //     constexpr int dtype_lanes = 1;
    //     constexpr int device_type = kDLGPU;//kDLGPU
    //     constexpr int device_id = 0;
    //     DLTensor *x;
    //     int in_ndim = 4;
    //     int in_c = 3, in_h = 640, in_w = 640;
    //     int64_t in_shape[4] = {1, in_c, in_h, in_w};
    //     TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

    //     int64_t w1 = ceil(in_w / 32.0), w2 = ceil(in_w / 16.0), w3 = ceil(in_w / 8.0), h1 = ceil(
    //             in_h / 32.0), h2 = ceil(in_h / 16.0), h3 = ceil(in_h / 8.0);
    //     int out_num = (w1 * h1 + w2 * h2 + w3 * h3) * (4 + 8 + 20);
    //     tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();
    //     int total_input = 3 * in_w * in_h;
    //     float *data_x = (float *) malloc(total_input * sizeof(float));
    //     //float* y_iter = (float*)malloc(out_num*4);
    //     if (!image.data)
    //         printf("load error");
    //     //input data
    //     cv::Mat resizeImage;
    //     cv::resize(image, resizeImage, cv::Size(in_w, in_h), cv::INTER_AREA);
    //     cv::Mat input_mat;
    //     resizeImage.convertTo(input_mat, CV_32FC3);
    //     //cv::cvtColor(input_mat, input_mat, cv::COLOR_BGR2RGB);
    //     cv::Mat split_mat[3];
    //     cv::split(input_mat, split_mat);
    //     memcpy(data_x, split_mat[2].ptr<float>(), input_mat.cols * input_mat.rows * sizeof(float));
    //     memcpy(data_x + input_mat.cols * input_mat.rows, split_mat[1].ptr<float>(),
    //            input_mat.cols * input_mat.rows * sizeof(float));
    //     memcpy(data_x + input_mat.cols * input_mat.rows * 2, split_mat[0].ptr<float>(),
    //            input_mat.cols * input_mat.rows * sizeof(float));
    //     TVMArrayCopyFromBytes(x, data_x, total_input * sizeof(float));
    //     // get the function from the module(set input data)
    //     tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
    //     set_input("data", x); // ------------------------data
    //     // get the function from the module(run it)
    //     tvm::runtime::PackedFunc run = mod->GetFunction("run");
    //     run();
    //     tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
    //     std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    //     for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
    //         int stride = _feat_stride_fpn[i];
    //         ac[i].Init(stride, anchor_cfg[stride], false);
    //     }
    //     std::vector<Anchor> proposals;
    //     proposals.clear();


