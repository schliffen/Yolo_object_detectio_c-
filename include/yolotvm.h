//
//
//
#include <iostream>

#include "tvm/runtime/module.h"
#include "tvm/runtime/registery.h"
#include "tvm/runtime/packed_func.h"
#include "dlpack/dlpack.h"
#include "config.h"


// tvm configuration!
constexpr int dtype_code = kDLFloat;
constexpr int dtype_bits = 32;
constexpr int dtype_lanes = 1;
constexpr int device_type = kDLGPU;
constexpr int device_id = 0;
constexpr int in_ndim = 4;
//

class TVM_Inference{
    private:
        std::unique_ptr<tvm::runtime::Module> handle;
        std::tuple<int, int, int, int> inpSize;

    public:
        TVM_Inference( const std::string modelFolder, const std::string detectionMDLname, std::tuple<int, int, int, int> mdlSize);
        tvm::runtime::PackedFunc forward( cv::Mat inpTensor );

    protected:
        DLTensor *input;  
}

class yoloinference{
    public:
        void yoloinference(){};
        void detector(cv::Mat);
        TVM_Inference yolodet{ modelFolder, detectionMDLname, std::make_tuple(1, 3, img_w, img_h)};

}
