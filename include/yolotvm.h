//
//
//

#include "config.h"



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
};

class yoloinference{
    public:
        // void yoloinference(){};
        void ydetector(cv::Mat);
        TVM_Inference dethandler{ modelFolder, detectionMDLname, std::make_tuple(1, 3, img_w, img_h)};

};
