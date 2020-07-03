//
// CUDA GPU programming
//
// #include "NumCpp.hpp"
#include <iostream>
#include <cstdint>
#include <cstring>
#include <fstream>
#include "nms.hpp"
//
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc.hpp>
#include "yolotvm.h"

// #include "tvm/runtime"
//
// using namespace std;
// using namespace cv;
// using namespace dnn;

// preprocess
inline void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
                       const cv::Scalar& mean, bool swapRB);

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend);
// remove unnecessary bounding boxes
void remove_box(cv::Mat&frame, const std::vector<cv::Mat>&out);
//
// void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
void drawPred( std::vector<size_t>, std::vector<float>, std::vector<std::vector<float>>, cv::Mat&);
// draw bounding boxes
void draw_box(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

// get output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
std::vector<std::string> classes;
//  bool swapRB = false; //parser.get<bool>("rgb");
 //
 std::vector<std::vector<float>> nmsIndices_c;
 const std::vector<std::vector<float>> nms_boxes;
  

// Initialize the parameters
float confThreshold = 0.6; // Confidence threshold
float nmsThreshold = 0.3;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image
// confidence threshold
float conf_threshold = 0.6;
// nms threshold
float nms_threshold = 0.4;
// backend
int backend = 0; 
std::string source  = "video";


int main(){
    //
    // float * arr ; //= new float[2];   
    //
    // nc::NdArray<double> sMeasured = { 0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.74 , 1.2};
    // nc::NdArray<double> rateMeasured = { 0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317 };
    // std::cout<<"array min"<<sMeasured.reshape(2,4)<<std::endl;
    //
    //
    // Load names of classes

    yoloinference deployolo;
    
    std::string classesFile = "/home/ali/ProjLAB/yoloTvm/cpp/Yolo_object_detectio_c-/weights/coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    std::string modelConfiguration = "/home/ali/ProjLAB/yoloTvm/cpp/Yolo_object_detectio_c-/weights/yolov3.cfg";
    std::string modelWeights = "/home/ali/ProjLAB/yoloTvm/cpp/Yolo_object_detectio_c-/weights/yolov3.weights";
    // std::string imgsDir = "/home/ali/ProjLAB/plateSegmentation/MySegmentation/data/ukrayna_01";
    std::vector<std::string> imgList;
    // reading the image
    cv::Mat frame, blob;
    // for reading videos 
    cv::VideoCapture cap;
    int im=0;
    std::cout<< source << std::endl;
    if (source == "video"){
        
        std::string videoPath = "../../data/video_test.mp4";
        cap.open( videoPath );
    }else if (source == "image"){
        
        // cv::glob(imgsDir + "/*.jpg" , imgList, false);
        
        
    }else{
        cap.open( 0 );
    }

    

    // loading network
    // cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);

    
    
    // used vars
    bool process = true;
    std::vector<cv::Mat> outs;
    std::vector<double> layersTimes;
    cv::Mat detectedFrame;


    // for (int im=0; im< imgList.size(); im++){
    
    while (process)
    {
        // frame=cv::imread(imgList[im]);
        
        if (source == "video"){
            std::cout<< "reading video \n";
            cap >> frame;
        } else if (source == "image"){

            
          frame = cv::imread(imgList[im]);
          im++;
        }
        

        if (frame.empty())
            process = false;

        cv::imshow("frame", frame);
        cv::waitKey(0);
        cv::resize(frame, frame, cv::Size(416, 416));

        std::cout<< " frame resized \n";

        deployolo.ydetector(frame);
        
        
        
        /*
        preprocess( frame, net, cv::Size(608, 608), 1/155.F, cv::Scalar(0,0,0), true);
        // convert image to blob
        // cv::dnn::blobFromImage(frame, blob, 1/255.F, cv::Size(inpWidth, inpHeight), cv::Scalar(), true, false);
        // net.setInput(blob);
        
        net.forward(outs, getOutputsNames(net));

        // std::cout<< " out size: "<< outs[2].size() << std::endl;
    
        // postprocess
        postprocess(frame, outs, net, backend);


        // remove_box(frame, outs);

        
        double freq = cv::getTickFrequency() / 1000;

        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time for a frame : %.2f ms", t);
        cv::putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        
        
        frame.convertTo(detectedFrame, CV_8U);
        static const std::string kWinName = "Deep learning object detection in OpenCV";
        
        cv::imshow(kWinName, frame);
        cv::waitKey(0);
        */

    };
    // cv::destroyAllWindows();


    return 0;
}


inline void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
                       const cv::Scalar& mean, bool swapRB)
{
    static cv::Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;
    cv::dnn::blobFromImage(frame, blob, scale, cv::Size(inpWidth, inpWidth), cv::Scalar(), swapRB, false);

    // Run a model.
    net.setInput(blob, "", 1.0, mean);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        cv::resize(frame, frame, inpSize);
        cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}


void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::cout<< " out layer: " << outLayerType << std::endl;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> boxes_vect;

    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back( cv::Rect(left, top, width, height));
                    boxes_vect.push_back({left, top, width, height});
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        // std::cout<< "--------------------- \n";

        for (size_t i = 0; i < outs.size(); ++i)
        {

            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            
            // std::cout<< "out rows: " << outs[i].rows<< " out cols: "<< outs[i].cols << " outlayer region\n";

            float* data = (float*)outs[i].data;


            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                // std::cout<< "j="<<j <<" data: " << data[j] << std::endl;

                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols); // searching in scores range 5,6, ...

                cv::Point classIdPoint;
                double confidence;
                
                // find the maximum scores and their location
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                // std::cout<< scores.size() << std::endl;
                // std::cout<< scores << std::endl;

                if (confidence > confThreshold)
                {
                    // std::cout<< data[0] << " - " << data[1] << " - " << data[2] << " - " << data[3] << std::endl;
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width   = (int)(data[2] * frame.cols);
                    int height  = (int)(data[3] * frame.rows); // data[4] is for background class
                    int left    = centerX - width  / 2;
                    int top     = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                    //
                    boxes_vect.push_back({left, top, width, height});
                }
            }
        }
    }
    else
        CV_Error(cv::Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (outLayers.size() > 1 || (outLayerType == "Region" && backend != cv::dnn::DNN_BACKEND_OPENCV))
    {

        std::cout<< " post process\n";
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            // iterate over class ids
            if (confidences[i] >= confThreshold)
            {
                std::cout<< i << " selected score index \n";
                class2indices[classIds[i]].push_back(i);
            }
        }

        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;

        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {

            std::vector<std::vector<float>> localBoxesArray;
            std::vector<float> outlocalConfidences;
            std::vector<std::vector<float>> outnmsBoxes;
            std::vector<size_t> outnmsClassIds;
            //    
            std::vector<cv::Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;


            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxesArray.push_back( boxes_vect[ classIndices[i] ]);
                localBoxes.push_back( boxes[ classIndices[i] ]);
                localConfidences.push_back(confidences[ classIndices[i] ]);
            }
            std::vector<int> nmsIndices;

            // non maximum suppression
            cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);

            // preparing the inputs

            // std::cout<< it->first << std::endl;
            //custom nonmax supp
            // const int classidi = it->first;
            const int nmsstatus = nms( localBoxesArray, confThreshold, nmsThreshold, it->first, localConfidences, outnmsClassIds, outlocalConfidences, outnmsBoxes);

            // plotting the rectangle
            if (nmsstatus)
                drawPred( outnmsClassIds, outlocalConfidences,  outnmsBoxes, frame);


            // std::cout<< "nms nmsboxes_c shape 0 : " << outnmsBoxes.size() << std::endl;
            // std::cout<< "nms outLocalConfidence shape 1 : " << outlocalConfidences.size() << std::endl;
            // for(int k=0; k<nmsIndices_c.size(); k++){
                // std::cout<<"nms indices: "<<  nmsIndices_c[k][0] << std::endl;
            // }
            
            /*
            std::vector<std::vector<float>>::iterator rows;
            std::vector<float>::iterator cols; 

            for (rows = outnmsBoxes.begin(); rows !=outnmsBoxes.end(); rows++){
                for(cols=rows->begin(); cols!=rows->end(); cols++){
                    std::cout<< *cols << std::endl;
                
                }

            }
            */
            
            // for (size_t i = 0; i < nmsIndices.size(); i++)
            // {
                // size_t idx   = nmsIndices[i];

                /*
                std::vector<std::vector<float>>::iterator rows;
                std::vector<float>::iterator cols; 

                for (rows = nmsIndices_c.begin(); rows !=nmsIndices_c.end(); rows++){
                    for(cols=rows->begin(); cols!=rows->end(); cols++){
                        std::cout<< *cols << std::endl;
                    }

                }
                */    
                // std::cout<< "indx_C: " << nmsIndices_c[ 0 ] << std::endl;
                // size_t idx_c = nmsIndices_c[i];
                
                // nms_boxes.push_back( localBoxesArray[idx_c] );

                // nmsBoxes.push_back(localBoxes[idx]);
                
                // nmsConfidences.push_back(localConfidences[idx]);
                // nmsClassIds.push_back(it->first);
            // }



        }
        
        // boxes_vect = nms_boxes;
        // boxes = outnmsBoxes;
        // classIds = outnmsClassIds;
        // confidences = outlocalConfidences;
    }

    
    // for (std::vector<T>::size_type idx = 0; idx != outnmsClassIds.size(); idx++)
    // {
        // outnmsClassIds, outlocalConfidences, outnmsBoxes
        // std::cout << "class id: " << outlocalConfidences[idx] << std::endl;
        // cv::Rect box = boxes_vect[idx];
        // drawPred(classIds[idx], confidences[idx], box.x, box.y,
                //  box.x + box.width, box.y + box.height, frame);
    // }

    

    /*
    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
    */
}

void drawPred(std::vector<size_t> outnmsClassIds, std::vector<float> outlocalConfidences, std::vector<std::vector<float>> outnmsBoxes, cv::Mat& frame)
{
    // writing the rectangles
    float rectangle[4]={};
    std::string label;
    int baseLine;
    cv::Size labelSize;    
    for (std::pair<std::vector<std::vector<float>>::iterator, std::vector<float>::iterator> iter (outnmsBoxes.begin(),  outlocalConfidences.begin());
         iter.first!=outnmsBoxes.end() && iter.second!=outlocalConfidences.end(); ++iter.first, ++iter.second){
                    int ri = 0;
                    for(std::vector<float>::iterator cols=iter.first->begin(); cols!=iter.first->end(); cols++){
                        std::cout<< *cols << std::endl;
                        rectangle[ri] =  *cols;
                        ri++;
                    }
                    // writing on image
                    cv::rectangle(frame, cv::Point(rectangle[0], rectangle[1]), cv::Point(rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), cv::Scalar(0, 255, 0));
                    //
                    label = cv::format("%.2d", iter.second);
                    // label = classes[ids] + ": " + label;   
                    labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    //    
                    // top = cv::max(rectangle[1], labelSize.height);
                    cv::rectangle(frame, cv::Point(rectangle[0], rectangle[1] - labelSize.height),
                            cv::Point(rectangle[0] + labelSize.width, rectangle[1] + baseLine), cv::Scalar::all(255), cv::FILLED);
                    cv::putText(frame, label, cv::Point(rectangle[0], rectangle[1]), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());    
                    
                    // labelSize.clear();
                    // baseLine.clear();

                
                }
    /*
    std::string label;
    int baseLine;
    cv::Size labelSize;
    for (const float conf : outlocalConfidences){
        label = cv::format("%.2f", conf);
        label = classes[classId] + ": " + label;
        
        labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
    }
    */


    // cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

    /*
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - labelSize.height),
              cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
    */
}

/*
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - labelSize.height),
              cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}
*/

void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}


std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
{
    static std::vector<cv::String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        std::vector<cv::String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}



void remove_box(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

        
            if (confidence > conf_threshold)
            {
                // std::cout<< "scores: "<< scores << "s conf: "<< confidence <<  "s classIdPoint: "<< classIdPoint <<std::endl;
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back( cv::Rect(left, top, width, height) );
            }
        }
    }
        // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;


    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);




    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        draw_box(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }

}




// Draw the predicted bounding box
void draw_box(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}


