#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// 配置结构体，包含YOLOv5的超参数和模型路径
struct Configuration {
    float confThreshold;  // 置信度阈值
    float nmsThreshold;   // 非极大值抑制阈值
    float objThreshold;   // 目标检测阈值
    string modelpath;     // 模型文件路径
};

// YOLOv5类，用于目标检测
class YOLOv5 {
public:
    // 构造函数，接收配置参数和是否使用CUDA的标志
    YOLOv5(Configuration config, bool isCuda = false);
    
    // 检测函数，处理输入帧并进行目标检测
    void detect(Mat& frame);

private:
    // 私有成员变量，用于存储配置参数
    float confThreshold;  // 置信度阈值
    float nmsThreshold;   // 非极大值抑制阈值
    float objThreshold;   // 目标检测阈值
    int inpWidth;         // 网络输入宽度
    int inpHeight;        // 网络输入高度
    int num_classes;      // 类别数量

    // 类别名称数组，包含80个常见检测对象类别
    string classes[80] = {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    const bool keep_ratio = true;  // 是否保持图像宽高比
    Net net;                       // OpenCV的DNN网络对象

    // 绘制预测框的函数
    void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);

    // 调整图像大小的函数
    Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
};

#endif // DETECTOR_H