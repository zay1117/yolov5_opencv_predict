# Opencv推理yolov5（onnx）

## 1.所需环境

Linux                22.04（windows也可也使用）

g++                   11.4.0

cmake              3.22.1

CUDA               11.8

OpenCV           4.9.0

## 2. 模型准备

**模型文件位于weight文件夹中，有yolov5s.onnx和yolov5s_smp.onnx两种，我们使用yolov5s_smp.onnx简化后的模型进行预测；**

在使用yolov5s.onnx会报错是因为在处理 ONNX 模型中的 `Floor` 节点时发生错误，具体来说是由于类型不匹配或尺寸不一致导致的，所以需要使用onnxsim对onnx模型进行简化处理，以适应OpenCV的需要。

```undefined
(base) zhangao@zhangao-HP-Pavilion-Notebook:~/code/yolov5_opencv_predict/cmake-build-debug$ ./yolov5_opencv_predict 1 /home/zhangao/code/yolov5_opencv_predict/image/street.jpg [ERROR:0@0.097] global onnx_importer.cpp:1031 handleNode DNN/ONNX: ERROR during processing node with 1 inputs and 1 outputs: [Floor]:(onnx_node!/model.11/Floor) from domain='ai.onnx' terminate called after throwing an instance of 'cv::Exception'  what():  OpenCV(4.9.0) /home/zhangao/software/opencv-4.9.0/modules/dnn/src/onnx/onnx_importer.cpp:1053: error: (-2:Unspecified error) in function 'handleNode' > Node [Floor@ai.onnx]:(onnx_node!/model.11/Floor) parse error: OpenCV(4.9.0) /home/zhangao/software/opencv-4.9.0/modules/dnn/src/layers/elementwise_layers.cpp:260: error: (-215:Assertion failed) src.size == dst.size && src.type() == dst.type() && src.isContinuous() && dst.isContinuous() && src.type() == CV_32F in function 'forward' >  已中止 (核心已转储)
```
**step1.安装onnxsim包**

```undefined
pip install onnx-simplifier
```
**step2.加载onnx文件，simplify处理后重新保存，代码如下：**
```python
from onnxsim import simplify
onnx_model = onnx.load(output_path)  # onnx原模型地址
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path) # output_path为简化后的模型，简化后的模型中Floor节点会被优化
print('finished exporting onnx')
```
也可以直接使用命令行进行简化
```undefined
onnxsim yolov5s.onnx yolov5s_smp.onnx
```
**step3.使用Netron查看模型结构，发现Floor结构已经被优化掉，可以直接使用优化后的onnx模型进行预测。**

## 3 程序运行

**step1.Opencv安装配置：主要是OpenCV的下载、编译和环境配置相关，网上有很多教程**

**step2.程序文件介绍:**

**1.detecor.h主要是配置头文件**

```python
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
```

**2.detector.cpp是预测逻辑的具体实现**
```python
#include "detector.h"
#include <iostream>

// YOLOv5构造函数，初始化模型配置和后端设置
YOLOv5::YOLOv5(Configuration config, bool isCuda)
{
    confThreshold = config.confThreshold;  // 设置置信度阈值
    nmsThreshold = config.nmsThreshold;    // 设置非极大值抑制阈值
    objThreshold = config.objThreshold;    // 设置目标检测阈值
    net = readNet(config.modelpath);       // 加载YOLOv5模型

    // 根据是否使用CUDA设置DNN后端和目标
    if (isCuda) {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
        cout << "Using CUDA backend" << endl;
    } else {
        net.setPreferableBackend(DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
    num_classes = sizeof(classes) / sizeof(classes[0]);  // 类别数量
    inpWidth = 640;  // 输入图像宽度
    inpHeight = 640; // 输入图像高度
}

// 调整图像尺寸，保持宽高比并添加填充
Mat YOLOv5::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = inpHeight;
    *neww = inpWidth;
    Mat dstimg;
    
    // 如果保持宽高比且图像不是正方形，进行缩放和填充
    if (keep_ratio && srch != srcw) {
        float hw_scale = (float)srch / srcw;  // 宽高比例
        if (hw_scale > 1) {  // 高大于宽
            *newh = inpHeight;
            *neww = int(inpWidth / hw_scale);
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *left = int((inpWidth - *neww) * 0.5);  // 计算左右填充
            copyMakeBorder(dstimg, dstimg, 0, 0, *left, inpWidth - *neww - *left, BORDER_CONSTANT, 114);
        } else {  // 宽大于高
            *newh = int(inpHeight * hw_scale);
            *neww = inpWidth;
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *top = int((inpHeight - *newh) * 0.5);  // 计算上下填充
            copyMakeBorder(dstimg, dstimg, *top, inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
        }
    } else {
        resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);  // 不需要保持比例时，直接缩放
    }
    return dstimg;
}

// 在图像上绘制预测框和标签
void YOLOv5::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)
{
    // 绘制矩形框
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);
    
    // 格式化标签内容
    string label = format("%.2f", conf);
    label = classes[classid] + ":" + label;
    
    // 绘制标签文本
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

// YOLOv5目标检测函数
void YOLOv5::detect(Mat& frame)
{
    int newh = 0, neww = 0, padh = 0, padw = 0;
    // 调整输入图像大小
    Mat dstimg = resize_image(frame, &newh, &neww, &padh, &padw);
    // 创建blob，将图像标准化并准备作为模型输入
    Mat blob = blobFromImage(dstimg, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
    net.setInput(blob);  // 将blob作为网络输入
    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());  // 前向传播获取输出结果

    // 存储检测结果的容器
    vector<float> confidences;
    vector<Rect> boxes;
    vector<int> classIds;
    
    // 计算缩放比例
    float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
    float* pdata = (float*)outs[0].data;  // 获取输出数据
    int num_proposal = outs[0].size[1];   // 提案数量
    int out_dim2 = outs[0].size[2];       // 输出维度
    
    if (outs[0].dims > 2)
        outs[0] = outs[0].reshape(0, num_proposal);  // 展开输出维度
    
    // 遍历每个提案，获取目标检测信息
    for (int i = 0; i < num_proposal; ++i) {
        int index = i * out_dim2;
        float obj_conf = pdata[index + 4];  // 目标置信度
        if (obj_conf > objThreshold) {
            Mat scores(1, num_classes, CV_32FC1, pdata + index + 5);
            Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);  // 获取最大类别得分
            max_class_socre *= obj_conf;  // 计算最终置信度
    
            if (max_class_socre > confThreshold) {  // 过滤低置信度目标
                float cx = pdata[index];  // 中心x
                float cy = pdata[index + 1];  // 中心y
                float w = pdata[index + 2];  // 宽度
                float h = pdata[index + 3];  // 高度
                int left = int((cx - padw - 0.5 * w) * ratiow);  // 计算左上角坐标
                int top = int((cy - padh - 0.5 * h) * ratioh);   // 计算左上角坐标
                confidences.push_back((float)max_class_socre);    // 记录置信度
                boxes.push_back(Rect(left, top, (int)(w * ratiow), (int)(h * ratioh)));  // 记录框
                classIds.push_back(classIdPoint.x);  // 记录类别ID
            }
        }
    }
    
    // 使用非极大值抑制筛选最终的检测框
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    // 绘制最终检测框和标签
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classIds[idx]);
    }
}
```
**3.main.cpp主函数**
```python
#include "detector.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 处理图像的函数
void processImage(YOLOv5& yolo_model, const string& imgpath) {
    Mat srcimg = imread(imgpath);
    if (srcimg.empty()) {
        cerr << "Image not found!" << endl;
        return;
    }

    double timeStart = (double)getTickCount();
    yolo_model.detect(srcimg);
    double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    
    imwrite("result_cpu.jpg", srcimg);
    cout << "Detection time: " << nTime << "s" << endl;
    
    static const string kWinName = "YOLOv5 Detection";
    namedWindow(kWinName, WINDOW_NORMAL);
    imshow(kWinName, srcimg);
    waitKey(0);
    destroyAllWindows();
}

// 处理摄像头的函数
void processCamera(YOLOv5& yolo_model) {
    VideoCapture cap(0); // 打开默认摄像头
    if (!cap.isOpened()) {
        cerr << "Camera not opened!" << endl;
        return;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "Empty frame captured!" << endl;
            break;
        }
    
        double timeStart = (double)getTickCount();
        yolo_model.detect(frame);
        double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    
        static const string kWinName = "YOLOv5 Detection";
        namedWindow(kWinName, WINDOW_NORMAL);
        imshow(kWinName, frame);
        cout << "Detection time: " << nTime << "s" << endl;
    
        if (waitKey(1) == 27) { // 按 'Esc' 键退出
            break;
        }
    }
    
    cap.release();
    destroyAllWindows();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <mode> [image_path]" << endl;
        cerr << "Mode: 1 for image, 2 for camera" << endl;
        return -1;
    }

    int mode = atoi(argv[1]);
    Configuration yolo_nets = { 0.3,
                                0.5,
                                0.3,
                                "/home/zhangao/code/yolov5_opencv_predict/weight/yolov5s.onnx" };
    YOLOv5 yolo_model(yolo_nets, false);
    
    if (mode == 1) {
        if (argc < 3) {
            cerr << "Image path required for image mode!" << endl;
            return -1;
        }
        string imgpath = argv[2];
        processImage(yolo_model, imgpath);
    } else if (mode == 2) {
        processCamera(yolo_model);
    } else {
        cerr << "Invalid mode!" << endl;
        return -1;
    }
    
    return 0;
}
```

**4.CMakeLists.txt**
```undefined
cmake_minimum_required(VERSION 3.10)
project(yolov5_opencv_predict)

set(CMAKE_CXX_STANDARD 17)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 包含OpenCV头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(yolov5_opencv_predict
               detector.cpp
               main.cpp)

# 链接OpenCV库,OpenCV库已经配置好环境变量
target_link_libraries(yolov5_opencv_predict ${OpenCV_LIBS})
```

**step3.程序运行:**

首先运行 CMake 配置和构建命令。

```undefined
mkdir build
cd build
cmake ..
make
```

之后 运行生成的可执行文件，当前有两种运行方式分别是图像模式和摄像头模式（摄像头预测涉及CUDA，没配置CUDA预测帧数很低）。

**图像模式**：提供图像路径作为参数。

```undefined
./yolov5_opencv_predict 1 path_to_image.jpg
```
**摄像头模式**：不需要额外的参数。

```undefined
./yolov5_opencv_predict 2
```

<img src=".\md_photo\predict_image.jpg" width = "80%">
<img src=".\md_photo\pridect_video.jpg" width = "80%">
