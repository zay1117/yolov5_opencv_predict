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