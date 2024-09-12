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