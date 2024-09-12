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
