cmake_minimum_required(VERSION 3.20)

project(ONNX_Runtime)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

add_executable(onnx_model main.cpp onnx_model.cpp)
target_include_directories(onnx_model PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnx_model PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBRARIES})
