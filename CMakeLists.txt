cmake_minimum_required(VERSION 3.22)
project(ort_runner)

find_package(onnxruntime REQUIRED COMPONENTS onnxruntime_providers_cuda OPTIONAL_COMPONENTS onnxruntime_providers_tensorrt)

add_executable(ort_runner ort_runner/ort_runner.cpp)
target_link_libraries(ort_runner PRIVATE onnxruntime::onnxruntime)
