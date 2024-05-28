# onnx-misuse
Some examples of general purpose algorithms implemented as ONNX pipelines.

`CMakeLists.txt` provided uses `onnxruntimeConfig.cmake` file which is only available if ONNX Runtime is built from sources. This file is not presented in github releases binary packages. To build ONNX Runtime with CUDA and TensorRT support use a command like this:
```
./build.sh --config Release --build_shared_lib --compile_no_warning_as_error --skip_submodule_sync --use_cuda --use_tensorrt --cmake_generator Ninja --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu --tensorrt_home /usr/lib/x86_64-linux-gnu
```
Note [compatibility table](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements) for TensorRT and ONNX Runtime versions.
