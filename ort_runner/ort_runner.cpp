// ONNX Runtime runner - example that launches given ONNX model using ONNX Runtime.

#include <cstdlib>
#include <iostream>
#include <string>

#include <onnxruntime_cxx_api.h>

namespace
{
std::string get_type_name(ONNXTensorElementDataType t)
{
    switch (t)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        return "undefined";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "double";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        return "complex float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        return "complex double";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        return "float8e4m3fn";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
        return "float8e4m3fnuz";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
        return "float8e5m2";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        return "float8e5m2fnuz";
    default:
        return "unknown";
    }
}

size_t get_type_size(ONNXTensorElementDataType t)
{
    switch (t)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        throw std::invalid_argument("unexpected type size");
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        throw std::runtime_error("uninplemented string type support");
        ;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        return 16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        return 1;
    default:
        throw std::invalid_argument("unknown type size");
    }
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <onnx_file> [dynamic input dimensions, if any, in order]" << std::endl;
        return EXIT_FAILURE;
    }
    char *onnf_fname = argv[1];
    std::vector<int64_t> dyn_dimensions;
    for (int i = 2; i < argc; i++)
    {
        try
        {
            dyn_dimensions.push_back(std::stoi(argv[i]));
        }
        catch (std::invalid_argument &e)
        {
            std::cerr << "Invalid dynamic input dimension " << argv[i] << ", integer expected." << std::endl;
            return EXIT_FAILURE;
        }
    }

    // ONNX Runtime environment.
    Ort::Env env;

    std::cout << "Available execution providers:" << std::endl;
    std::vector<std::string> providers = Ort::GetAvailableProviders();
    bool has_cuda = false;
    bool has_tensorrt = false;
    for (size_t i = 0; i < providers.size(); i++)
    {
        std::cout << "  " << i << ". " << providers[i] << std::endl;
        if ("CUDAExecutionProvider" == providers[i])
        {
            has_cuda = true;
        }
        if ("TensorrtExecutionProvider" == providers[i])
        {
            has_tensorrt = true;
        }
    }

    Ort::SessionOptions session_options;
    if (has_tensorrt)
    {
        OrtTensorRTProviderOptions trt_options = {};
        trt_options.device_id = 0;
        // TensorRT builds own models from ONNX, caching them increases the performance of first session.Run().
        trt_options.trt_engine_cache_enable = 1;
        trt_options.trt_engine_cache_path = "./";
        session_options.AppendExecutionProvider_TensorRT(trt_options);
    }
    else if (has_cuda)
    {
        OrtCUDAProviderOptions cuda_options = {};
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Uncomment to enable built-in ONNX Runtime profiler information dump to json file.
    // Json can be inspected using about:tracing tab in chrome based browsers.
    // session_options.EnableProfiling(argv[0]);

    Ort::Session session{env, onnf_fname, session_options};

    Ort::AllocatorWithDefaultOptions allocator;
    // Get names, types and shapes of inputs and outputs.
    size_t inputs_count = session.GetInputCount();
    size_t outputs_count = session.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> input_names;
    std::vector<Ort::AllocatedStringPtr> output_names;
    std::cout << "Model inputs:" << std::endl;
    size_t next_dyn_dim = 0;
    std::vector<std::vector<int64_t>> input_shapes;
    for (size_t i = 0; i < inputs_count; i++)
    {
        input_names.push_back(session.GetInputNameAllocated(i, allocator));
        std::cout << "  " << i << ". " << input_names.back().get() << std::endl;
        std::cout << "    type: "
                  << get_type_name(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType())
                  << std::endl;
        input_shapes.push_back(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        std::cout << "    shape: [ ";
        for (int64_t &x : input_shapes.back())
        {
            if (-1 == x)
            {
                if (next_dyn_dim >= dyn_dimensions.size())
                {
                    std::cerr << "Please provide more values for dynamic input dimensions of input "
                              << input_names.back().get() << std::endl;
                    return EXIT_FAILURE;
                }
                x = dyn_dimensions[next_dyn_dim++];
                std::cout << " (" << x << ") ";
            }
            else
            {
                std::cout << x << " ";
            }
        }
        std::cout << "]" << std::endl;
    }
    if (next_dyn_dim != dyn_dimensions.size())
    {
        std::cerr << "Too many values for dynamic input dimensions of input provided." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Model outputs:" << std::endl;
    for (size_t i = 0; i < outputs_count; i++)
    {
        output_names.push_back(session.GetOutputNameAllocated(i, allocator));
        std::cout << "  " << i << ". " << output_names.back().get() << std::endl;
        std::cout << "    type: "
                  << get_type_name(session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType())
                  << std::endl;
        std::vector<int64_t> shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "    shape: [ ";
        for (auto x : shape)
        {
            std::cout << x << " ";
        }
        std::cout << "]" << std::endl;
    }

    // Where to allocate the tensors.
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Allocate input tensors.
    std::vector<Ort::Value> input_tensors;
    std::vector<std::vector<char>> input_data;
    for (size_t i = 0; i < inputs_count; i++)
    {
        ONNXTensorElementDataType t = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        size_t input_size = get_type_size(t);
        for (auto x : input_shapes[i])
        {
            input_size *= x;
        }
        input_data.emplace_back(input_size);
        // Creates tensor with supplied buffer, need to keep input_data in memory while it is alive.
        input_tensors.push_back(Ort::Value::CreateTensor(memory_info, input_data.back().data(),
                                                         input_data.back().size(), input_shapes[i].data(),
                                                         input_shapes[i].size(), t));
    }

    std::vector<char *> input_names_c;
    for (auto &name : input_names)
    {
        input_names_c.push_back(name.get());
    }

    std::vector<char *> output_names_c;
    for (auto &name : output_names)
    {
        output_names_c.push_back(name.get());
    }

    // Run the inference.
    std::cout << "Running inference ... " << std::flush;
    std::vector<Ort::Value> output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_names_c.data(), input_tensors.data(), input_tensors.size(),
                    output_names_c.data(), output_names_c.size());

    std::cout << "done." << std::endl;

    // Dynamic output shapes are calculated by ONNX execution provider.
    for (size_t i = 0; i < output_tensors.size(); i++)
    {
        std::cout << "  " << i << ". " << output_names[i].get() << std::endl;
        std::cout << "    type: " << get_type_name(output_tensors[i].GetTensorTypeAndShapeInfo().GetElementType())
                  << std::endl;
        std::vector<int64_t> shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "    shape: [ ";
        for (auto x : shape)
        {
            std::cout << x << " ";
        }
        std::cout << "]" << std::endl;
    }
}
