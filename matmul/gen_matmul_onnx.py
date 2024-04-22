#!/usr/bin/env python3
import onnx
from onnx.tools import update_model_dims

M = 1024
N = 1024
K = 1024
ONNX_FILE_NAME = "matmul.onnx"

print("Generating ONNX file {}.".format(ONNX_FILE_NAME))
print("onnx.__version__={!r}, opset={}, IR_VERSION={}".format(
    onnx.__version__, onnx.defs.onnx_opset_version(), onnx.IR_VERSION))

a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [M, K])
b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [K, N])
c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [M, N])

matmul = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["c"])

# Create the graph (GraphProto)
graph_def = onnx.helper.make_graph(
    nodes=[matmul],
    name="Matrix multiplication",
    inputs=[a, b],  # Graph input
    outputs=[c],  # Graph output
    initializer=[]
)

# Create the model (ModelProto)
model_def = onnx.helper.make_model(
    graph_def, producer_name="onnx-misuse")
model_def.opset_import[0].version = 13

model_def = update_model_dims.update_inputs_outputs_dims(
    model_def, {"a": ["M", "K"], "b": ["K", "N"]}, {"c": ["M", "N"]})

onnx.save_model(model_def, ONNX_FILE_NAME)

# Check model
model_def = onnx.load(ONNX_FILE_NAME, load_external_data=False)
model_def = onnx.shape_inference.infer_shapes(model_def)
onnx.checker.check_model(model_def)
onnx.save_model(model_def, ONNX_FILE_NAME)

print("Done!")
