#!/bin/env python3

# Make simple ONNX model that upscales input image with cubic algorithm.

import onnx

# FullHD input.
IN_W = 1920
IN_H = 1080

# 5K output.
OUT_W = 5120
OUT_H = 2880

INPUT_NAME = "input"
OUTPUT_NAME = "output"
ONNX_FILE_NAME = "upscale.onnx"

input = onnx.helper.make_tensor_value_info(
    INPUT_NAME, onnx.TensorProto.FLOAT, [3, IN_H, IN_W])

output = onnx.helper.make_tensor_value_info(
    OUTPUT_NAME, onnx.TensorProto.FLOAT, [3, OUT_H, OUT_W])

sizes = onnx.helper.make_tensor(
    name="sizes",
    data_type=onnx.TensorProto.INT64,
    dims=(3,),
    vals=(3, OUT_H, OUT_W))

resize_node = onnx.helper.make_node(
    name="Cubic Upscaler",
    op_type="Resize",
    inputs=[INPUT_NAME, "", "", "sizes"],
    outputs=[OUTPUT_NAME],
    mode="cubic")

graph_def = onnx.helper.make_graph(
    nodes=[resize_node],
    name="Upscaler",
    inputs=[input],
    outputs=[output],
    initializer=[sizes])

model_def = onnx.helper.make_model(
    graph_def, producer_name="onnx-misuse")
model_def = onnx.shape_inference.infer_shapes(model_def)
onnx.save(model_def, ONNX_FILE_NAME)

# Check model
model_def = onnx.load(ONNX_FILE_NAME, load_external_data=False)
model_def = onnx.shape_inference.infer_shapes(model_def)
onnx.checker.check_model(model_def)
onnx.save_model(model_def, ONNX_FILE_NAME)
