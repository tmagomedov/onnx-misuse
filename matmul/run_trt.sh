#~/bin/sh
M=4096
N=4096
K=4096
/usr/src/tensorrt/bin/trtexec --onnx=matmul.onnx --shapes=a:${M}x${K},b:${K}x${N}
