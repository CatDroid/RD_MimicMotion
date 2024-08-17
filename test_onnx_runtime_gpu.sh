
# 由于系统没有安装 cudnn 
# 所以使用torch安装时候带有de cudnn
# onnx runtime 使用gpu推理时候需要cudnn
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=~/anaconda3/envs/mimicmotion/lib/python3.11/site-packages/torch/lib/
python test_onnx_runtime_gpu.py

#LD_LIBRARY_PATH=~/anaconda3/envs/mimicmotion/lib/python3.11/site-packages/torch/lib/ python test_onnx_runtime_gpu.py