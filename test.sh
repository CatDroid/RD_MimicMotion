PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python inference.py --inference_config configs/test.yaml

# 

# For the 35s demo video, the 72-frame model requires 16GB VRAM (4060ti)   # 72帧模型需要16G显存  
#                                and finishes in 20 minutes on a 4090 GPU. # 20分钟完成35s的Demo 在4090显卡上

# The minimum VRAM requirement for the 16-frame U-Net model is 8GB;                 # 16帧的U-Net模型是8GB
#                               however, the VAE decoder demands 16GB.              # VAE解码器是16GB
#                               You have the option to run the VAE decoder on CPU.

 