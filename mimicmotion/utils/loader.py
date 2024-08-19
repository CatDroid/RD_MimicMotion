import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler

# https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers
# __init__.py
# from .scheduling_euler_discrete import EulerDiscreteScheduler
# .scheduling_euler_discrete 这个就是euler的调度器 

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/image_processing_clip.py
# CLIP Contrastive Language-Image Pre-training 对比语言-图像预训练 -- OpenAI提出的一种多模态模型预训练方法

from ..modules.unet import UNetSpatioTemporalConditionModel
from ..modules.pose_net import PoseNet
from ..pipelines.pipeline_mimicmotion import MimicMotionPipeline

logger = logging.getLogger(__name__)

class MimicMotionModel(torch.nn.Module):
    def __init__(self, base_model_path):
        """construnct base model components and load pretrained svd model except pose-net
        Args:
            base_model_path (str): pretrained svd model path
        """
        super().__init__()
        

        # huggingface 默认下载路径:
        # /home/xxx/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1

        #huggingface_token = "hf_0000000000000000000000"
        huggingface_token = None

        # VAE 和 Clip 的区别 
        # 
        # VAE编码图像 : VAE主要用于图像的生成、重建、以及压缩。它将图像编码到一个低维的潜在空间（latent space），然后从这个潜在空间中解码出图像
        #               生成新图像 去噪、填补缺失图像部分 
        #               VAE由编码器、解码器和采样过程组成
        #               编码: 编码器将输入图像转换为"潜在空间"中的"概率分布参数（均值和方差），
        #               采样：然后从这个分布中"采样"生成一个"潜在向量"。
        #               解码：解码器则从潜在向量生成重构图像
        # Clip编码图像: 通过同时学习"图像和文本"的表示来实现多模态的对齐。它能够将"图像和文本" "嵌入" 到同一个"向量空间"中，以便可以进行跨模态的任务
        #               CLIP由一个图像编码器和一个文本编码器组成

        # 在 svd 模型路径下 有多个子集目录


        # UNetSpatioTemporalConditionModel  nn.Module
        # ---------- UNet 时空条件模型 Spatio Temporal Condition
        #            结合了空间和时间信息，用于捕捉视频数据中不仅存在于单个帧内的特征，还能够建模帧与帧之间的时间依赖性
        #            空间建模（Spatial Modeling）提取多尺度的空间特征，并在解码过程中重建高分辨率的输出。
        #            时间建模（Temporal Modeling）在 U-Net 的架构中引入时间维度  捕捉视频帧之间的时间依赖性   
        # ---------- MimicMotion_1-1.pth 会修改这个参数
        # ---------- 根据huggingface/diffusers重写了? 所以还用可以用他原来的权重文件 
        #            原来的: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_spatio_temporal_condition.py
        #            修改的: mimicmotion/modules/unet.py
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet", token=huggingface_token ))

        # AutoencoderKLTemporalDecoder nn.Module
        # ---------- MimicMotion_1-1.pth 不会 覆盖参数
        # ---------- 处理视频或时序图像数据  
        #            基于 KL (Kullback-Leibler) 散度的变分自编码器VAE架构
        #            "Temporal" 确实指的是时域或时间维度的连续性。这个术语强调了该模型处理 "时间序列" 数据的能力
        #            能够处理多帧图像序列，保持帧间的时间连贯性
        #            在压缩的潜在空间中进行操作，而不是直接在像素空间中操作
        #            将其转换回像素空间 
        #            滑动窗口 ; 一次处理一小段连续的帧序列，而不是整个视频  ;  对于长视频，可以逐步移动这个窗口来处理整个视频; 
        #            虽然每次只处理一小段序列，但模型仍能捕捉到局部的时间上下文。
        #            模型设计为处理固定长度的帧序列，比如8帧、16帧 
        # ---------- decode_latents @ pipeline_mimicmotion.py 滑动窗口并不重合 
        # ---------- 配置文件  ~/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1/ \
        #                     snapshots/043843887ccd51926e3efed36270444a838e7861/vae/config.json
        #            "in_channels": 3
        #            "out_channels": 3,
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=torch.float16, variant="fp16", token=huggingface_token)

        # 图片编码器  transformers.CLIPVisionModelWithProjection 是nn.Module 但是不会被 MimicMotion_1-1.pth  覆盖参数
        # ---------- 接受预处理后的图像输入，提取高级视觉特征 
        #            图像转换为"高维向量"表示，通常使用 Vision Transformer (ViT) 架构
        #            包含一个额外的投影层，将视觉特征映射到与文本特征相同的多模态空间
        #            通常输出两种向量：a) 最后一层的隐藏状态 last_hidden_state b) 经过投影层处理的向量 image_embeds （用于与文本特征进行对比）
        # ----------  image_encoder/config.json
        #               "num_attention_heads": 16
        #               "image_size": 224,
        #               "patch_size": 14,
        #               "hidden_act": "gelu"
        #               "projection_dim": 1024,
        # ----------  image_encoder/model.fp16.safetensors
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder", torch_dtype=torch.float16, variant="fp16", token=huggingface_token)


        # 噪声调度器  Euler  不是nn.Module 
        # ----------- ~/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1/\
        #               snapshots/043843887ccd51926e3efed36270444a838e7861/scheduler/scheduler_config.json
        #               "_class_name": "EulerDiscreteScheduler",
        #              beta参数  "beta_start": 0.00085,  "beta_schedule": "scaled_linear",   "beta_end": 0.012,
        #              插值类型   "interpolation_type": "linear",
        #              sigma参数  "sigma_max": 700.0, "sigma_min": 0.002,
        #              karras    "use_karras_sigmas": true
        # ----------  MimicMotion_1-1.pth 不会覆盖 这个参数
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler", token=huggingface_token)

        # 图像预处理-特征提取器  transformers.CLIPImageProcessor (不是nn.Module)
        # ----------- 将原始图像预处理成 CLIP 模型可以直接使用的格式 
        #               图像预处理: 调整图像大小  裁剪 归一化  转换为 PyTorch tensor 等等 
        # ----------- 配置文件在 ~/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1/snapshots/043843887ccd51926e3efed36270444a838e7861/feature_extractor/preprocessor_config.json
        #             feature_extractor/preprocessor_config.json 包含了 crop_size 224 224 image_mean image_std
        #                                                       "feature_extractor_type": "CLIPFeatureExtractor"
        #                                                       "image_processor_type": "CLIPImageProcessor"
        # ----------  MimicMotion_1-1.pth 不会覆盖 这个参数
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor", token=huggingface_token)  

        # unet输出到  PoseNet ??

        # pose_net 是 PoseNet(nn.Module 
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])
 

def create_pipeline(infer_config, device):
    """create mimicmotion pipeline and load pretrained weight

    Args:
        infer_config (str): 
        device (str or torch.device): "cpu" or "cuda:{device_id}"
    """

    # base_model_path = stabilityai/stable-video-diffusion-img2vid-xt-1-1
    print(f"create_pipeline infer_config.base_model_path = {infer_config.base_model_path}")

    # 创建 MimicMotionModel 对象 里面加载 sd目录下 的 vae unet noise_scheduler等模型 
    mimicmotion_models = MimicMotionModel(infer_config.base_model_path)

    # ckpt_path 这个是 models/MimicMotion_1-1.pth 的 模型   这个会否覆盖掉 ???

    mimicmotion_state_dict = torch.load(infer_config.ckpt_path, map_location="cpu")

    # MimicMotionModel 是nn.Module 
    # key_to_check = 'pose_net'  # checkpoint存放模型参数 不是用'递归的字典'实现的 
    key_to_check = 'pose_net.conv_layers.0.weight'
    if key_to_check in mimicmotion_state_dict : # mimicmotion_state_dict 只有模型参数 没有训练信息
        print(f"Key '{key_to_check}' exists in the state_dict.")
    else:
        print(f"Key '{key_to_check}' does not exist in the state_dict.")

    # 这里是 strict=False ?? 代表   MimicMotion_1-1.pth 跟 MimicMotionModel 模型定义会不一样 
    mimicmotion_models.load_state_dict(mimicmotion_state_dict, strict=False)

  
    pipeline = MimicMotionPipeline(
        vae=mimicmotion_models.vae,  # vae 
        image_encoder=mimicmotion_models.image_encoder,  # 图片编码器
        unet=mimicmotion_models.unet,  # unet
        scheduler=mimicmotion_models.noise_scheduler, # 调度器 
        feature_extractor=mimicmotion_models.feature_extractor, 
        pose_net=mimicmotion_models.pose_net # 这是工作流的最后一部分 ??
    )
    return pipeline

