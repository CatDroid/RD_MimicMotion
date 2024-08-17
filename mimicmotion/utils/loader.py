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
        
        huggingface_token = "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"

        # 在 svd 模型路径下 有多个子集目录
       
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet", token=huggingface_token ))

        
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=torch.float16, variant="fp16", token=huggingface_token)
        # 图片编码器 
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder", torch_dtype=torch.float16, variant="fp16", token=huggingface_token)
        # 噪声调度器  Euler  不是nn.Module ?? 也会被 load_state_dict影响 
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler", token=huggingface_token)
        # 特征提取器 
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor", token=huggingface_token)

        # unet输出到  PoseNet ??

        # pose_net
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
    mimicmotion_models.load_state_dict(torch.load(infer_config.ckpt_path, map_location="cpu"), strict=False)

  
    pipeline = MimicMotionPipeline(
        vae=mimicmotion_models.vae,  # vae 
        image_encoder=mimicmotion_models.image_encoder,  # 图片编码器
        unet=mimicmotion_models.unet,  # unet
        scheduler=mimicmotion_models.noise_scheduler, # 调度器 
        feature_extractor=mimicmotion_models.feature_extractor, 
        pose_net=mimicmotion_models.pose_net # 这是工作流的最后一部分 ??
    )
    return pipeline

