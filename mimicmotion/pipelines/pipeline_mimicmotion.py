import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import PIL.Image
import einops
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion \
    import _resize_with_antialiasing, _append_dims
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..modules.pose_net import PoseNet

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")

    return outputs


@dataclass
class MimicMotionPipelineOutput(BaseOutput):
    r"""
    Output class for mimicmotion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class MimicMotionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K]
            (https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
        pose_net ([`PoseNet`]):
            A `` to inject pose signals into unet.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
        pose_net: PoseNet,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
        self, 
        image: PipelineImageInput, 
        device: Union[str, torch.device], 
        num_videos_per_prompt: int, 
        do_classifier_free_guidance: bool):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # 这类的处理 跟后面的vae image类似 做了一个 全0的负向image embeding 
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # 对于无分类器引导（classifier free guidance），我们需要进行两次前向传播。
            # 这里我们将无条件的嵌入和"文本嵌入" 连接成一个批次，以避免进行两次前向传播。 -- 图+视频 ==> 视频 ?? 为什么需要 文本嵌入 ??

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

            # !!! 第一个是 negative_image_embeddings 第二个 image_embeddings 才是参考图的embeding 

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device, dtype=self.vae.dtype)
        image_latents = self.vae.encode(image).latent_dist.mode()

        # CFG -- classifier_free_guidance
        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)
            # 生成一个与 image_latents 形状和数据类型相同的张量，但所有元素都为零。这被称为“负样本 潜在向量”（negative_image_latents）
            # 在 CFG 中，negative_image_latents 代表无条件的（或中性）生成，通常与某种随机或零向量相关联。

            # For classifier free guidance, we need to do two forward passes. 两个向前推理pass?
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes 为了避免执行两次? 所以合并了 无条件和文本的embeding ??
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, " \
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. " \
                f"Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(
        self, 
        latents: torch.Tensor, 
        num_frames: int, 
        decode_chunk_size: int = 8):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        # i的 步进是 decode_chunk_size  遍历整个 latents
        # 窗口滑动  不重叠 
        print(f"latents.shape[0] = {latents.shape[0]} decode_chunk_size = {decode_chunk_size} accepts_num_frames = {accepts_num_frames}")
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                # 我们只在需要时才传递 num_frames_in
                decode_kwargs["num_frames"] = num_frames_in

            # 使用VAE解码 
            # decode_chunk_size 只是影响一次 送 self.vae.decode的数量  并不影响效果??
            frame = self.vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame.cpu())
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # call by  run_pipeline @ inference.py 

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor], # 可以是 列表 ?? list len =1 
        image_pose: Union[torch.FloatTensor], # pose是 pose视频帧数/sample_stride + 1个参考图像
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        tile_size: Optional[int] = 16,
        tile_overlap: Optional[int] = 4,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        # inference.py 配置这个 fps 也是7 ? 注释写的是 '生成后,将"生成的图像"导出到视频的速率'  ?inference.py最后导出视频是15fps ? 
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        image_only_indicator: bool = False,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        device: Union[str, torch.device] =None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/
                feature_extractor/preprocessor_config.json).

                用于指导图像生成的图像

            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.

                生成图像的高度  默认是 self.unet.config.sample_size * self.vae_scale_factor


            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` 
                and to 25 for `stable-video-diffusion-img2vid-xt`
                生成视频帧的数量 
                14 用 stable-video-diffusion-img2vid    模型的话 
                25 用 stable-video-diffusion-img2vid-xt 模型的话 

            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.

                去噪步骤的数量。去噪步骤越多，通常图像质量越高，但推理速度越慢。此参数受 `strength` 调节

            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.

                最小/最大 指导尺度 guidance scale。用于分类器 对 第一帧 / 最后一帧的自由指导 free guidance 。

            fps (`int`, *optional*, defaults to 7):
                Frames per second.The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.

                生成后 将生成的图像 导出到视频的速率 。
                请注意 在训练期间 Stable Diffusion Video 的 UNet 是在 fps-1 上进行 "微调" 的   ??? micro-conditioned ???

            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. 
                The higher the number the more motion will be in the video.

                运动桶 ID  默认 127
                用作生成generation的调节conditioning ???
                数字越高，视频中的运动就越多。

            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, 
                the higher it is the less the video will look like the init image. Increase it for more motion.

                添加到初始图像的噪声量，
                噪声量越高，视频看起来就“越不像初始图像”。  ---- test.yaml 中 noise_aug_strength = 0.0
                增加噪声量可获得“更多运动” ???

            image_only_indicator (`bool`, *optional*, defaults to False):
                Whether to treat the inputs as batch of images instead of videos.
                是否将输入视为"图像批次"而不是"视频"  ??? 

            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time.The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. 
                By default, the decoder will decode all frames at once for maximal quality. 
                Reduce `decode_chunk_size` to reduce memory usage.

                一次解码的帧数。块大小越大，"帧之间"的"时间一致性"越高，但内存消耗也越高。 
                --->>> 为什么这么说?? decode_latents  只是一次送self.vae.decode的数量 ???  

                默认情况下，解码器将一次性解码"所有帧"以获得"最佳质量"。

                减少`decode_chunk_size`以减少内存使用量。

            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.

                每个提示(prompt)生成的图像数量。 默认是1  

            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.

            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.

                可选
                从高斯分布中采样的"预生成" 噪声潜变量(noisy latents)，用作图像生成的输入。
                可用于调整tweak"具有不同提示 (different prompts)"的相同生成(the same generation)。
                如果未提供，则通过使用提供的随机`generator`进行采样来生成潜变量张量。


            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.

            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.

                在推理期间，在每个去噪步骤结束时调用的函数

            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

            device:
                On which device the pipeline runs on.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, 
                [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image(
        "https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        print(f"self.unet.config = {self.unet.config}, vae_scale_factor = {self.vae_scale_factor}")
        print(f"width = {width}  height = {height} , decode_chunk_size = {decode_chunk_size}, num_frames = {num_frames}")

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames

        # 如果不提供 decode_chunk_size 那么直接用 num_frames 整个pose视频的长度
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            # 传入是一个Image图对象 
            batch_size = 1
        elif isinstance(image, list):
            # 传入是一个列表 [ [C,H,W], [C,H,W],..]
            batch_size = len(image)
            print(f"image is list, batch_size = {batch_size}")
        else:
            # 使用 [B,C,H,W]的形式 
            batch_size = image.shape[0]
        device = device if device is not None else self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        self.image_encoder.to(device)
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        self.image_encoder.cpu()

        # ????
        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width).to(device) # 预处理 参考图 裁剪等 为了给 self.vae
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype) #  noise_aug_strength: 0 不加噪声到参考图 
        image = image + noise_aug_strength * noise

        # print(f"image = {image.shape}") # image = torch.Size([1, 3, 1024, 576]) 就是参考图的尺寸 

        self.vae.to(device)
        image_latents = self._encode_vae_image( # 也用vae编码 参考图
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
             # num_videos_per_prompt 默认是1 
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)
        self.vae.cpu()

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        #print(f"image_latents 1 = {image_latents.shape}") # [2, 4, 128, 72]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        # unsqueeze(1) 在axix=1前增加一个维度(作为帧数)
        #print(f"image_latents 2 = {image_latents.shape}") # torch.Size([2, 531=姿态视频帧数/sample_stride+1, 4, 128, 72])

        # fps=6,  motion_bucket_id=127, noise_aug_strength = 0, batch_size = 1, num_videos_per_prompt = 1. self.do_classifier_free_guidance = True
        print(f"_get_add_time_ids fps={fps},"\
                f"motion_bucket_id={motion_bucket_id},"\
                f"noise_aug_strength = {noise_aug_strength},"\
                f"batch_size = {batch_size},"\
                f"num_videos_per_prompt = {num_videos_per_prompt},"\
                f"self.do_classifier_free_guidance = {self.do_classifier_free_guidance}")

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps

        # diffuser retrieve_timesteps 
        # 用于获取扩散过程中的 "时间步长"（timesteps）  不是 时间戳 timestamp !
        # 在扩散模型中，"时间步长"是指模型在生成图像或其他数据时所经过的步骤。
        # 每个时间步长对应于一个"噪声水平"，模型通过"反向扩散"过程逐步"去噪"，从而生成目标数据。
        # 生成时间步长序列: 根据特定的调度策略（如线性、指数等）生成一个时间步长序列，定义从初始噪声到最终输出的去噪过程
        # 控制生成过程:    通过调整时间步长的"数量和分布"，可以控制生成过程的"速度和质量"。时间步长的"精细程度"直接影响生成的"细节和多样性"。

        # self.scheduler 在 loader.py 是 EulerDiscreteScheduler "use_karras_sigmas": true
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)
        print(f"self.scheduler = {self.scheduler}")
        print(f"num_inference_steps = {num_inference_steps}, timesteps = {timesteps}")
        # num_inference_steps 就是 test.yaml配置的 25 步  timesteps 长度也是25 (不是等间隔的??) => 使用Karras分布 不是线性的

        print(f"prepare_latents before latents = {latents} generator = {generator}")

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            tile_size,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        print(f"prepare_latents latents = {latents.shape} ")                # torch.Size([1, 16, 4, 96, 56]) 
        latents = latents.repeat(1, num_frames // tile_size + 1, 1, 1, 1)   # num_frames // tile_size + 1 = 34  # 531//16+1 = 33+1=34 # num_frames这个不是test.yaml配置的num_frames
        print(f"repeat {num_frames // tile_size + 1} latents = {latents.shape} ") #  torch.Size([1, 544, 4, 96, 56])  #  544 = 16 * 34  # num_frames是 pose视频帧数/sample_stride + 1参考图
        latents = latents[:, :num_frames]
        # 最多取 num_frames  这个也跟视频的长度 有关系   ??? 4x96x56 跟什么有关系??
        print(f"crop to {num_frames} latents = {latents.shape}") # torch.Size([1, 531, 4, 96, 56])

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        # batch_size = 1 num_videos_per_prompt = 1
        print(f"batch_size = {batch_size} num_videos_per_prompt = {num_videos_per_prompt}")

        # 7. Prepare guidance scale   
        # 从 min_guidance_scale 到 max_guidance_scale 的 "等间距" 张量，长度为 num_frames。这通常是为了在多个帧之间逐渐调整引导参数。
        #  (num_frames,) 变为 (1, num_frames)
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        print(f"guidance_scale 1 = {guidance_scale.shape}") # torch.Size([1, 531])  531 = pose视频帧数/sample_stride + 一个参考图像 

        # 为了确保数据类型和设备一致
        guidance_scale = guidance_scale.to(device, latents.dtype)
        # 通过重复操作将 guidance_scale 的第一个维度扩展，使其适应批量处理。
        # batch_size * num_videos_per_prompt 决定了批次的大小，
        # 因此扩展后的 guidance_scale 形状将是 (batch_size * num_videos_per_prompt, num_frames)，
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        print(f"guidance_scale 2 = {guidance_scale.shape}") # torch.Size([1, 531]) 因为 batch_size =1 num_videos_per_prompt = 1
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        print(f"guidance_scale 3 = {guidance_scale.shape}") # torch.Size([1, 531, 1, 1, 1]) ?? 多了后面的 1 1 1 ??

        self._guidance_scale = guidance_scale

        print(f"num_frames = {num_frames} tile_size = {tile_size} tile_overlap = {tile_overlap}")
        # num_frames = 531 ( 姿态视频帧数/stride ） tile_size = 配置中的 test_case.num_frames  tile_overlap = 配置中的 test_case.frames_overlap

        # 8. Denoising loop
        self._num_timesteps = len(timesteps)
        indices = [[0, *range(i + 1, min(i + tile_size, num_frames))] for i in # i 从0开始 到 最后 num_frames - tile_size + 1   
                   range(0, num_frames - tile_size + 1, tile_size - tile_overlap)] # 步进是 tile_size - tile_overlap(tile_size中有tile_overlap是重叠)
        
        # [0, *range(i + 1, min(i + tile_size, num_frames))]  0是一定包含 也就是 tile_size中有一个是0 tile_overlap中也有一个是0

        # e.g 
        # num_frames: 4  frames_overlap: 2  sample_stride: 2
        # indices = [   
        # [0, 1, 2, 3]   
        # [0, 3, 4, 5]
        # [0, 5, 6, 7] 
        # [0, 7, 8, 9]
        # [0, 9, 10, 11]
        # .. ]
        
        if indices[-1][-1] < num_frames - 1:
            indices.append([0, *range(num_frames - tile_size + 1, num_frames)])

        print(f"len(indices) = {len(indices)}")
        print(f"len(timesteps) = {len(timesteps)}") # 25 ??

        print(f"image_embeddings = {image_embeddings.shape}")   # torch.Size([2, 1, 1024]) 2是因为 多了一个 negative_image_embeddings 
        print(f"added_time_ids   = {added_time_ids.shape}")     # torch.Size([2, 3])

        self.pose_net.to(device)
        self.unet.to(device)

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        with self.progress_bar(total=len(timesteps) * len(indices)) as progress_bar:
            for i, t in enumerate(timesteps):
                # 遍历所有的时间步, 每个时间步中会遍历 整个视频 的 所有带部分重叠的 片段 (也就是会遍历整个视频所有帧)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                #  这是图生图? 参考图的vae latent也作为输入  每个timpsteap都会加入一样的？
                # Concatenate image_latents over channels dimension 
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                # 预测噪声残差 ?? 
                # 跟 image_latents 的尺寸是一样的 ?? (参考图的vae latent)
                noise_pred = torch.zeros_like(image_latents)
                noise_pred_cnt = image_latents.new_zeros((num_frames,))
                #print(f"noise_pred = {noise_pred.shape}") # [2, 531, 4, 128, 72]) 2=无添加+有条件  531=pose视频帧数/sample_stride+1 

                # 分块处理中的加权平均或平滑操作。生成的权重在块的中心位置具有较高的值，而在边缘位置权重较低。
                # 这种对称权重可以用于平滑过渡或减少边界效应。
                weight = (torch.arange(tile_size, device=device) + 0.5) * 2. / tile_size
                weight = torch.minimum(weight, 2 - weight)

                #print(f"weight = {weight}") # weight = tensor([0.2500, 0.7500, 0.7500, 0.2500], device='cuda:0')

                # 一个 timestep (迭代 降噪 comfyui面板上steps) 要遍历 整个indices(有重叠) 
                for idx in indices:

                    # print(f"idx = {type(idx)} {len(idx)}") # <class 'list'> 4==tile_size

                    # classification-free inference 
                    # Classification-Free Inference 是更广义的术语，可以指任何"不依赖分类器"的"推理过程"， 
                    # Classifier-Free Guidance 可以看作是 Classification-Free Inference 的一种特定应用形式，主要用于扩散模型中，以实现"更好的条件控制"


                    # image_pose[idx]是取tile_size个pose视频帧
                    # self.pose_net 相当于controlnet的作用 ?   是不是可以一次预生成好全部, 后面就直接取就好? (内存占用??)
                    # pose_net.py forward没有看到 self.pose_net会记录之前的状态 
                    pose_latents = self.pose_net(image_pose[idx].to(device))
                    # print(f"pose_latents = {pose_latents.shape}")  # torch.Size([tile_size, 320 ?, 128 ?, 72 ?])


                    # 扩散模型（diffusion model）中，每一个时间步（time step）通常都要运行一次 U-Net 模型
                    #   噪声添加过程（Forward Process）：扩散模型的训练过程中，将逐步增加噪声到图像上，模拟从原始图像到纯噪声的过程。这个过程不涉及 U-Net。
                    #   噪声去除过程（Reverse Process）：在推理阶段，扩散模型从纯噪声开始，通过"逐步去除"噪声来生成图像。
                    #                                   每一个"时间步"，都会"预测"当前图像中的噪声分布，并根据该预测”更新图像"
                    # U-Net的作用
                    #   在每个时间步，U-Net 接收 "当前的噪声图像" 和 "时间步的编码"（可能还有额外的 "条件信息"，如文本嵌入）作为输入，并输出一个"噪声的估计值"。  
                    #   通过这个噪声"估计值"，模型可以"更新"当前的图像状态，从而在"多个时间步" 后从纯噪声生成清晰的图像 
                    #  
                    # 有条件和无条件生成：
                    #    在生成任务中，模型可能希望生成与特定条件（如文本描述）一致的图像。这被称为"有条件生成"

                    # Classifier-Free Guidance 
                    #    提供了一种"通过模型自身"实现"条件控制"的方式，而不依赖于"外部分类器"。
                    #
                    # 两次 U-Net 前向传播的目的
                    #       第一次前向传播（无条件）
                    #           模型接收带有噪声的图像和一个“空”或无条件的输入（通常通过将条件信息设为零或忽略）进行前向传播。
                    #           模型输出这个图像中估计的噪声。
                    #           这个过程提供了一个基础的噪声估计，它不依赖于任何条件信息
                    #       第二次前向传播（有条件）   
                    #           模型接收相同的带有噪声的图像，但这次附带有条件信息（如文本嵌入）。
                    #           模型输出估计的噪声，这次的估计是基于条件信息的。
                    #           这个过程提供了一个基于条件信息的噪声估计，用于生成与条件相关的图像。

                    # self.unet 就是 UNetSpatioTemporalConditionModel @ loader.py  # UNet 时空条件模型 Spatio Temporal Condition

                    _noise_pred = self.unet(
                        latent_model_input[:1, idx],# 噪声图也是 1参考图+pose视频/2
                        t,
                        # encoder_hidden_states 这个是参考图的embeding/condition向量 (第一个是全0的条件 相当于没有)
                        encoder_hidden_states=image_embeddings[:1], # !! 第一个是全0的image embeding 
                                                                    # 取反面的image embeding ? 也可能是None?  
                                                                    #  _encode_image 函数 判断 do_classifier_free_guidance 加入一个全0的negative_image_embeddings
                        added_time_ids=added_time_ids[:1],
                        pose_latents=None,
                        # 第一次 没有 pose_latents 
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]


                    # print(f"idx = {idx}") # idx 是会有重叠的 这里+=也会包含之前已经有的idx
                    # [0, 1, 2, 3]   num_frames: 4  frames_overlap: 2  sample_stride: 2
                    # [0, 3, 4, 5]
                    # [0, 5, 6, 7] 
                    # [0, 7, 8, 9]
                    # [0, 9, 10, 11]
                    # 加权的方式 加入到 这个时间步 所有视频帧 的 noise_pred 中
                    noise_pred[:1, idx] += _noise_pred * weight[:, None, None, None]
                    
                    # 上面latent_model_input image_embeddings和added_time_ids  都是取 [:1]  下面 第二次是取 [1:]  正面和方面的image embeding 
                    # 第一次 unet的区别 加入到 noise_pred[0]  :1
                    # 第二次 unet的区别 加入到 noise_pred[1]  1:

                    # normal inference
                    _noise_pred = self.unet(
                        latent_model_input[1:, idx],
                        t,
                        encoder_hidden_states=image_embeddings[1:],   # 编码器隐状态 ?? 取正面的image embeding ?                                         
                        added_time_ids=added_time_ids[1:],
                        pose_latents=pose_latents,
                        # 第二次unet的区别 就是加了 pose_latents 相当于加了条件 
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]
                    noise_pred[1:, idx] += _noise_pred * weight[:, None, None, None]

                    #  noise_pred [2, 531, 4, 128, 72]) 
                    #           531 = pose视频帧数/sample_stride + 参考图  #  这个大小 就是视频的长度相关了 
                    #           2 =  无条件的噪声预测  + 有条件的噪声预测

                    noise_pred_cnt[idx] += weight
                    progress_bar.update()
                    # 遍历完 indices 中所有的 tile (每个大小是 tile_size)

                noise_pred.div_(noise_pred_cnt[:, None, None, None])
                # 当前时间步, 遍历了完整个视频 
                # ?? 这个时间步 每个视频帧的 噪声预测  noise_pred ?
                
                # 直到生成
                # 混合噪声估计：通过对有条件和无条件的噪声估计进行线性组合（通常是通过一种权重的方式），模型可以更精确地控制生成图像的质量和"条件一致性"。
                #               用一个权重参数 α / guidance_scale 来调节有条件估计和无条件估计的比重：
                #               Final Output = Unconditional Output + α × (Conditional Output − Unconditional Output)
                #               这个权重参数 控制了条件信息对生成结果的影响程度
                #                   较高的值意味着生成的图像将更"严格地遵循条件信息"，
                #                   较低的值则会保留更多的随机性。
                # 
                # 通过对这两个估计进行组合, 模型可以在不依赖"外部分类器"的情况下，增强生成图像"与条件信息之间的相关性"
                # 
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # ?? 用 这个时间步 预测的 (所有帧) noise_pred 来更新 潜空间表述 的 带噪声的图像(去噪)
                # latents    [1, 531, 4,  96, 56]
                # noise_pred [2, 531, 4, 128, 72]  最后俩的分辨率不同 ??
                #  
                # compute the previous noisy sample x_t -> x_t-1
                # 从估计的噪声分布中'采样' 去噪 x_t 得到 x_t - 1 
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

        # 从gpu转回cpu??
        self.pose_net.cpu()
        self.unet.cpu()

        # vae 解码 latent
        if not output_type == "latent":
            print(f"use vae decoder ! latents = {latents.shape} num_frames = {num_frames} decode_chunk_size = {decode_chunk_size}")
            self.vae.decoder.to(device)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return MimicMotionPipelineOutput(frames=frames)
