import os
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

from einops import repeat
import folder_paths
import comfy.model_management as mm
import comfy.utils

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    is_accelerate_available = True
except:
    pass

from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline

from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mimicmotion.modules.unet import UNetSpatioTemporalConditionModel
from mimicmotion.modules.pose_net import PoseNet
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline

class MimicMotionModel(torch.nn.Module):
    def __init__(self, base_model_path):
        """construnct base model components and load pretrained svd model except pose-net
        Args:
            base_model_path (str): pretrained svd model path
        """
        super().__init__()
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet", variant="fp16"))
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae", variant="fp16")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder", variant="fp16")
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        # pose_net
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])

class DownloadAndLoadMimicMotionModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [   'MimicMotion-fp16.safetensors',
                    ],
                    ),
            "precision": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                    ], {
                        "default": 'fp16'
                    }),
            },
        }

    RETURN_TYPES = ("MIMICPIPE",)
    RETURN_NAMES = ("mimic_pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "MimicMotionWrapper"

    def loadmodel(self, precision, model):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        
        download_path = os.path.join(folder_paths.models_dir, "mimicmotion")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/MimicMotion_pruned", 
                                allow_patterns=[f"*{model}*"],
                                local_dir=download_path, 
                                local_dir_use_symlinks=False)

        ckpt_base_name = os.path.basename(model_path)
        print(f"Loading model from: {model_path}")

        svd_path = os.path.join(folder_paths.models_dir, "diffusers", "stable-video-diffusion-img2vid-xt-1-1")

        if not os.path.exists(svd_path):
            raise ValueError(f"Please download stable-video-diffusion-img2vid-xt-1-1 to {svd_path}")

        mimicmotion_models = MimicMotionModel(svd_path).to(device=device).eval()
        mimicmotion_models.load_state_dict(comfy.utils.load_torch_file(model_path), strict=False)

        pipeline = MimicMotionPipeline(
            vae=mimicmotion_models.vae, 
            image_encoder=mimicmotion_models.image_encoder, 
            unet=mimicmotion_models.unet, 
            scheduler=mimicmotion_models.noise_scheduler,
            feature_extractor=mimicmotion_models.feature_extractor, 
            pose_net=mimicmotion_models.pose_net,
        )
        pipeline.unet.to(dtype)
        pipeline.pose_net.to(dtype)
        pipeline.vae.to(dtype)
        pipeline.image_encoder.to(dtype)
        pipeline.pose_net.to(dtype)        

        mimic_model = {
            'pipeline': pipeline,
            'dtype': dtype
        }
        return (mimic_model,)
    
class MimicMotionSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mimic_pipeline": ("MIMICPIPE",),
            "ref_image": ("IMAGE",),
            "pose_images": ("IMAGE",),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "cfg_min": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "cfg_max": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "fps": ("INT", {"default": 15, "min": 2, "max": 100, "step": 1}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),            
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "MimicMotionWrapper"

    def process(self, mimic_pipeline, ref_image, pose_images, cfg_min, cfg_max, steps, seed, noise_aug_strength, fps, keep_model_loaded):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mimic_pipeline['dtype']
        pipeline = mimic_pipeline['pipeline']

        B, H, W, C = pose_images.shape
        ref_image = ref_image.permute(0, 3, 1, 2).to(device).to(dtype)
        pose_images = pose_images.permute(0, 3, 1, 2).to(device).to(dtype)
        ref_image = ref_image * 2 - 1
        pose_images = pose_images * 2 - 1

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        frames = pipeline(
            ref_image, 
            image_pose=pose_images, 
            num_frames=B,
            tile_size = 16, 
            tile_overlap= 6,
            height=H,
            width=W, 
            fps=fps,
            noise_aug_strength=noise_aug_strength, 
            num_inference_steps=steps,
            generator=generator,
            min_guidance_scale=cfg_min, 
            max_guidance_scale=cfg_max, 
            decode_chunk_size=8, 
            output_type="pt", 
            device=device
        ).frames
        frames = frames.squeeze(0).permute(0, 2, 3, 1).cpu().float()
        print(frames.shape)

        return frames,
    

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadMimicMotionModel": DownloadAndLoadMimicMotionModel,
    "MimicMotionSampler": MimicMotionSampler,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadMimicMotionModel": "DownloadAndLoadMimicMotionModel",
    "MimicMotionSampler": "MimicMotionSampler",
}
