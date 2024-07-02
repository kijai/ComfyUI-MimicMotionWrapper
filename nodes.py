import os
import torch
import numpy as np
import gc

import folder_paths
import comfy.model_management as mm
import comfy.utils

from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

script_directory = os.path.dirname(os.path.abspath(__file__))

from .mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from .mimicmotion.modules.unet import UNetSpatioTemporalConditionModel
from .mimicmotion.modules.pose_net import PoseNet

from .lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler

class MimicMotionModel(torch.nn.Module):
    def __init__(self, base_model_path, lcm=False):
        """construnct base model components and load pretrained svd model except pose-net
        Args:
            base_model_path (str): pretrained svd model path
        """
        super().__init__()
        unet_subfolder = "unet_lcm" if lcm else "unet"
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder=unet_subfolder, variant="fp16"))
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
            "lcm": ("BOOLEAN", {"default": False}),
            
            },
        }

    RETURN_TYPES = ("MIMICPIPE",)
    RETURN_NAMES = ("mimic_pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "MimicMotionWrapper"

    def loadmodel(self, precision, model, lcm):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        pbar = comfy.utils.ProgressBar(3)
        
        download_path = os.path.join(folder_paths.models_dir, "mimicmotion")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/MimicMotion_pruned", 
                                allow_patterns=[f"*{model}*"],
                                local_dir=download_path, 
                                local_dir_use_symlinks=False)

        print(f"Loading model from: {model_path}")
        pbar.update(1)

        svd_path = os.path.join(folder_paths.models_dir, "diffusers", "stable-video-diffusion-img2vid-xt-1-1")
        svd_lcm_path = os.path.join(folder_paths.models_dir, "diffusers", "stable-video-diffusion-img2vid-xt-1-1-lcm", "unet_lcm")
        
        if lcm and not os.path.exists(svd_lcm_path):
            print(f"Downloading AnimateLCM SVD model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/AnimateLCM-SVD-Comfy", 
                                allow_patterns=[f"*.json", "*diffusion_pytorch_model.fp16.safetensors*"],
                                local_dir=svd_path, 
                                local_dir_use_symlinks=False)
        else:
            if not os.path.exists(svd_path):
                print(f"Downloading SVD model to: {model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="vdo/stable-video-diffusion-img2vid-xt-1-1", 
                                    allow_patterns=[f"*.json", "*fp16*"],
                                    local_dir=svd_path, 
                                    local_dir_use_symlinks=False)
        pbar.update(1)

        mimicmotion_models = MimicMotionModel(svd_path, lcm=lcm).to(device=device).eval()
        mimicmotion_models.load_state_dict(comfy.utils.load_torch_file(model_path), strict=False)

        if lcm:
            lcm_noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
                num_train_timesteps=40,
                sigma_min=0.002,
                sigma_max=700.0,
                sigma_data=1.0,
                s_noise=1.0,
                rho=7,
                clip_denoised=False,
            )
            scheduler = lcm_noise_scheduler
        else:
            scheduler = mimicmotion_models.noise_scheduler

        pipeline = MimicMotionPipeline(
            vae = mimicmotion_models.vae, 
            image_encoder = mimicmotion_models.image_encoder, 
            unet = mimicmotion_models.unet, 
            scheduler = scheduler,
            feature_extractor = mimicmotion_models.feature_extractor, 
            pose_net = mimicmotion_models.pose_net,
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
        pbar.update(1)
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
            "context_size": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
            "context_overlap": ("INT", {"default": 6, "min": 1, "max": 128, "step": 1}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),            
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "MimicMotionWrapper"

    def process(self, mimic_pipeline, ref_image, pose_images, cfg_min, cfg_max, steps, seed, noise_aug_strength, fps, keep_model_loaded, context_size, context_overlap):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mimic_pipeline['dtype']
        pipeline = mimic_pipeline['pipeline']

        B, H, W, C = pose_images.shape
       
        ref_image = ref_image.permute(0, 3, 1, 2)
        pose_images = pose_images.permute(0, 3, 1, 2)

        if ref_image.shape[1:3] != (224, 224):
            ref_img = comfy.utils.common_upscale(ref_image, 224, 224, "lanczos", "disabled")
        else:
            ref_img = ref_image

        pose_images = pose_images * 2 - 1

        ref_img = ref_img.to(device).to(dtype)
        pose_images = pose_images.to(device).to(dtype)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        
        frames = pipeline(
            ref_img, 
            image_pose=pose_images, 
            num_frames=B,
            tile_size = context_size, 
            tile_overlap= context_overlap,
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
        frames = frames.squeeze(0)[1:].permute(0, 2, 3, 1).cpu().float()

        if not keep_model_loaded:
            pipeline.unet.to(offload_device)
            pipeline.vae.to(offload_device)

            mm.soft_empty_cache()
            gc.collect()

        return frames,

class MimicMotionGetPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ref_image": ("IMAGE",),
            "pose_images": ("IMAGE",),
            "include_body": ("BOOLEAN", {"default": True}),
            "include_hand": ("BOOLEAN", {"default": True}),
            "include_face": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("poses_with_ref", "pose_images")
    FUNCTION = "process"
    CATEGORY = "MimicMotionWrapper"

    def process(self, ref_image, pose_images, include_body, include_hand, include_face):
        device = mm.get_torch_device()
        from .mimicmotion.dwpose.util import draw_pose
        from .mimicmotion.dwpose.dwpose_detector import DWposeDetector

        yolo_model = "yolox_l.onnx"
        dw_pose_model = "dw-ll_ucoco_384.onnx"
        model_base_path = os.path.join(script_directory, "models", "DWPose")

        model_det=os.path.join(model_base_path, yolo_model)
        model_pose=os.path.join(model_base_path, dw_pose_model)

        if not os.path.exists(model_det):
            print(f"Downloading yolo model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="yzd-v/DWPose", 
                                allow_patterns=[f"*{yolo_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        if not os.path.exists(model_pose):
            print(f"Downloading dwpose model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="yzd-v/DWPose", 
                                allow_patterns=[f"*{dw_pose_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)

        dwprocessor = DWposeDetector(
            model_det=os.path.join(model_base_path, "yolox_l.onnx"),
            model_pose=os.path.join(model_base_path, "dw-ll_ucoco_384.onnx"),
            device=device)
        
        ref_image = ref_image.squeeze(0).cpu().numpy() * 255

        # select ref-keypoint from reference pose for pose rescale
        ref_pose = dwprocessor(ref_image)
        ref_keypoint_id = [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]
        ref_keypoint_id = [i for i in ref_keypoint_id \
            if ref_pose['bodies']['score'].shape[0] > 0 and ref_pose['bodies']['score'][0][i] > 0.3]
        ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]
 

        height, width, _ = ref_image.shape
        pose_images_np = pose_images.cpu().numpy() * 255

        # read input video
        pbar = comfy.utils.ProgressBar(len(pose_images_np))
        detected_poses_np_list = []
        for img_np in pose_images_np:
            detected_poses_np_list.append(dwprocessor(img_np))
            pbar.update(1)

        detected_bodies = np.stack(
            [p['bodies']['candidate'] for p in detected_poses_np_list if p['bodies']['candidate'].shape[0] == 18])[:,
                        ref_keypoint_id]
        # compute linear-rescale params
        ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
        fh, fw, _ = pose_images_np[0].shape
        ax = ay / (fh / fw / height * width)
        bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
        a = np.array([ax, ay])
        b = np.array([bx, by])
        output_pose = []
        # pose rescale 
        for detected_pose in detected_poses_np_list:
            detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
            detected_pose['faces'] = detected_pose['faces'] * a + b
            detected_pose['hands'] = detected_pose['hands'] * a + b
            im = draw_pose(detected_pose, height, width, include_body=include_body, include_hand=include_hand, include_face=include_face)
            output_pose.append(np.array(im))

        output_pose_tensors = [torch.tensor(np.array(im)) for im in output_pose]
        output_tensor = torch.stack(output_pose_tensors) / 255

        ref_pose_img = draw_pose(ref_pose, height, width, include_body=include_body, include_hand=include_hand, include_face=include_face)
        ref_pose_tensor = torch.tensor(np.array(ref_pose_img)) / 255
        output_tensor = torch.cat((ref_pose_tensor.unsqueeze(0), output_tensor))
        output_tensor = output_tensor.permute(0, 2, 3, 1).cpu().float()
        
        return output_tensor, output_tensor[1:]

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadMimicMotionModel": DownloadAndLoadMimicMotionModel,
    "MimicMotionSampler": MimicMotionSampler,
    "MimicMotionGetPoses": MimicMotionGetPoses

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadMimicMotionModel": "DownloadAndLoadMimicMotionModel",
    "MimicMotionSampler": "MimicMotionSampler",
    "MimicMotionGetPoses": "MimicMotionGetPoses"
}
