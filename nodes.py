import os
import torch
import numpy as np
import gc

import folder_paths
import comfy.model_management as mm
import comfy.utils

try:
    import diffusers.models.activations
    def patch_geglu_inplace():
        """Patch GEGLU with inplace multiplication to save GPU memory."""
        def forward(self, hidden_states):
            hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
            return hidden_states.mul_(self.gelu(gate))
        diffusers.models.activations.GEGLU.forward = forward
except:
    pass

from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

script_directory = os.path.dirname(os.path.abspath(__file__))

from .mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline, tensor2vid
from .mimicmotion.modules.unet import UNetSpatioTemporalConditionModel
from .mimicmotion.modules.pose_net import PoseNet

from .lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False
    pass


def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys

class DownloadAndLoadMimicMotionModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [   'MimicMotionMergedUnet_1-0-fp16.safetensors',
                        'MimicMotionMergedUnet_1-1-fp16.safetensors',
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

        pbar = comfy.utils.ProgressBar(5)
        
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
        
        if not os.path.exists(svd_path):
            print(f"Downloading SVD model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="vdo/stable-video-diffusion-img2vid-xt-1-1", 
                                allow_patterns=[f"*.json", "*fp16*"],
                                ignore_patterns=["*unet*"],
                                local_dir=svd_path, 
                                local_dir_use_symlinks=False)
        pbar.update(1)

        unet_config = UNetSpatioTemporalConditionModel.load_config(os.path.join(script_directory, "configs", "unet_config.json"))
        print("Loading UNET")
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            self.unet = UNetSpatioTemporalConditionModel.from_config(unet_config)
        sd = comfy.utils.load_torch_file(os.path.join(model_path))
        if is_accelerate_available:
            for key in sd:
                set_module_tensor_to_device(self.unet, key, dtype=dtype, device=device, value=sd[key])
        else:
            self.unet.load_state_dict(sd, strict=False)
        del sd
        pbar.update(1)

        print("Loading VAE")
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae", variant="fp16", low_cpu_mem_usage=True).to(dtype).to(device).eval()

        print("Loading IMAGE_ENCODER")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(svd_path, subfolder="image_encoder", variant="fp16", low_cpu_mem_usage=True).to(dtype).to(device).eval()
        pbar.update(1)
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(svd_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(svd_path, subfolder="feature_extractor")
        
        print("Loading POSE_NET")
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0]).to(dtype).to(device).eval()
        pose_net_sd = comfy.utils.load_torch_file(os.path.join(script_directory, 'models', 'mimic_motion_pose_net.safetensors'))
        
        self.unet.load_state_dict(pose_net_sd, strict=False)
        self.pose_net.load_state_dict(pose_net_sd, strict=False)
        del pose_net_sd
       
        pipeline = MimicMotionPipeline(
            vae = self.vae, 
            image_encoder = self.image_encoder, 
            unet = self.unet, 
            scheduler = self.noise_scheduler,
            feature_extractor = self.feature_extractor, 
            pose_net = self.pose_net,
        )
        
        mimic_model = {
            'pipeline': pipeline,
            'dtype': dtype
        }
        pbar.update(1)
        return (mimic_model,)
    
class DiffusersScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "scheduler": (
                    [   
                        'EulerDiscreteScheduler',
                        'AnimateLCM_SVD'
                    ],
                    ), 
            "sigma_min": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 700.0, "step": 0.001}),
            "sigma_max": ("FLOAT", {"default": 700.0, "min": 0.0, "max": 700.0, "step": 0.001}),
            "align_your_steps": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("DIFFUSERS_SCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "loadmodel"
    CATEGORY = "MimicMotionWrapper"

    def loadmodel(self, scheduler, sigma_min, sigma_max, align_your_steps):

        scheduler_config = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "interpolation_type": "linear",
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "set_alpha_to_one": False,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "timestep_type": "continuous",
            "trained_betas": None,
            "use_karras_sigmas": True
            }
        if scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        elif scheduler == 'AnimateLCM_SVD':
            noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
                num_train_timesteps=40,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_data=1.0,
                s_noise=1.0,
                rho=7,
                clip_denoised=False,
            )
        if align_your_steps:
            sigmas = [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002]
        
        scheduler_options = {
            "noise_scheduler": noise_scheduler,
            "sigmas": sigmas if align_your_steps else None,
        }

        return (scheduler_options,)
        
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
            "optional": {
                "optional_scheduler": ("DIFFUSERS_SCHEDULER",),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_embed_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "MimicMotionWrapper"

    def process(self, mimic_pipeline, ref_image, pose_images, cfg_min, cfg_max, steps, seed, noise_aug_strength, fps, keep_model_loaded, 
                context_size, context_overlap, optional_scheduler=None, pose_strength=1.0, image_embed_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mimic_pipeline['dtype']
        pipeline = mimic_pipeline['pipeline']           

        original_scheduler = pipeline.scheduler

        if optional_scheduler is not None:
            print("Using optional scheduler: ", optional_scheduler['noise_scheduler'])
            pipeline.scheduler = optional_scheduler['noise_scheduler']
            sigmas = optional_scheduler['sigmas']

            if sigmas is not None and (steps + 1) != len(sigmas):
                sigmas = loglinear_interp(sigmas, steps + 1)
                sigmas = sigmas[-(steps + 1):]
                sigmas[-1] = 0
                print("Using timesteps: ", sigmas)
        else:
            pipeline.scheduler = original_scheduler
            sigmas = None
  
        B, H, W, C = pose_images.shape

        assert B >= context_size, "The number of poses must be greater than the context size"

        ref_image = ref_image.permute(0, 3, 1, 2)
        pose_images = pose_images.permute(0, 3, 1, 2)

        pose_images = pose_images * 2 - 1

        ref_image = ref_image.to(device).to(dtype)
        pose_images = pose_images.to(device).to(dtype)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        
        frames = pipeline(
            ref_image, 
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
            decode_chunk_size=4, 
            output_type="latent", 
            device=device,
            sigmas=sigmas,
            pose_strength=pose_strength,
            pose_start_percent=pose_start_percent,
            pose_end_percent=pose_end_percent,
            image_embed_strength=image_embed_strength
        ).frames

        if not keep_model_loaded:
            pipeline.unet.to(offload_device)
            pipeline.vae.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        return {"samples": frames},

class MimicMotionDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mimic_pipeline": ("MIMICPIPE",),
            "samples": ("LATENT",),
            "decode_chunk_size": ("INT", {"default": 4, "min": 1, "max": 200, "step": 1})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "MimicMotionWrapper"

    def process(self, mimic_pipeline, samples, decode_chunk_size):
        mm.soft_empty_cache()
    
        pipeline = mimic_pipeline['pipeline']
        num_frames = samples['samples'].shape[0]
        try:
            frames = pipeline.decode_latents(samples['samples'], num_frames, decode_chunk_size)
        except:
            frames = pipeline.decode_latents(samples['samples'], num_frames, 1)
        frames = tensor2vid(frames, pipeline.image_processor, output_type="pt")
        
        frames = frames.squeeze(1)[1:].permute(0, 2, 3, 1).cpu().float()

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
        offload_device = mm.unet_offload_device()
        from .mimicmotion.dwpose.util import draw_pose
        from .mimicmotion.dwpose.dwpose_detector import DWposeDetector

        assert ref_image.shape[1:3] == pose_images.shape[1:3], "ref_image and pose_images must have the same resolution"

        #yolo_model = "yolox_l.onnx"
        #dw_pose_model = "dw-ll_ucoco_384.onnx"
        dw_pose_model = "dw-ll_ucoco_384_bs5.torchscript.pt"
        yolo_model = "yolox_l.torchscript.pt"

        model_base_path = os.path.join(script_directory, "models", "DWPose")

        model_det=os.path.join(model_base_path, yolo_model)
        model_pose=os.path.join(model_base_path, dw_pose_model)

        if not os.path.exists(model_det):
            print(f"Downloading yolo model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/yolox-onnx", 
                                allow_patterns=[f"*{yolo_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        if not os.path.exists(model_pose):
            print(f"Downloading dwpose model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/DWPose-TorchScript-BatchSize5", 
                                allow_patterns=[f"*{dw_pose_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        model_det=os.path.join(model_base_path, yolo_model)
        model_pose=os.path.join(model_base_path, dw_pose_model) 

        if not hasattr(self, "det") or not hasattr(self, "pose"):
            self.det = torch.jit.load(model_det)
            self.pose = torch.jit.load(model_pose)

            self.dwprocessor = DWposeDetector(
                model_det=self.det,
                model_pose=self.pose)
        
        ref_image = ref_image.squeeze(0).cpu().numpy() * 255

        self.det = self.det.to(device)
        self.pose = self.pose.to(device)

        # select ref-keypoint from reference pose for pose rescale
        ref_pose = self.dwprocessor(ref_image)
        #ref_keypoint_id = [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]
        ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        ref_keypoint_id = [i for i in ref_keypoint_id \
            #if ref_pose['bodies']['score'].shape[0] > 0 and ref_pose['bodies']['score'][0][i] > 0.3]
            if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
        ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]
 
        height, width, _ = ref_image.shape
        pose_images_np = pose_images.cpu().numpy() * 255

        # read input video
        pbar = comfy.utils.ProgressBar(len(pose_images_np))
        detected_poses_np_list = []
        for img_np in pose_images_np:
            detected_poses_np_list.append(self.dwprocessor(img_np))
            pbar.update(1)

        self.det = self.det.to(offload_device)
        self.pose = self.pose.to(offload_device)

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
            if include_body:
                detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
            if include_hand:
                detected_pose['faces'] = detected_pose['faces'] * a + b
            if include_face:
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
    "MimicMotionGetPoses": MimicMotionGetPoses,
    "MimicMotionDecode": MimicMotionDecode,
    "DiffusersScheduler": DiffusersScheduler,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadMimicMotionModel": "(Down)Load MimicMotionModel",
    "MimicMotionSampler": "MimicMotion Sampler",
    "MimicMotionGetPoses": "MimicMotion GetPoses",
    "MimicMotionDecode": "MimicMotion Decode",
    "DiffusersScheduler": "Diffusers Scheduler",
}
