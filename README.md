# WORK IN PROGRESS
# MimicMotion wrapper for ComfyUI

## Installation
Clone this repo into custom_nodes folder.

Install dependencies: `pip install -r requirements.txt`

or if you use the portable install, run this in ComfyUI_windows_portable -folder:

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-MimicMotionWrapper\requirements.txt`

Models are auto downloaded to their respective folders.

Along with the MimicMotion -model (3.05 GB), to `ComfyUI\models\mimicmotion`:

https://huggingface.co/Kijai/MimicMotion_pruned/tree/main

This needs Fp16 version (4.19 GB) diffusers version of SVD XT 1.1 to `ComfyUI/models/diffusers`:

https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main


https://github.com/kijai/ComfyUI-MimicMotionWrapper/assets/40791699/c1517e20-8537-4ab0-b6fb-2d4aefa618d2


Original repo:
https://github.com/tencent/MimicMotion
