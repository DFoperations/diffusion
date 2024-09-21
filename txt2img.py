# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:47:58 2024

@author: Sankalp Sahu
"""


import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import cv2
import numpy as np
# Replace the model version with your required version if needed
pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)

# Running the inference on GPU with cuda enabled
pipeline = pipeline.to('cuda')

prompt = 'a man telling story to his daughter on a snow camp with bonefire around on a night sky with giant moon rings visible'

image = pipeline(prompt=prompt).images[0]
image_np = np.array(image)
image_bgr = image_np[:, :, ::-1]
cv2.imwrite("E:\license_img\generated_images\image_8.png", image_bgr)
