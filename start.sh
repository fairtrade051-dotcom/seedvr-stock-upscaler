#!/bin/bash

cd /workspace

if [ ! -d "ComfyUI-SeedVR2_VideoUpscaler" ]; then
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
fi

# ก๊อปปี้ app.py ไปใส่
cp /workspace/my_template_repo/app.py /workspace/ComfyUI-SeedVR2_VideoUpscaler/

cd /workspace/ComfyUI-SeedVR2_VideoUpscaler

# ลงเฉพาะตัวที่จำเป็น (ตัด Slider ออกแล้ว)
pip install einops safetensors diffusers transformers accelerate pillow gradio

# รันหน้าเว็บ
python app.py
