#!/bin/bash
cd /workspace

# 1. โหลดโค้ด AI ต้นฉบับ
if [ ! -d "ComfyUI-SeedVR2_VideoUpscaler" ]; then
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
fi

# 2. ก๊อปไฟล์หน้าเว็บ (app.py) จาก Repo ของคุณไปใส่ในโฟลเดอร์ AI
cp /workspace/my_template_repo/app.py /workspace/ComfyUI-SeedVR2_VideoUpscaler/

cd /workspace/ComfyUI-SeedVR2_VideoUpscaler

# 3. ติดตั้ง Dependencies (ล็อกเวอร์ชัน diffusers แก้บั๊ก xpu)
pip install einops safetensors "diffusers==0.29.2" transformers accelerate pillow gradio opencv-python-headless

# 4. รันหน้าเว็บ
python app.py
