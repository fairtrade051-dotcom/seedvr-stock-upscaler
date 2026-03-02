#!/bin/bash
cd /workspace

# 1. โหลดโค้ด AI ต้นฉบับ
if [ ! -d "ComfyUI-SeedVR2_VideoUpscaler" ]; then
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
fi

# 2. ก๊อปไฟล์หน้าเว็บ (app.py) จาก Repo ของคุณไปใส่ในโฟลเดอร์ AI
cp /workspace/my_template_repo/app.py /workspace/ComfyUI-SeedVR2_VideoUpscaler/

cd /workspace/ComfyUI-SeedVR2_VideoUpscaler

# 3. อัปเกรด PyTorch ให้เป็นเวอร์ชัน 2.4+
pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu121

# 4. ติดตั้ง Dependencies (เพิ่ม gguf เข้าไปท้ายสุดแล้ว)
pip install einops safetensors "diffusers==0.29.2" transformers accelerate pillow gradio opencv-python-headless rotary-embedding-torch omegaconf gguf

# 5. รันหน้าเว็บ
python app.py
