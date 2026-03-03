#!/bin/bash
cd /workspace

# 1. โหลดโค้ด AI ต้นฉบับ
if [ ! -d "ComfyUI-SeedVR2_VideoUpscaler" ]; then
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
fi

# 2. ก๊อปไฟล์หน้าเว็บ (app.py) จาก Repo ของคุณไปใส่ในโฟลเดอร์ AI
cp /workspace/my_template_repo/app.py /workspace/ComfyUI-SeedVR2_VideoUpscaler/

cd /workspace/ComfyUI-SeedVR2_VideoUpscaler

# 3. ติดตั้ง Dependencies และ Jupyter Lab
# เพิ่ม jupyterlab เข้าไปในลิสต์ติดตั้ง
pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu121
pip install jupyterlab einops safetensors "diffusers==0.29.2" transformers accelerate pillow gradio opencv-python-headless rotary-embedding-torch omegaconf gguf

# 4. สั่งรัน Jupyter Lab ในเบื้องหลัง (พอร์ต 8080)
# เครื่องหมาย & ที่ต่อท้ายหมายถึงให้รันค้างไว้แล้วไปทำคำสั่งถัดไปทันที
nohup jupyter lab --allow-root --ip=0.0.0.0 --port=8080 --no-browser --ServerApp.token='' --ServerApp.password='' > /workspace/jupyter.log 2>&1 &

# 5. รันหน้าเว็บ AI (พอร์ต 7860) เป็นหลัก
python app.py
