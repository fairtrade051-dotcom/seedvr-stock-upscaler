#!/bin/bash

# ย้ายไปที่พื้นที่หลักของ RunPod
cd /workspace

# Clone โค้ดถ้ายังไม่มี
if [ ! -d "ComfyUI-SeedVR2_VideoUpscaler" ]; then
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
fi

cd ComfyUI-SeedVR2_VideoUpscaler

# ติดตั้ง Dependencies
pip install -r requirements.txt
pip install gradio gradio-image-slider pillow

# (นำโค้ด Python ด้านบนไปเซฟทับหรือสร้างเป็นไฟล์ app.py ในโฟลเดอร์นี้)
# สมมติว่าคุณอัปโหลดไฟล์ app.py เข้ามาแล้ว

# เริ่มรัน Web UI
python app.py
