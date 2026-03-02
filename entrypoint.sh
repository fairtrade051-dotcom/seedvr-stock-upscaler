#!/bin/bash
# 1. เข้าไปในโฟลเดอร์งาน
cd /root/SeedVR2

# 2. ติดตั้ง Library (ขั้นตอนนี้อาจใช้เวลา 2-3 นาที)
pip install -r requirements.txt

# 3. ตรวจสอบว่ามีโมเดลหรือยัง (ถ้าไม่มีให้โหลด - ตัวอย่างโมเดล 3B)
if [ ! -f "models/seedvr2_ema_3b_fp8_e4m3fn.safetensors" ]; then
    echo "📥 Downloading Model..."
    # ใส่ URL ตรงของโมเดลคุณที่นี่
    wget -O models/seedvr2_ema_3b_fp8_e4m3fn.safetensors https://huggingface.co/your-repo/model-url
fi

# 4. รันหน้าเว็บ Gradio
python app.py
