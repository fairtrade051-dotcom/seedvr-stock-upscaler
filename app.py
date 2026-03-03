import gradio as gr
import os
import zipfile
import subprocess
import shutil
import time
from PIL import Image

# --- การตั้งค่าเส้นทาง (Paths) ---
WORKSPACE_DIR = "/workspace/ComfyUI-SeedVR2_VideoUpscaler"
temp_in = os.path.join(WORKSPACE_DIR, "temp_in")
temp_out = os.path.join(WORKSPACE_DIR, "temp_out")

# --- ระบบจัดการสถานะ (Global State) ---
class GlobalState:
    is_running = False
    is_paused = False
    should_cancel = False
    current_index = 0

state = GlobalState()

def setup_dirs():
    """สร้างโฟลเดอร์ชั่วคราวสำหรับงานแต่ละรอบ"""
    for d in [temp_in, temp_out]:
        if not os.path.exists(d):
            os.makedirs(d)

def handle_upload(files):
    """จัดการไฟล์ที่อัปโหลดและเตรียมตัวอย่างรูปแรก (Before)"""
    if not files:
        return "⚠️ ยังไม่มีไฟล์ถูกเลือก", [], None
    
    # ล้างโฟลเดอร์ input เก่าก่อนรับใหม่
    if os.path.exists(temp_in): shutil.rmtree(temp_in)
    os.makedirs(temp_in, exist_ok=True)

    for f in files:
        if f.name.lower().endswith('.zip'):
            with zipfile.ZipFile(f.name, 'r') as z:
                z.extractall(temp_in)
        else:
            shutil.copy(f.name, temp_in)
    
    all_files = sorted([f for f in os.listdir(temp_in) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # ดึงรูปแรกมาแสดงพรีวิวทันทีที่อัปโหลด
    first_image_path = os.path.join(temp_in, all_files[0]) if all_files else None
    
    return f"📦 พร้อมประมวลผล: {len(all_files)} รูป", all_files, first_image_path

def process_images(file_list, format_out, upscale_size, model_choice, steps, guidance, progress=gr.Progress()):
    """ฟังก์ชันหลัก: วนลูปส่งคำสั่งรัน AI ทีละรูป"""
    state.is_running = True
    state.should_cancel = False
    state.is_paused = False
    
    # ล้างโฟลเดอร์ผลลัพธ์เก่าก่อนเริ่มงานใหม่
    if os.path.exists(temp_out): shutil.rmtree(temp_out)
    os.makedirs(temp_out, exist_ok=True)

    total = len(file_list)
    if total == 0:
        raise gr.Error("ไม่พบไฟล์รูปภาพในคิว!")

    # แปลงชื่อโมเดลเป็นชื่อไฟล์จริง
    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model = model_map[model_choice]
    res_val = {"2K (1440p)": 1440, "4K (2160p)": 2160, "6K (3240p)": 3240, "8K (4320p)": 4320}[upscale_size]

    for i, filename in enumerate(file_list):
        # ตรวจสอบปุ่ม Cancel
        if state.should_cancel:
            yield "🛑 ยกเลิกการทำงานแล้ว", None, None, None
            state.is_running = False
            return

        # ตรวจสอบปุ่ม Pause
        while state.is_paused:
            yield f"⏸️ พักที่รูป {i+1}/{total} (กด 'ทำต่อ' เพื่อรันต่อ)", None, None, None
            time.sleep(1)
            if state.should_cancel: break

        input_path = os.path.join(temp_in, filename)
        
        # แจ้งเตือนหน้าเว็บว่ากำลังทำรูปไหน (ส่งรูป Before เข้าไปโชว์)
        progress(i/total, desc=f"กำลังทำรูปที่ {i+1}/{total}")
        yield f"⏳ กำลังรันรูป {i+1}/{total}: {filename} (ใช้เวลา 5-10 นาที...)", input_path, None, None

        # คำสั่งรัน AI แบบ List (ป้องกันปัญหาช่องว่างในชื่อไฟล์)
        cmd = [
            "python", "inference_cli.py",
            "--output", temp_out,
            "--resolution", str(res_val),
            "--dit_model", selected_model,
            "--steps", str(steps),
            "--guidance", str(guidance),
            "--vae_encode_tiled", "--vae_decode_tiled",
            "--vae_encode_tile_size", "256", "--vae_decode_tile_size", "256",
            "--dit_offload_device", "cpu",
            input_path
        ]
        
        # รัน AI
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_DIR)

        # หาไฟล์ที่ออกมา
        new_files = [f for f in os.listdir(temp_out) if f.lower().endswith(('.png', '.jpg'))]
        base_name = os.path.splitext(filename)[0]
        matches = [f for f in new_files if base_name in f]

        if matches:
            out_path = os.path.join(temp_out, matches[-1])
            
            # แปลงเป็น JPG Quality 100 ถ้าผู้ใช้เลือก
            if format_out == 'jpg' and out_path.lower().endswith('.png'):
                jpg_path = os.path.splitext(out_path)[0] + ".jpg"
                with Image.open(out_path) as im:
                    im.convert('RGB').save(jpg_path, "JPEG", quality=100, subsampling=0)
                os.remove(out_path)
                out_path = jpg_path
            
            # ส่งพรีวิวผลลัพธ์ (After) ไปโชว์คู่กับต้นฉบับ (Before)
            yield f"✅ เสร็จรูปที่ {i+1}/{total}", input_path, out_path, None
        else:
            yield f"⚠️ หารูป {filename} ไม่เจอ", input_path, None, None

    # เมื่อทำครบทุกรูป บีบไฟล์เป็น ZIP
    zip_path = os.path.join(WORKSPACE_DIR, "output_result.zip")
    if os.path.exists(zip_path): os.remove(zip_path)
    if os.listdir(temp_out):
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_out)
        state.is_running = False
        yield "🎉 เสร็จครบทุกรูปแล้ว! ดาวน์โหลดได้ที่ด้านล่าง", None, None, zip_path
    else:
        yield "❌ รันจบแต่ไม่พบไฟล์ผลลัพธ์", None, None, None

# --- ส่วนติดต่อผู้ใช้งาน (UI) ---
with gr.Blocks(title="SeedVR2 Pro Sharp") as demo:
    gr.Markdown("# 🚀 SeedVR2 Pro Sharp (Batch Upscaler)")
    gr.Markdown("เพิ่มความคมชัดด้วยการปรับ Steps และ Guidance Scale (CFG)")
    
    files_state = gr.State([]) # สำหรับเก็บรายชื่อไฟล์ในคิว
    
    with gr.Row():
        # ฝั่งซ้าย: การตั้งค่าและปุ่มควบคุม
        with gr.Column(scale=1):
            upload_ui = gr.File(label="1. อัปโหลดรูป (ลากวางได้หลายรูป) หรือ .zip", file_count="multiple")
            
            with gr.Accordion("⚙️ ตั้งค่าระดับความคมชัด", open=True):
                model_choice = gr.Dropdown(
                    choices=[
                        "3B FP8 (สมดุล/ค่าเริ่มต้น)", 
                        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)", 
                        "7B GGUF Q4 (สวยและประหยัด VRAM)",
                        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)"
                    ], 
                    label="🧠 โมเดล AI", 
                    value="3B FP8 (สมดุล/ค่าเริ่มต้น)"
                )
                upscale_size = gr.Radio(["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], label="🎯 ความละเอียด", value="2K (1440p)")
                
                steps_slider = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="🔄 จำนวนรอบ (Steps) - ยิ่งมากยิ่งละเอียด (ช้าลง)")
                guidance_slider = gr.Slider(minimum=1.0, maximum=15.0, value=4.0, step=0.5, label="✨ ความคม/Contrast (Guidance) - ยิ่งมากยิ่งคมเข้ม")
                
                format_out = gr.Radio(["png", "jpg"], label="🖼️ นามสกุลไฟล์", value="jpg")

            with gr.Row():
                start_btn = gr.Button("▶️ เริ่มอัปสเกล", variant="primary")
                pause_btn = gr.Button("⏸️ พัก/ทำต่อ")
                cancel_btn = gr.Button("🛑 ยกเลิก", variant="stop")
            
            restart_btn = gr.Button("🔄 ล้างคิวและลบไฟล์ชั่วคราว")

        # ฝั่งขวา: สถานะและพรีวิว
        with gr.Column(scale=2):
            status_msg = gr.Markdown("### 📋 สถานะ: รอการอัปโหลด...")
            
            with gr.Row():
                # แสดงภาพเปรียบเทียบ Before/After
                prev_before = gr.Image(label="ก่อน (Before)", type="filepath", interactive=False)
                prev_after = gr.Image(label="หลัง (After)", type="filepath", interactive=False)
            
            download_ui = gr.File(label="⬇️ ดาวน์โหลดผลลัพธ์ทั้งหมด (ZIP)")

    # --- ส่วนการทำงาน (Logic) ---
    
    # เมื่อไฟล์ถูกอัปโหลด
    upload_ui.change(handle_upload, inputs=[upload_ui], outputs=[status_msg, files_state, prev_before])
    
    # ปุ่ม Pause
    def toggle_pause():
        state.is_paused = not state.is_paused
        return "▶️ ทำต่อ" if state.is_paused else "⏸️ พัก"

    # ปุ่ม Restart
    def restart_all():
        setup_dirs()
        return "### 📋 ล้างข้อมูลเรียบร้อย พร้อมรับไฟล์ใหม่", None, None, None

    pause_btn.click(toggle_pause, outputs=[pause_btn])
    cancel_btn.click(lambda: setattr(state, 'should_cancel', True), outputs=None)
    restart_btn.click(restart_all, outputs=[status_msg, prev_before, prev_after, download_ui])

    # ปุ่ม Start
    start_btn.click(
        fn=process_images,
        inputs=[files_state, format_out, upscale_size, model_choice, steps_slider, guidance_slider],
        outputs=[status_msg, prev_before, prev_after, download_ui]
    )

# รันหน้าเว็บ
if __name__ == "__main__":
    setup_dirs()
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True, 
        allowed_paths=["/workspace"],
        css=".gradio-container {background-color: #0b111b; color: #ffffff;}"
    )
