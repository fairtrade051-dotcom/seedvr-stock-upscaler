import gradio as gr
import os
import zipfile
import subprocess
import shutil
import time
from PIL import Image

# การตั้งค่า Path
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
    """ล้างและสร้างโฟลเดอร์ชั่วคราวใหม่"""
    for d in [temp_in, temp_out]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def handle_upload(files):
    """จัดการไฟล์ที่อัปโหลดเข้ามา"""
    if not files:
        return "⚠️ ยังไม่มีไฟล์ถูกเลือก", []
    
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
    return f"📦 พร้อมประมวลผล: {len(all_files)} รูป", all_files

def process_images(file_list, format_out, upscale_size, model_choice, progress=gr.Progress()):
    """ฟังก์ชันหลักในการรัน AI ทีละรูป"""
    state.is_running = True
    state.should_cancel = False
    state.is_paused = False
    
    # ล้างโฟลเดอร์ output เก่าก่อนเริ่มรันใหม่
    if os.path.exists(temp_out): shutil.rmtree(temp_out)
    os.makedirs(temp_out, exist_ok=True)

    total = len(file_list)
    if total == 0:
        raise gr.Error("ไม่พบไฟล์รูปภาพในระบบ!")

    # แมปค่าพารามิเตอร์
    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model = model_map[model_choice]
    res_val = {"2K (1440p)": 1440, "4K (2160p)": 2160, "6K (3240p)": 3240, "8K (4320p)": 4320}[upscale_size]

    for i, filename in enumerate(file_list):
        # 1. เช็คว่ากดยกเลิกหรือไม่
        if state.should_cancel:
            yield "🛑 ยกเลิกการทำงานแล้ว", None, None, None
            state.is_running = False
            return

        # 2. เช็คว่ากดพักหรือไม่
        while state.is_paused:
            yield f"⏸️ หยุดชั่วคราวที่รูป {i+1}/{total} (กด 'ทำต่อ' เพื่อรันต่อ)", None, None, None
            time.sleep(1)
            if state.should_cancel: break

        state.current_index = i
        progress(i/total, desc=f"กำลังทำรูปที่ {i+1}/{total}: {filename}")
        
        input_path = os.path.join(temp_in, filename)
        
        # 3. คำสั่งรัน AI
        cmd = (
            f"python inference_cli.py --output {temp_out} --resolution {res_val} "
            f"--dit_model {selected_model} --vae_encode_tiled --vae_decode_tiled "
            f"--vae_encode_tile_size 256 --vae_decode_tile_size 256 --dit_offload_device cpu {input_path}"
        )
        
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR)

        # 4. ตรวจสอบไฟล์ผลลัพธ์ (แก้ไข IndexError)
        new_files = [f for f in os.listdir(temp_out) if f.lower().endswith(('.png', '.jpg'))]
        base_name = os.path.splitext(filename)[0]
        # ค้นหาไฟล์ที่ชื่อมีส่วนประกอบของชื่อเดิม
        matches = [f for f in new_files if base_name in f]

        if not matches:
            print(f"❌ Error: หาไฟล์ผลลัพธ์ของ {filename} ไม่เจอ")
            yield f"⚠️ ข้ามรูป {filename} (AI ไม่สร้างไฟล์)", None, None, None
            continue

        out_path = os.path.join(temp_out, matches[-1])

        # 5. แปลงเป็น JPG ถ้าเลือกไว้
        if format_out == 'jpg' and out_path.lower().endswith('.png'):
            jpg_path = os.path.splitext(out_path)[0] + ".jpg"
            with Image.open(out_path) as im:
                im.convert('RGB').save(jpg_path, "JPEG", quality=100, subsampling=0)
            os.remove(out_path)
            out_path = jpg_path

        # แสดงพรีวิวรูปปัจจุบัน
        yield f"⏳ กำลังทำ: {i+1}/{total}", input_path, out_path, None

    # 6. จบการทำงานและบีบไฟล์ ZIP
    zip_path = os.path.join(WORKSPACE_DIR, "output_result.zip")
    if os.path.exists(zip_path): os.remove(zip_path)
    
    # ตรวจสอบว่ามีไฟล์ให้บีบไหม
    if os.listdir(temp_out):
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_out)
        state.is_running = False
        yield "✅ เสร็จสิ้นทุกรูป!", None, None, zip_path
    else:
        yield "❌ รันเสร็จแต่ไม่มีไฟล์ผลลัพธ์ออกมาเลย", None, None, None

# --- ส่วนติดต่อผู้ใช้งาน (UI) ---
with gr.Blocks(title="SeedVR2 Batch Pro", css=".gradio-container {background-color: #0b0f19; color: #e5e7eb;}") as demo:
    gr.Markdown("# 🚀 SeedVR2 Auto Upscaler Pro")
    gr.Markdown("รองรับการรันจำนวนมาก พร้อมระบบคุมสถานะ (Pause/Cancel)")
    
    files_state = gr.State([]) # เก็บรายชื่อไฟล์ในคิว
    
    with gr.Row():
        # ฝั่งควบคุม
        with gr.Column(scale=1):
            upload_ui = gr.File(label="1. อัปโหลดรูป (เลือกหลายรูปได้) หรือไฟล์ .zip", file_count="multiple")
            status_msg = gr.Markdown("### 📋 สถานะ: รอรับไฟล์...")
            
            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=["3B FP8 (สมดุล/ค่าเริ่มต้น)", "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)", "7B GGUF Q4 (สวยและประหยัด VRAM)", "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)"], 
                    label="🧠 โมเดล AI", 
                    value="3B FP8 (สมดุล/ค่าเริ่มต้น)"
                )
                upscale_size = gr.Radio(["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], label="🎯 ความละเอียด", value="2K (1440p)")
            
            format_out = gr.Radio(["png", "jpg"], label="🖼️ นามสกุลไฟล์", value="jpg")
            
            with gr.Row():
                start_btn = gr.Button("▶️ เริ่มอัปสเกล", variant="primary")
                pause_btn = gr.Button("⏸️ พัก/ทำต่อ")
                cancel_btn = gr.Button("🛑 ยกเลิก", variant="stop")
            
            restart_btn = gr.Button("🔄 ล้างคิวและโฟลเดอร์ขยะ")

        # ฝั่งแสดงผล
        with gr.Column(scale=2):
            with gr.Row():
                prev_before = gr.Image(label="ก่อน (Before)", interactive=False, height=400)
                prev_after = gr.Image(label="หลัง (After)", interactive=False, height=400)
            
            download_ui = gr.File(label="⬇️ ดาวน์โหลดผลลัพธ์ทั้งหมด (ZIP)")

    # --- ส่วนตรรกะ (Logic) ---
    
    # เมื่ออัปโหลดไฟล์
    upload_ui.change(handle_upload, inputs=[upload_ui], outputs=[status_msg, files_state])
    
    # ปุ่มพักการทำงาน
    def toggle_pause():
        state.is_paused = not state.is_paused
        return "▶️ ทำต่อ" if state.is_paused else "⏸️ พัก"

    # ปุ่มยกเลิก
    def request_cancel():
        state.should_cancel = True
        state.is_paused = False
        return "🛑 กำลังหยุดคิว..."

    # ปุ่มเริ่มใหม่
    def restart_all():
        setup_dirs()
        return "### 📋 ล้างข้อมูลแล้ว พร้อมรับไฟล์ใหม่", None, None, None

    pause_btn.click(toggle_pause, outputs=[pause_btn])
    cancel_btn.click(request_cancel, outputs=[status_msg])
    restart_btn.click(restart_all, outputs=[status_msg, prev_before, prev_after, download_ui])

    # ปุ่มเริ่มรัน
    start_btn.click(
        fn=process_images,
        inputs=[files_state, format_out, upscale_size, model_choice],
        outputs=[status_msg, prev_before, prev_after, download_ui]
    )

# รันระบบ
if __name__ == "__main__":
    setup_dirs() # เตรียมโฟลเดอร์ตอนเริ่ม
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
