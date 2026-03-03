import gradio as gr
import os
import zipfile
import subprocess
import shutil
import time
from PIL import Image

# การตั้งค่า Path สำหรับ RunPod
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
    """ล้างและสร้างโฟลเดอร์ชั่วคราว"""
    for d in [temp_in, temp_out]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def handle_upload(files):
    """จัดการไฟล์ที่อัปโหลดและแสดงจำนวนรูป"""
    if not files:
        return "⚠️ ยังไม่มีไฟล์ถูกเลือก", []
    
    # ล้างข้อมูลเก่า
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
    """ฟังก์ชันรัน AI ทีละรูป รองรับชื่อไฟล์มีช่องว่าง"""
    state.is_running = True
    state.should_cancel = False
    state.is_paused = False
    
    if os.path.exists(temp_out): shutil.rmtree(temp_out)
    os.makedirs(temp_out, exist_ok=True)

    total = len(file_list)
    if total == 0:
        raise gr.Error("ไม่พบไฟล์รูปภาพ!")

    # แมปค่าโมเดล
    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model = model_map[model_choice]
    res_val = {"2K (1440p)": 1440, "4K (2160p)": 2160, "6K (3240p)": 3240, "8K (4320p)": 4320}[upscale_size]

    for i, filename in enumerate(file_list):
        if state.should_cancel:
            yield "🛑 ยกเลิกการทำงานแล้ว", None, None, None
            break

        while state.is_paused:
            yield f"⏸️ พักที่รูป {i+1}/{total} (รอการ Resume)", None, None, None
            time.sleep(1)

        input_path = os.path.join(temp_in, filename)
        
        # 1. แสดงรูป Before ทันทีและอัปเดตสถานะ
        msg = f"⏳ กำลังรันรูปที่ {i+1}/{total}: {filename}\n(ขั้นตอนนี้ใช้เวลา 5-10 นาที...)"
        progress(i/total, desc=f"Processing {i+1}/{total}")
        yield msg, input_path, None, None

        # 2. คำสั่งรัน AI (ใช้แบบ List เพื่อกันปัญหาช่องว่างในชื่อไฟล์)
        cmd = [
            "python", "inference_cli.py",
            "--output", temp_out,
            "--resolution", str(res_val),
            "--dit_model", selected_model,
            "--vae_encode_tiled", "--vae_decode_tiled",
            "--vae_encode_tile_size", "256", "--vae_decode_tile_size", "256",
            "--dit_offload_device", "cpu",
            input_path
        ]
        
        # รัน AI จริงๆ
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_DIR)

        # 3. ตรวจสอบผลลัพธ์
        new_files = [f for f in os.listdir(temp_out) if f.lower().endswith(('.png', '.jpg'))]
        base_name = os.path.splitext(filename)[0]
        matches = [f for f in new_files if base_name in f]

        if not matches:
            yield f"⚠️ ข้ามรูป {filename} (หาผลลัพธ์ไม่เจอ)", input_path, None, None
            continue

        out_path = os.path.join(temp_out, matches[-1])

        # 4. แปลงเป็น JPG Quality 100
        if format_out == 'jpg' and out_path.lower().endswith('.png'):
            jpg_path = os.path.splitext(out_path)[0] + ".jpg"
            with Image.open(out_path) as im:
                im.convert('RGB').save(jpg_path, "JPEG", quality=100, subsampling=0)
            os.remove(out_path)
            out_path = jpg_path

        # แสดงพรีวิวผลลัพธ์
        yield f"✅ ทำรูปที่ {i+1}/{total} เสร็จแล้ว", input_path, out_path, None

    # บีบไฟล์ ZIP ตอนจบ
    zip_path = os.path.join(WORKSPACE_DIR, "output_result.zip")
    if os.path.exists(zip_path): os.remove(zip_path)
    if os.listdir(temp_out):
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_out)
        yield "🎉 เสร็จสิ้นครบทุกรูป!", None, None, zip_path
    else:
        yield "❌ ไม่พบไฟล์ผลลัพธ์ที่รันสำเร็จ", None, None, None

# --- UI (Gradio) ---
with gr.Blocks(title="SeedVR2 Pro", css=".gradio-container {background-color: #0b111b; color: white;}") as demo:
    gr.Markdown("# 🚀 SeedVR2 Auto Upscaler Pro (Batch Mode)")
    
    files_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            upload_ui = gr.File(label="อัปโหลดรูป/ZIP", file_count="multiple")
            status_msg = gr.Markdown("### 📋 สถานะ: รอรับไฟล์...")
            
            with gr.Row():
                model_choice = gr.Dropdown(choices=["3B FP8 (สมดุล/ค่าเริ่มต้น)", "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)", "7B GGUF Q4 (สวยและประหยัด VRAM)", "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)"], label="🧠 โมเดล", value="3B FP8 (สมดุล/ค่าเริ่มต้น)")
                upscale_size = gr.Radio(["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], label="🎯 ความละเอียด", value="2K (1440p)")
            
            format_out = gr.Radio(["png", "jpg"], label="🖼️ นามสกุล", value="jpg")
            
            with gr.Row():
                start_btn = gr.Button("▶️ เริ่ม", variant="primary")
                pause_btn = gr.Button("⏸️ พัก")
                cancel_btn = gr.Button("🛑 ยกเลิก", variant="stop")
            
            restart_btn = gr.Button("🔄 ล้างข้อมูลใหม่")

        with gr.Column(scale=2):
            with gr.Row():
                prev_before = gr.Image(label="ก่อน (Before)", interactive=False)
                prev_after = gr.Image(label="หลัง (After)", interactive=False)
            download_ui = gr.File(label="⬇️ ดาวน์โหลด ZIP")

    # การเชื่อมต่อ UI กับ Logic
    upload_ui.change(handle_upload, inputs=[upload_ui], outputs=[status_msg, files_state])
    
    def toggle_pause():
        state.is_paused = not state.is_paused
        return "▶️ ทำต่อ" if state.is_paused else "⏸️ พัก"

    pause_btn.click(toggle_pause, outputs=[pause_btn])
    cancel_btn.click(lambda: setattr(state, 'should_cancel', True), outputs=[status_msg])
    restart_btn.click(lambda: (setup_dirs(), "### 📋 พร้อมรันใหม่", None, None, None)[1:], outputs=[status_msg, prev_before, prev_after, download_ui])

    start_btn.click(process_images, inputs=[files_state, format_out, upscale_size, model_choice], outputs=[status_msg, prev_before, prev_after, download_ui])

if __name__ == "__main__":
    setup_dirs()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
