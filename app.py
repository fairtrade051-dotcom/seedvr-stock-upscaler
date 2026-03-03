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
    """เตรียมโฟลเดอร์ทำงาน"""
    if not os.path.exists(temp_in): os.makedirs(temp_in)
    if not os.path.exists(temp_out): os.makedirs(temp_out)

def handle_upload(files):
    """จัดการไฟล์ที่อัปโหลดและส่งรูปแรกไปโชว์ Before ทันที"""
    if not files:
        return "⚠️ ยังไม่มีไฟล์ถูกเลือก", [], None
    
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
    
    # ดึง Path ของรูปแรกมาโชว์พรีวิว
    first_image_path = os.path.join(temp_in, all_files[0]) if all_files else None
    
    return f"📦 พร้อมประมวลผล: {len(all_files)} รูป", all_files, first_image_path

def process_images(file_list, format_out, upscale_size, model_choice, progress=gr.Progress()):
    """รัน AI ทีละรูป และอัปเดตพรีวิว Before/After"""
    state.is_running = True
    state.should_cancel = False
    state.is_paused = False
    
    # ล้าง Output เก่าก่อนรันใหม่
    if os.path.exists(temp_out): shutil.rmtree(temp_out)
    os.makedirs(temp_out, exist_ok=True)

    total = len(file_list)
    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model = model_map[model_choice]
    res_val = {"2K (1440p)": 1440, "4K (2160p)": 2160, "6K (3240p)": 3240, "8K (4320p)": 4320}[upscale_size]

    for i, filename in enumerate(file_list):
        if state.should_cancel: break
        while state.is_paused: time.sleep(1)

        input_path = os.path.join(temp_in, filename)
        
        # อัปเดตสถานะและพรีวิวรูปฝั่งซ้าย (Before)
        msg = f"⏳ กำลังรันรูป {i+1}/{total}: {filename}"
        progress(i/total, desc=f"ทำรูปที่ {i+1}/{total}")
        yield msg, input_path, None, None

        cmd = [
            "python", "inference_cli.py", "--output", temp_out, "--resolution", str(res_val),
            "--dit_model", selected_model, "--vae_encode_tiled", "--vae_decode_tiled",
            "--vae_encode_tile_size", "256", "--vae_decode_tile_size", "256",
            "--dit_offload_device", "cpu", input_path
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_DIR)

        # หาไฟล์ผลลัพธ์
        new_files = [f for f in os.listdir(temp_out) if f.lower().endswith(('.png', '.jpg'))]
        base_name = os.path.splitext(filename)[0]
        matches = [f for f in new_files if base_name in f]

        if matches:
            out_path = os.path.join(temp_out, matches[-1])
            # แปลงไฟล์เป็น JPG (จุดที่แก้ Syntax Error)
            if format_out == 'jpg' and out_path.lower().endswith('.png'):
                jpg_path = os.path.splitext(out_path)[0] + ".jpg"
                with Image.open(out_path) as im:
                    im.convert('RGB').save(jpg_path, "JPEG", quality=100, subsampling=0)
                os.remove(out_path)
                out_path = jpg_path
            
            # ส่งผลลัพธ์ไปโชว์ฝั่งขวา (After)
            yield f"✅ เสร็จรูปที่ {i+1}/{total}", input_path, out_path, None
        else:
            yield f"⚠️ หารูป {filename} ไม่เจอ", input_path, None, None

    # จบงาน บีบ ZIP
    zip_path = os.path.join(WORKSPACE_DIR, "output_result.zip")
    if os.path.exists(zip_path): os.remove(zip_path)
    if os.listdir(temp_out):
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_out)
        yield "🎉 เสร็จครบทุกรูปแล้ว!", None, None, zip_path
    else:
        yield "❌ ไม่พบไฟล์ผลลัพธ์ที่รันสำเร็จ", None, None, None

# --- UI Layout ---
with gr.Blocks(title="SeedVR2 Pro", css=".gradio-container {background-color: #0b111b; color: #ffffff;}") as demo:
    gr.Markdown("# 🚀 SeedVR2 Auto Upscaler Pro")
    
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
            
            restart_btn = gr.Button("🔄 ล้างข้อมูล")

        with gr.Column(scale=2):
            with gr.Row():
                prev_before = gr.Image(label="ก่อน (Before)", type="filepath", interactive=False)
                prev_after = gr.Image(label="หลัง (After)", type="filepath", interactive=False)
            download_ui = gr.File(label="⬇️ ดาวน์โหลด ZIP")

    # --- Logic ---
    upload_ui.change(handle_upload, inputs=[upload_ui], outputs=[status_msg, files_state, prev_before])
    
    def toggle_pause():
        state.is_paused = not state.is_paused
        return "▶️ ทำต่อ" if state.is_paused else "⏸️ พัก"

    pause_btn.click(toggle_pause, outputs=[pause_btn])
    cancel_btn.click(lambda: setattr(state, 'should_cancel', True), outputs=None)
    restart_btn.click(lambda: (setup_dirs(), "### 📋 พร้อมรันใหม่", None, None, None)[1:], outputs=[status_msg, prev_before, prev_after, download_ui])

    start_btn.click(process_images, inputs=[files_state, format_out, upscale_size, model_choice], outputs=[status_msg, prev_before, prev_after, download_ui])

if __name__ == "__main__":
    setup_dirs()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, allowed_paths=["/workspace"])
