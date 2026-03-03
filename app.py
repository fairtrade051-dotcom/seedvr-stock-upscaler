import gradio as gr
import os
import zipfile
import subprocess
import shutil
import time
from PIL import Image

WORKSPACE_DIR = "/workspace/ComfyUI-SeedVR2_VideoUpscaler"
temp_in = os.path.join(WORKSPACE_DIR, "temp_in")
temp_out = os.path.join(WORKSPACE_DIR, "temp_out")

# --- State Management ---
class GlobalState:
    is_running = False
    is_paused = False
    should_cancel = False
    current_index = 0

state = GlobalState()

def setup_dirs():
    if os.path.exists(temp_in): shutil.rmtree(temp_in)
    if os.path.exists(temp_out): shutil.rmtree(temp_out)
    os.makedirs(temp_in, exist_ok=True)
    os.makedirs(temp_out, exist_ok=True)

def handle_upload(files):
    if not files: return "ยังไม่มีไฟล์", []
    file_list = []
    for f in files:
        if f.name.endswith('.zip'):
            with zipfile.ZipFile(f.name, 'r') as z:
                z.extractall(temp_in)
        else:
            shutil.copy(f.name, temp_in)
    
    all_files = [f for f in os.listdir(temp_in) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return f"เตรียมพร้อมแล้ว: {len(all_files)} รูป", sorted(all_files)

def process_images(file_list, format_out, upscale_size, model_choice, progress=gr.Progress()):
    state.is_running = True
    state.should_cancel = False
    state.current_index = 0
    
    total = len(file_list)
    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model = model_map[model_choice]
    res_val = {"2K (1440p)": 1440, "4K (2160p)": 2160, "6K (3240p)": 3240, "8K (4320p)": 4320}[upscale_size]

    results = []
    
    for i, filename in enumerate(file_list):
        # ตรวจสอบสถานะ Cancel
        if state.should_cancel:
            yield "ยกเลิกการทำงานแล้ว", None, None, None
            break
            
        # ตรวจสอบสถานะ Pause
        while state.is_paused:
            yield f"⏸️ หยุดชั่วคราวที่รูป {i+1}/{total}", None, None, None
            time.sleep(1)

        state.current_index = i
        progress(i/total, desc=f"กำลังทำรูปที่ {i+1}/{total}: {filename}")
        
        input_path = os.path.join(temp_in, filename)
        
        # รัน AI ทีละรูป
        cmd = (
            f"python inference_cli.py --output {temp_out} --resolution {res_val} "
            f"--dit_model {selected_model} --vae_encode_tiled --vae_decode_tiled "
            f"--vae_encode_tile_size 256 --vae_decode_tile_size 256 --dit_offload_device cpu {input_path}"
        )
        
        subprocess.run(cmd, shell=True, capture_output=True, cwd=WORKSPACE_DIR)
        
        # หาไฟล์ที่ออกมา (AI มักจะเติมเลขท้ายไฟล์)
        new_files = os.listdir(temp_out)
        current_out_name = [f for f in new_files if f.startswith(os.path.splitext(filename)[0])][-1]
        out_path = os.path.join(temp_out, current_out_name)

        # แปลงเป็น JPG ถ้าเลือกไว้
        if format_out == 'jpg' and out_path.endswith('.png'):
            jpg_path = out_path.replace('.png', '.jpg')
            with Image.open(out_path) as im:
                im.convert('RGB').save(jpg_path, "JPEG", quality=100, subsampling=0)
            os.remove(out_path)
            out_path = jpg_path

        yield f"กำลังประมวลผล: {i+1}/{total}", input_path, out_path, None

    # เมื่อเสร็จทั้งหมด
    zip_path = os.path.join(WORKSPACE_DIR, "output_result.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_out)
    state.is_running = False
    yield "✅ ทำครบทุกรูปแล้ว!", None, None, zip_path

# --- UI Layout ---
with gr.Blocks(css=".gradio-container {background-color: #111; color: white;}") as demo:
    gr.Markdown("# 🚀 SeedVR2 Batch Upscaler Pro")
    
    files_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            upload_ui = gr.File(label="อัปโหลดรูปภาพ/ZIP", file_count="multiple")
            status_msg = gr.Markdown("### สถานะ: รอการอัปโหลด...")
            
            with gr.Row():
                model_choice = gr.Dropdown(choices=["3B FP8 (สมดุล/ค่าเริ่มต้น)", "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)", "7B GGUF Q4 (สวยและประหยัด VRAM)", "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)"], label="โมเดล", value="3B FP8 (สมดุล/ค่าเริ่มต้น)")
                upscale_size = gr.Radio(["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], label="ความละเอียด", value="2K (1440p)")
            
            format_out = gr.Radio(["png", "jpg"], label="Format", value="jpg")
            
            with gr.Row():
                start_btn = gr.Button("▶️ เริ่มอัปสเกล", variant="primary")
                pause_btn = gr.Button("⏸️ พัก")
                cancel_btn = gr.Button("🛑 ยกเลิก", variant="stop")
            
            restart_btn = gr.Button("🔄 เริ่มคิวใหม่ (ล้างไฟล์เก่า)")

        with gr.Column(scale=2):
            progress_ui = gr.Markdown("### ความคืบหน้า: 0%")
            with gr.Row():
                prev_before = gr.Image(label="ก่อน (Before)", interactive=False)
                prev_after = gr.Image(label="หลัง (After)", interactive=False)
            download_ui = gr.File(label="ดาวน์โหลดผลลัพธ์ ZIP")

    # --- Logic ---
    upload_ui.change(handle_upload, inputs=[upload_ui], outputs=[status_msg, files_state])
    
    def toggle_pause():
        state.is_paused = not state.is_paused
        return "▶️ ทำต่อ" if state.is_paused else "⏸️ พัก"

    def request_cancel():
        state.should_cancel = True
        return "🛑 กำลังยกเลิก..."

    def restart_all():
        setup_dirs()
        return "### สถานะ: ล้างข้อมูลเรียบร้อย พร้อมรันใหม่", None, None, None

    pause_btn.click(toggle_pause, outputs=[pause_btn])
    cancel_btn.click(request_cancel, outputs=[status_msg])
    restart_btn.click(restart_all, outputs=[status_msg, prev_before, prev_after, download_ui])

    start_btn.click(
        fn=process_images,
        inputs=[files_state, format_out, upscale_size, model_choice],
        outputs=[status_msg, prev_before, prev_after, download_ui]
    )

setup_dirs()
demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
