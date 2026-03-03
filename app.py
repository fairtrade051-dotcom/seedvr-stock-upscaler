import gradio as gr
import os
import zipfile
import subprocess
import shutil
import time
from PIL import Image

# --- การตั้งค่าเส้นทางแบบ Absolute Path ---
WORKSPACE_DIR = "/workspace/ComfyUI-SeedVR2_VideoUpscaler"
temp_in = os.path.abspath(os.path.join(WORKSPACE_DIR, "temp_in"))
temp_out = os.path.abspath(os.path.join(WORKSPACE_DIR, "temp_out"))

class GlobalState:
    is_running = False
    is_paused = False
    should_cancel = False

state = GlobalState()

def setup_dirs():
    """เตรียมพื้นที่ทำงานและล้างขยะ"""
    for d in [temp_in, temp_out]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def handle_upload(files):
    if not files: return "⚠️ ยังไม่มีไฟล์", [], None
    setup_dirs() # ล้างของเก่าทุกครั้งที่อัปโหลดใหม่
    for f in files:
        if f.name.lower().endswith('.zip'):
            with zipfile.ZipFile(f.name, 'r') as z: z.extractall(temp_in)
        else: shutil.copy(f.name, temp_in)
    all_files = sorted([f for f in os.listdir(temp_in) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    first_path = os.path.join(temp_in, all_files[0]) if all_files else None
    return f"📦 พร้อมรัน: {len(all_files)} รูป", all_files, first_path

def process_images(file_list, format_out, upscale_size, model_choice, steps, guidance, progress=gr.Progress()):
    state.is_running = True
    state.should_cancel = False
    if os.path.exists(temp_out): shutil.rmtree(temp_out)
    os.makedirs(temp_out, exist_ok=True)

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

        input_path = os.path.abspath(os.path.join(temp_in, filename))
        progress(i/len(file_list), desc=f"ทำรูปที่ {i+1}/{len(file_list)}")
        yield f"⏳ กำลังประมวลผลรูปที่ {i+1}/{len(file_list)}...", input_path, None, None

        # คำสั่งรัน AI
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
        
        # รันและดึง Error Log
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_DIR)

        # ตรวจสอบว่า AI พังไหม
        if result.returncode != 0:
            error_log = result.stderr if result.stderr else result.stdout
            print(f"❌ AI Error Log:\n{error_log}")
            yield f"❌ รูปที่ {i+1} พัง! สาเหตุ: {error_log[-200:]}", input_path, None, None
            continue # ข้ามไปรูปถัดไป

        # ค้นหาไฟล์ผลลัพธ์แบบยืดหยุ่น
        new_files = sorted([f for f in os.listdir(temp_out) if f.lower().endswith(('.png', '.jpg'))])
        base_name = os.path.splitext(filename)[0]
        matches = [f for f in new_files if base_name in f]

        if matches:
            out_path = os.path.join(temp_out, matches[-1])
            if format_out == 'jpg' and out_path.lower().endswith('.png'):
                jpg_path = os.path.splitext(out_path)[0] + ".jpg"
                with Image.open(out_path) as im: im.convert('RGB').save(jpg_path, "JPEG", quality=100, subsampling=0)
                os.remove(out_path)
                out_path = jpg_path
            yield f"✅ รูปที่ {i+1}/{len(file_list)} เสร็จแล้ว", input_path, out_path, None
        else:
            yield f"🔎 หารูปที่ {i+1} ไม่เจอใน Output", input_path, None, None

    # บีบ ZIP
    zip_path = os.path.join(WORKSPACE_DIR, "output_result.zip")
    if os.path.exists(zip_path): os.remove(zip_path)
    if os.listdir(temp_out):
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_out)
        yield "🎉 ประมวลผลเสร็จสิ้นทุกรูป!", None, None, zip_path
    else:
        yield "❌ ไม่พบไฟล์ผลลัพธ์ที่รันสำเร็จเลย", None, None, None

# --- UI Layout ---
with gr.Blocks(title="SeedVR2 Ultimate") as demo:
    gr.Markdown("# 🚀 SeedVR2 Auto Upscaler (Ultimate Version)")
    files_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            upload_ui = gr.File(label="อัปโหลดไฟล์", file_count="multiple")
            with gr.Accordion("⚙️ ตั้งค่าความละเอียดและความคม", open=True):
                model_choice = gr.Dropdown(choices=["3B FP8 (สมดุล/ค่าเริ่มต้น)", "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)", "7B GGUF Q4 (สวยและประหยัด VRAM)", "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)"], label="🧠 โมเดล", value="3B FP8 (สมดุล/ค่าเริ่มต้น)")
                upscale_size = gr.Radio(["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], label="🎯 ความละเอียด", value="2K (1440p)")
                steps_slider = gr.Slider(10, 50, 20, step=1, label="🔄 Steps")
                guidance_slider = gr.Slider(1.0, 15.0, 4.0, step=0.5, label="✨ Guidance (คมชัด)")
                format_out = gr.Radio(["png", "jpg"], label="🖼️ Format", value="jpg")

            with gr.Row():
                start_btn = gr.Button("▶️ เริ่ม", variant="primary")
                cancel_btn = gr.Button("🛑 ยกเลิก")
            
        with gr.Column(scale=2):
            status_msg = gr.Markdown("### 📋 สถานะ: รอไฟล์...")
            with gr.Row():
                prev_before = gr.Image(label="ก่อน (Before)", type="filepath")
                prev_after = gr.Image(label="หลัง (After)", type="filepath")
            download_ui = gr.File(label="⬇️ ดาวน์โหลด ZIP")

    upload_ui.change(handle_upload, inputs=[upload_ui], outputs=[status_msg, files_state, prev_before])
    start_btn.click(process_images, inputs=[files_state, format_out, upscale_size, model_choice, steps_slider, guidance_slider], outputs=[status_msg, prev_before, prev_after, download_ui])
    cancel_btn.click(lambda: setattr(state, 'should_cancel', True))

if __name__ == "__main__":
    setup_dirs()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, allowed_paths=["/workspace"], css=".gradio-container {background-color: #0b111b; color: white;}")
