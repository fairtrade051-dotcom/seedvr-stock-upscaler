import gradio as gr
import os
import zipfile
import subprocess
import shutil
import time
from PIL import Image

# --- การตั้งค่าเส้นทาง ---
WORKSPACE_DIR = "/workspace/ComfyUI-SeedVR2_VideoUpscaler"
temp_in = os.path.join(WORKSPACE_DIR, "temp_in")
temp_out = os.path.join(WORKSPACE_DIR, "temp_out")
MODEL_DIR = os.path.join(WORKSPACE_DIR, "models") # แก้ตามโครงสร้างจริงของ SeedVR2

class GlobalState:
    is_running = False
    should_cancel = False

state = GlobalState()

def setup_dirs():
    """ล้างขยะเตรียมรัน"""
    for d in [temp_in, temp_out]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def handle_upload(files):
    if not files: return "⚠️ ยังไม่มีไฟล์", [], None
    setup_dirs()
    for f in files:
        if f.name.lower().endswith('.zip'):
            with zipfile.ZipFile(f.name, 'r') as z: z.extractall(temp_in)
        else: shutil.copy(f.name, temp_in)
    
    all_files = sorted([f for f in os.listdir(temp_in) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    first_path = os.path.join(temp_in, all_files[0]) if all_files else None
    return f"📦 รับทราบ: {len(all_files)} รูป", all_files, first_path

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

    full_log = ""

    for i, filename in enumerate(file_list):
        if state.should_cancel: break
        
        input_path = os.path.join(temp_in, filename)
        progress(i/len(file_list), desc=f"ทำรูปที่ {i+1}/{len(file_list)}")
        yield f"⏳ กำลังรันรูป {i+1}...", input_path, None, None, "เริ่มการทำงาน..."

        # บันทึกรายชื่อไฟล์ที่มีอยู่ก่อนรันรูปนี้
        files_before = set(os.listdir(temp_out))

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
        
        # รัน AI และเก็บ Log
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_DIR)
        current_log = f"--- Image {i+1} ({filename}) ---\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n"
        full_log += current_log
        print(current_log) # พิมพ์ลง Terminal RunPod ด้วย

        # เช็คไฟล์ใหม่ที่โผล่มา
        files_after = set(os.listdir(temp_out))
        new_files = list(files_after - files_before)
        # กรองเอาเฉพาะไฟล์ภาพ
        new_images = [f for f in new_files if f.lower().endswith(('.png', '.jpg'))]

        if new_images:
            # ใช้ไฟล์ล่าสุดที่โผล่มา
            out_path = os.path.join(temp_out, sorted(new_images)[-1])
            
            if format_out == 'jpg' and out_path.lower().endswith('.png'):
                jpg_path = os.path.splitext(out_path)[0] + ".jpg"
                with Image.open(out_path) as im:
                    im.convert('RGB').save(jpg_path, "JPEG", quality=100, subsampling=0)
                os.remove(out_path)
                out_path = jpg_path
            
            yield f"✅ รูปที่ {i+1} สำเร็จ", input_path, out_path, None, full_log
        else:
            yield f"⚠️ รูปที่ {i+1} ไม่มีไฟล์ออกมา (เช็ค Log ด้านล่าง)", input_path, None, None, full_log

    # บีบ ZIP
    zip_path = os.path.join(WORKSPACE_DIR, "output_result.zip")
    if os.path.exists(zip_path): os.remove(zip_path)
    if os.listdir(temp_out):
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_out)
        yield "🎉 เสร็จสิ้น!", None, None, zip_path, full_log
    else:
        yield "❌ ไม่พบไฟล์ผลลัพธ์ใดๆ เลย", None, None, None, full_log

# --- UI ---
with gr.Blocks(title="SeedVR2 Debug Mode") as demo:
    gr.Markdown("# 🚀 SeedVR2 Auto Upscaler (Debug & Fix Mode)")
    files_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            upload_ui = gr.File(label="อัปโหลดไฟล์", file_count="multiple")
            with gr.Accordion("⚙️ ตั้งค่า", open=True):
                model_choice = gr.Dropdown(choices=["3B FP8 (สมดุล/ค่าเริ่มต้น)", "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)", "7B GGUF Q4 (สวยและประหยัด VRAM)", "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)"], label="โมเดล", value="3B FP8 (สมดุล/ค่าเริ่มต้น)")
                upscale_size = gr.Radio(["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], label="ความละเอียด", value="2K (1440p)")
                steps_slider = gr.Slider(10, 50, 20, step=1, label="Steps")
                guidance_slider = gr.Slider(1.0, 15.0, 4.0, step=0.5, label="Guidance")
                format_out = gr.Radio(["png", "jpg"], label="Format", value="jpg")
            
            start_btn = gr.Button("▶️ เริ่ม", variant="primary")
            cancel_btn = gr.Button("🛑 ยกเลิก")
            
        with gr.Column(scale=2):
            status_msg = gr.Markdown("### 📋 สถานะ: รอไฟล์...")
            with gr.Row():
                prev_before = gr.Image(label="ก่อน", type="filepath")
                prev_after = gr.Image(label="หลัง", type="filepath")
            download_ui = gr.File(label="⬇️ ดาวน์โหลด ZIP")
            
    # เพิ่มช่อง Debug Log
    debug_log = gr.Textbox(label="🛠️ AI Debug Console (ถ้าพังให้อ่านตรงนี้)", lines=10, max_lines=20)

    upload_ui.change(handle_upload, inputs=[upload_ui], outputs=[status_msg, files_state, prev_before])
    start_btn.click(process_images, inputs=[files_state, format_out, upscale_size, model_choice, steps_slider, guidance_slider], outputs=[status_msg, prev_before, prev_after, download_ui, debug_log])
    cancel_btn.click(lambda: setattr(state, 'should_cancel', True))

if __name__ == "__main__":
    setup_dirs()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, allowed_paths=["/workspace"], css=".gradio-container {background-color: #0b111b; color: white;}")
