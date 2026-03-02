import gradio as gr
import os
import zipfile
import subprocess
import shutil
from PIL import Image

# 1. ฟังก์ชันจัดการโฟลเดอร์
def setup_dirs(input_dir, output_dir):
    if os.path.exists(input_dir): shutil.rmtree(input_dir)
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

# 2. ฟังก์ชันหลัก
def run_upscale(input_files, format_out, zip_out, upscale_size, model_choice):
    temp_in = os.path.abspath("./temp_in")
    temp_out = os.path.abspath("./temp_out")
    setup_dirs(temp_in, temp_out)

    if not input_files:
        raise gr.Error("ใส่รูปก่อนพี่!")
        
    for file_obj in input_files:
        shutil.copy(file_obj.name, os.path.join(temp_in, os.path.basename(file_obj.name)))

    res_map = {"2K (1440p)": 1440, "4K (2160p)": 2160, "6K (3240p)": 3240, "8K (4320p)": 4320}
    res_val = res_map.get(upscale_size, 1440)
    
    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model_file = model_map.get(model_choice, "seedvr2_ema_3b_fp8_e4m3fn.safetensors")

    cmd = [
        "python", "inference_cli.py",
        "--output", temp_out,
        "--resolution", str(res_val),
        "--dit_model", selected_model_file,
        "--vae_encode_tiled", "--vae_decode_tiled",
        "--vae_encode_tile_size", "256", "--vae_decode_tile_size", "256",
        "--dit_offload_device", "cpu",
        temp_in
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        raise gr.Error(f"AI พัง: {process.stderr[:150]}")

    out_files = sorted([f for f in os.listdir(temp_out) if f.lower().endswith(('.png', '.jpg'))])
    if not out_files:
        raise gr.Error("ไม่มีไฟล์ออกมา เช็คโมเดลดูหน่อย")

    if format_out == 'jpg':
        for img in out_files:
            if img.lower().endswith('.png'):
                img_path = os.path.join(temp_out, img)
                new_path = os.path.join(temp_out, os.path.splitext(img)[0] + ".jpg")
                with Image.open(img_path) as im:
                    im.convert('RGB').save(new_path, format="JPEG", quality=100, subsampling=0)
                os.remove(img_path)
        out_files = sorted([f for f in os.listdir(temp_out) if f.lower().endswith('.jpg')])

    preview = os.path.join(temp_out, out_files[0]) if out_files else None

    if zip_out:
        zip_path = os.path.abspath("result_stock")
        shutil.make_archive(zip_path, 'zip', temp_out)
        return preview, zip_path + ".zip"
    return preview, preview

# 3. ส่วน UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💎 SeedVR2 Pro Stock Upscaler")
    with gr.Row():
        with gr.Column():
            file_in = gr.File(label="Upload Images", file_count="multiple")
            model_choice = gr.Dropdown(
                choices=["3B FP8 (สมดุล/ค่าเริ่มต้น)", "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)", "7B GGUF Q4 (สวยและประหยัด VRAM)", "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)"], 
                label="Model", 
                value="3B FP8 (สมดุล/ค่าเริ่มต้น)"
            )
            upscale_size = gr.Radio(["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], label="Resolution", value="4K (2160p)")
            format_out = gr.Radio(["png", "jpg"], label="Format", value="jpg")
            zip_out = gr.Checkbox(label="Zip Download", value=True)
            submit_btn = gr.Button("🚀 START", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="Preview", type="filepath")
            file_download = gr.File(label="Download")

    submit_btn.click(run_upscale, [file_in, format_out, zip_out, upscale_size, model_choice], [img_out, file_download])

# แก้ไขบรรทัดสุดท้ายจาก port เป็น server_port
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
