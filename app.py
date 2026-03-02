import gradio as gr  # <--- อันนี้ถูกต้อง
import os
import zipfile
import subprocess
import shutil
from PIL import Image

# ฟังก์ชันจัดการโฟลเดอร์ให้สะอาด
def setup_dirs(input_dir, output_dir):
    if os.path.exists(input_dir): shutil.rmtree(input_dir)
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(input_dir)
    os.makedirs(output_dir)

def run_upscale(input_files, format_out, zip_out, upscale_size, model_choice):
    temp_in = os.path.abspath("./temp_in")
    temp_out = os.path.abspath("./temp_out")
    setup_dirs(temp_in, temp_out)

    if not input_files:
        raise gr.Error("กรุณาอัปโหลดไฟล์ก่อนครับ!")
        
    for file_obj in input_files:
        file_path = file_obj.name
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_in)
        else:
            shutil.copy(file_path, temp_in)

    # 1. ปรับค่าความละเอียด
    res_map = {"2K (1440p)": 1440, "4K (2160p)": 2160, "6K (3240p)": 3240, "8K (4320p)": 4320}
    res_val = res_map.get(upscale_size, 1440)

    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model_file = model_map[model_choice]

    print(f"🚀 Processing: {res_val}p | Model: {selected_model_file}")
    
    # 2. คำสั่ง CLI (เพิ่ม --vae_encode_tiled เพื่อประหยัด VRAM)
    cmd = (
        f"python inference_cli.py "
        f"--output {temp_out} --resolution {res_val} "
        f"--dit_model {selected_model_file} "
        f"--vae_encode_tiled --vae_decode_tiled "
        f"--vae_encode_tile_size 256 --vae_decode_tile_size 256 " 
        f"--dit_offload_device cpu "
        f"{temp_in}"
    )
    
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(process.stderr)
        raise gr.Error("AI รันล้มเหลว! การ์ดจออาจไม่พอ ลองลดขนาดภาพดูครับ")

    out_files = sorted([f for f in os.listdir(temp_out) if f.endswith(('.png', '.jpg'))])
    
    # 3. แปลงไฟล์เป็น JPG (Quality 100% สำหรับ Adobe Stock)
    if format_out == 'jpg':
        print("🔄 Converting to JPG (Quality 100%)...")
        for img in out_files:
            if img.endswith('.png'):
                img_path = os.path.join(temp_out, img)
                new_path = os.path.join(temp_out, img.replace('.png', '.jpg'))
                with Image.open(img_path) as im:
                    rgb_im = im.convert('RGB')
                    rgb_im.save(new_path, format="JPEG", quality=100, subsampling=0) # subsampling=0 เพื่อความชัดสูงสุด
                os.remove(img_path)
        out_files = sorted([f for f in os.listdir(temp_out) if f.endswith('.jpg')])

    # 4. จัดการการส่งออกไฟล์
    if zip_out:
        zip_path = os.path.abspath("output_result.zip")
        if os.path.exists(zip_path): os.remove(zip_path)
        shutil.make_archive('output_result', 'zip', temp_out)
        return zip_path
    else:
        return os.path.join(temp_out, out_files[0]) if out_files else None

# --- หน้า UI (ถอด Slider ออกตามคำขอ) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 💎 SeedVR2 Pro Stock Upscaler")
    gr.Markdown("ส่ง Adobe Stock ได้มั่นใจด้วย JPG Quality 100%")
    
    with gr.Row():
        with gr.Column():
            file_in = gr.File(label="Upload Images/ZIP", file_count="multiple")
            model_choice = gr.Dropdown(
                choices=["3B FP8 (สมดุล/ค่าเริ่มต้น)", "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)", "7B GGUF Q4 (สวยและประหยัด VRAM)", "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)"], 
                label="🧠 Model", value="3B FP8 (สมดุล/ค่าเริ่มต้น)"
            )
            upscale_size = gr.Radio(["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], label="Resolution", value="4K (2160p)")
            format_out = gr.Radio(["png", "jpg"], label="Output Format", value="jpg")
            zip_out = gr.Checkbox(label="Pack as ZIP", value=True)
            submit_btn = gr.Button("🚀 Start Upscaling", variant="primary")
            
        with gr.Column():
            file_out = gr.File(label="Download Here ⬇️")

    submit_btn.click(
        fn=run_upscale,
        inputs=[file_in, format_out, zip_out, upscale_size, model_choice],
        outputs=[file_out]
    )

demo.launch(server_name="0.0.0.0", share=True)
