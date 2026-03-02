import gradio as gr
import os
import zipfile
import subprocess
import shutil
from PIL import Image
from gradio_image_slider import ImageSlider

# ใช้ /workspace เป็นหลักสำหรับ RunPod
WORKSPACE_DIR = "/workspace/ComfyUI-SeedVR2_VideoUpscaler"

def setup_dirs(input_dir, output_dir):
    if os.path.exists(input_dir): shutil.rmtree(input_dir)
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

def run_upscale(input_files, format_out, zip_out, upscale_size, model_choice):
    temp_in = os.path.join(WORKSPACE_DIR, "temp_in")
    temp_out = os.path.join(WORKSPACE_DIR, "temp_out")
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

    # คำนวณความละเอียดเป้าหมาย
    if "2K" in upscale_size: res_val = 1440
    elif "4K" in upscale_size: res_val = 2160
    elif "6K" in upscale_size: res_val = 3240
    else: res_val = 4320

    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model_file = model_map[model_choice]

    print(f"🚀 เริ่มรันที่ {res_val}p | โมเดล: {selected_model_file}")
    
    cmd = (
        f"python inference_cli.py "
        f"--output {temp_out} --resolution {res_val} "
        f"--dit_model {selected_model_file} "
        f"--vae_encode_tiled --vae_decode_tiled "
        f"--vae_encode_tile_size 256 --vae_decode_tile_size 256 " 
        f"--dit_offload_device cpu "
        f"{temp_in}"
    )
    
    # รันโดยให้ working directory อยู่ในโฟลเดอร์โปรเจกต์
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR)
    print(process.stdout)
    
    if process.returncode != 0:
        print(process.stderr)
        raise gr.Error("รันล้มเหลว! การ์ดจออาจรับไม่ไหว ลองลดขนาดภาพหรือเปลี่ยนโมเดลดูครับ")

    out_files = sorted(os.listdir(temp_out))
    if not os.path.exists(temp_out) or len(out_files) == 0:
        raise gr.Error("ไม่พบไฟล์ผลลัพธ์! AI ทำงานไม่สำเร็จ")

    # --- แปลงไฟล์เป็น JPG ด้วย Quality 100 ---
    if format_out == 'jpg':
        print("🔄 กำลังแปลงไฟล์เป็น JPG Quality 100...")
        for img in list(out_files):
            if img.endswith('.png'):
                img_path = os.path.join(temp_out, img)
                new_path = os.path.join(temp_out, img.replace('.png', '.jpg'))
                with Image.open(img_path) as im:
                    rgb_im = im.convert('RGB')
                    rgb_im.save(new_path, quality=100) # คงคุณภาพสูงสุด
                os.remove(img_path)
        out_files = sorted(os.listdir(temp_out))

    # เตรียมภาพพรีวิวสำหรับ Slider
    in_files = sorted(os.listdir(temp_in))
    preview_in = os.path.join(temp_in, in_files[0]) if in_files else None
    preview_out = os.path.join(temp_out, out_files[0]) if out_files else None

    # จัดการไฟล์ดาวน์โหลด
    if zip_out:
        zip_path = os.path.join(WORKSPACE_DIR, "output_result.zip")
        if os.path.exists(zip_path): os.remove(zip_path)
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_out)
        return (preview_in, preview_out), zip_path
    else:
        return (preview_in, preview_out), os.path.join(temp_out, out_files[0])


# --- UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🚀 SeedVR2 Auto Upscaler (RunPod Edition)")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="อัปโหลดภาพ (ลากคลุมได้หลายรูป หรือโยนไฟล์ ZIP)", file_count="multiple")
            model_choice = gr.Dropdown(
                choices=[
                    "3B FP8 (สมดุล/ค่าเริ่มต้น)", 
                    "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)", 
                    "7B GGUF Q4 (สวยและประหยัด VRAM)",
                    "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)"
                ], 
                label="🧠 เลือกโมเดล (Model Selection)", 
                value="3B FP8 (สมดุล/ค่าเริ่มต้น)"
            )
            upscale_size = gr.Radio(
                ["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], 
                label="ความละเอียดเป้าหมาย", 
                value="2K (1440p)"
            )
            format_out = gr.Radio(["png", "jpg"], label="Format ผลลัพธ์", value="jpg") # ตั้งค่าเริ่มต้นเป็น jpg ให้เลย
            zip_out = gr.Checkbox(label="ดาวน์โหลดกลับเป็นไฟล์ ZIP", value=True)
            submit_btn = gr.Button("เริ่มประมวลผล", variant="primary")
            
        with gr.Column(scale=2):
            slider_preview = ImageSlider(label="เปรียบเทียบ ก่อน/หลัง (รูปแรก)")
            file_out = gr.File(label="ไฟล์ผลลัพธ์พร้อมดาวน์โหลด ⬇️")

    submit_btn.click(
        fn=run_upscale,
        inputs=[file_in, format_out, zip_out, upscale_size, model_choice],
        outputs=[slider_preview, file_out]
    )

# กำหนด Port เป็น 7860 เพื่อให้เชื่อมกับระบบ Proxy ของ RunPod ได้ง่าย
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
