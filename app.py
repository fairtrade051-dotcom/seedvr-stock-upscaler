import gradio as gr
import os
import zipfile
import subprocess
import shutil
from PIL import Image

# กำหนด Path ให้รันบน RunPod ได้ชัวร์ๆ
WORKSPACE_DIR = "/workspace/ComfyUI-SeedVR2_VideoUpscaler"

def setup_dirs(input_dir, output_dir):
    # ปิดการลบโฟลเดอร์ทิ้งเพื่อความปลอดภัย (ถ้าอยากลบเองค่อยสั่งใน Terminal)
    # หรือถ้าจะลบ ต้องมั่นใจว่าย้ายไฟล์เก่าออกไปแล้ว
    if os.path.exists(input_dir): shutil.rmtree(input_dir)
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

def run_upscale(input_files, format_out, zip_out, upscale_size, model_choice):
    temp_in = os.path.join(WORKSPACE_DIR, "temp_in")
    temp_out = os.path.join(WORKSPACE_DIR, "temp_out")
    setup_dirs(temp_in, temp_out)

    if not input_files:
        raise gr.Error("กรุณาอัปโหลดไฟล์ก่อน!")
        
    # จัดการไฟล์อัปโหลด
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

    # เลือกโมเดล
    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model_file = model_map[model_choice]

    print(f"🚀 เริ่มรันที่ {res_val}p | โมเดล: {selected_model_file}")
    
    # คำสั่งรัน AI
    cmd = (
        f"python inference_cli.py "
        f"--output {temp_out} --resolution {res_val} "
        f"--dit_model {selected_model_file} "
        f"--vae_encode_tiled --vae_decode_tiled "
        f"--vae_encode_tile_size 256 --vae_decode_tile_size 256 " 
        f"--dit_offload_device cpu "
        f"{temp_in}"
    )
    
    # รัน process
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR)
    print(process.stdout)
    
    if process.returncode != 0:
        error_msg = process.stderr.strip() or process.stdout.strip()
        print(f"🔥 ERROR LOG:\n{error_msg}")
        raise gr.Error(f"AI ทำงานพัง! สาเหตุ: {error_msg[-1000:]}")

    # ดึงรายชื่อไฟล์หลัง AI ทำงานเสร็จ
    out_files = sorted(os.listdir(temp_out))
    
    # --- ส่วนแปลงไฟล์เป็น JPG (ปรับปรุงใหม่) ---
    if format_out == 'jpg':
        print("🔄 กำลังแปลงไฟล์เป็น JPG Quality 100 (No Subsampling)...")
        for img in out_files:
            if img.lower().endswith('.png'):
                img_path = os.path.join(temp_out, img)
                new_name = os.path.splitext(img)[0] + ".jpg"
                new_path = os.path.join(temp_out, new_name)
                
                with Image.open(img_path) as im:
                    rgb_im = im.convert('RGB')
                    # บังคับคุณภาพสูงสุด และปิดการบีบอัดสี (subsampling=0)
                    rgb_im.save(new_path, "JPEG", quality=100, subsampling=0)
                
                os.remove(img_path) # ลบ PNG ทิ้ง
        
        # อัปเดตรายชื่อไฟล์ใหม่หลังจากแปลงแล้ว
        out_files = sorted(os.listdir(temp_out))

    if not out_files:
        raise gr.Error("รันเสร็จแต่ไม่พบไฟล์ผลลัพธ์!")

    # พรีวิวภาพแรก
    in_files = sorted(os.listdir(temp_in))
    preview_in = os.path.join(temp_in, in_files[0]) if in_files else None
    preview_out = os.path.join(temp_out, out_files[0]) if out_files else None

    # จัดการไฟล์ดาวน์โหลด
    if zip_out:
        zip_path = os.path.join(WORKSPACE_DIR, "output_result.zip")
        if os.path.exists(zip_path): os.remove(zip_path)
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_out)
        return preview_in, preview_out, zip_path
    else:
        return preview_in, preview_out, os.path.join(temp_out, out_files[0])

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
                label="🧠 เลือกโมเดล", 
                value="3B FP8 (สมดุล/ค่าเริ่มต้น)"
            )
            upscale_size = gr.Radio(
                ["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], 
                label="ความละเอียดเป้าหมาย", 
                value="2K (1440p)"
            )
            format_out = gr.Radio(["png", "jpg"], label="Format ผลลัพธ์", value="jpg")
            zip_out = gr.Checkbox(label="ดาวน์โหลดกลับเป็นไฟล์ ZIP", value=True)
            submit_btn = gr.Button("เริ่มประมวลผล", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Row():
                preview_in_ui = gr.Image(label="ภาพก่อนอัป (Before)", type="filepath")
                preview_out_ui = gr.Image(label="ภาพหลังอัป (After)", type="filepath")
            file_out = gr.File(label="ไฟล์ผลลัพธ์พร้อมดาวน์โหลด ⬇️")

    submit_btn.click(
        fn=run_upscale,
        inputs=[file_in, format_out, zip_out, upscale_size, model_choice],
        outputs=[preview_in_ui, preview_out_ui, file_out]
    )

# เพิ่ม .queue() เพื่อป้องกันหน้าเว็บเด้งตอนรันนานๆ
demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
