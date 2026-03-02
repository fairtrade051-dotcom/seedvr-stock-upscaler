import gradio as gr
import os
import zipfile
import subprocess
import shutil
from PIL import Image

# 1. ฟังก์ชันจัดการความสะอาดของเครื่อง (ล้างขยะก่อนเริ่ม)
def setup_dirs(input_dir, output_dir):
    if os.path.exists(input_dir): shutil.rmtree(input_dir)
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

# 2. ฟังก์ชันหลัก (Core logic)
def run_upscale(input_files, format_out, zip_out, upscale_size, model_choice):
    temp_in = os.path.abspath("./temp_in")
    temp_out = os.path.abspath("./temp_out")
    setup_dirs(temp_in, temp_out)

    if not input_files:
        raise gr.Error("กรุณาอัปโหลดไฟล์ก่อนครับ!")
        
    # จัดการไฟล์ขาเข้า (รองรับทั้งไฟล์เดี่ยวและ ZIP)
    for file_obj in input_files:
        file_path = file_obj.name
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_in)
        else:
            # เก็บชื่อไฟล์เดิมไว้เป๊ะๆ
            shutil.copy(file_path, os.path.join(temp_in, os.path.basename(file_path)))

    # ตั้งค่าความละเอียดและโมเดล
    res_map = {"2K (1440p)": 1440, "4K (2160p)": 2160, "6K (3240p)": 3240, "8K (4320p)": 4320}
    res_val = res_map.get(upscale_size, 1440)
    
    model_map = {
        "3B FP8 (สมดุล/ค่าเริ่มต้น)": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "3B GGUF Q4 (ประหยัด VRAM ขั้นสุด)": "seedvr2_ema_3b-Q4_K_M.gguf",
        "7B GGUF Q4 (สวยและประหยัด VRAM)": "seedvr2_ema_7b-Q4_K_M.gguf",
        "7B FP8 (ภาพสวยสุด/กิน VRAM โหด)": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    }
    selected_model_file = model_map[model_choice]

    print(f"🚀 เริ่มประมวลผล: {res_val}p | โมเดล: {selected_model_file}")
    
    # สั่งรัน AI ผ่าน CLI (เรียกใช้ inference_cli.py)
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
        print("Error Log:", process.stderr)
        raise gr.Error(f"AI รันล้มเหลว: {process.stderr[:150]}")

    # ดึงไฟล์ที่รันเสร็จแล้วออกมา
    out_files = sorted([f for f in os.listdir(temp_out) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not out_files:
        raise gr.Error("ไม่พบไฟล์ผลลัพธ์!")

    # แปลงไฟล์เป็น JPG Quality 100% (Adobe Stock Standard)
    if format_out == 'jpg':
        print("🔄 กำลังแปลงไฟล์เป็น JPG (Quality 100%)...")
        for img in out_files:
            if img.lower().endswith('.png'):
                img_path = os.path.join(temp_out, img)
                new_path = os.path.join(temp_out, os.path.splitext(img)[0] + ".jpg")
                with Image.open(img_path) as im:
                    rgb_im = im.convert('RGB')
                    rgb_im.save(new_path, format="JPEG", quality=100, subsampling=0)
                os.remove(img_path)
        out_files = sorted([f for f in os.listdir(temp_out) if f.lower().endswith('.jpg')])

    # เตรียมพรีวิว (รูปแรกที่ทำเสร็จ) และไฟล์สำหรับดาวน์โหลด
    preview_img = os.path.join(temp_out, out_files[0]) if out_files else None

    if zip_out:
        zip_name = "output_result"
        shutil.make_archive(zip_name, 'zip', temp_out)
        return preview_img, f"{zip_name}.zip"
    else:
        return preview_img, preview_img

# 3. ส่วนหน้าตาเว็บ (UI Layout)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💎 SeedVR2 Pro Stock Upscaler")
    gr.Markdown("รองรับ Batch | ชื่อเดิม | JPG 100% | พร้อมส่ง Adobe Stock")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="ลากรูป/ZIP ใส่ตรงนี้", file_count="multiple")
            model_choice = gr.Dropdown(choices=list(model_map.keys()) if 'model_map' in locals() else ["3B FP8 (สมดุล/ค่าเริ่มต้น)", "3B GGUF Q4", "7B GGUF Q4", "7B FP8"], label="🧠 เลือกโมเดล", value="3B FP8 (สมดุล/ค่าเริ่มต้น)")
            upscale_size = gr.Radio(["2K (1440p)", "4K (2160p)", "6K (3240p)", "8K (4320p)"], label="ความละเอียด", value="4K (2160p)")
            format_out = gr.Radio(["png", "jpg"], label="ไฟล์ที่ต้องการ", value="jpg")
            zip_out = gr.Checkbox(label="มัดรวมเป็น ZIP", value=True)
            submit_btn = gr.Button("🚀 เริ่ม Upscale", variant="primary")
            
        with gr.Column(scale=1):
            img_preview = gr.Image(label="ตัวอย่างภาพผลลัพธ์ (รูปแรก)", type="filepath")
            file_out = gr.File(label="ดาวน์โหลดไฟล์ที่นี่ ⬇️")

    submit_btn.click(
        fn=run_upscale,
        inputs=[file_in, format_out, zip_out, upscale_size, model_choice],
        outputs=[img_preview, file_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", port=7860, share=True)
