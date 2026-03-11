import argparse
import os
import sys
import torch
import numpy as np
import gc
from PIL import Image, ImageDraw, ImageFont

import mvadapter
print(f"!!! ACTUAL CODE PATH: {mvadapter.__file__}")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === 1. 稳健的 torch.load 修复 (Fix weights_only error) ===
# 保存原始的 torch.load
_original_torch_load = torch.load

def aggressive_safe_load(*args, **kwargs):
    # 强制覆盖 weights_only 参数
    if 'weights_only' in kwargs:
        kwargs['weights_only'] = False
    else:
        kwargs['weights_only'] = False
    # 调用原始函数
    return _original_torch_load(*args, **kwargs)

# 覆盖回去
torch.load = aggressive_safe_load
print("[INFO] Applied robust torch.load patch (forcing weights_only=False).")
# ==========================================================

from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils.mesh_utils import get_orthogonal_camera
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from diffusers import DDPMScheduler

# ==========================================
# 2. 核心函数
# ==========================================
def smart_load_weights(model, state_dict, model_name="Model"):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    # 增加更多前缀容错
    prefixes = ["system.unet.", "system.cond_encoder.", "module.", "unet.", "cond_encoder.", "pipeline.unet.", "pipeline.cond_encoder."]
    
    new_state_dict = {}
    matched = 0
    for k in model_keys:
        if k in ckpt_keys:
            new_state_dict[k] = state_dict[k]
            matched += 1
            continue
        
        found = False
        for p in prefixes:
            if p + k in ckpt_keys:
                new_state_dict[k] = state_dict[p+k]
                matched += 1
                found = True
                break
        
        if not found:
             # 反向查找: ckpt key 包含 model key
             for ck in ckpt_keys:
                 for p in prefixes:
                     if ck == p + k:
                         new_state_dict[k] = state_dict[ck]
                         matched += 1
                         found = True
                         break
                 if found: break

    match_rate = matched / len(model_keys) if len(model_keys) > 0 else 0
    print(f"[{model_name}] Match Rate: {match_rate:.2%}")
    model.load_state_dict(new_state_dict, strict=False)

def preprocess_image(image: Image.Image, height: int = 512, width: int = 512):
    if image.mode != "RGBA": image = image.convert("RGBA")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.array(image)
    alpha = arr[..., 3] > 0
    
    # 简单的居中裁剪逻辑
    if alpha.sum() == 0:
        final_arr = np.zeros((height, width, 4), dtype=np.uint8)
        final_arr[:] = [127, 127, 127, 255]
    else:
        y_indices, x_indices = np.where(alpha)
        y0, y1 = y_indices.min(), y_indices.max() + 1
        x0, x1 = x_indices.min(), x_indices.max() + 1
        obj = arr[y0:y1, x0:x1]
        h_obj, w_obj = obj.shape[:2]
        
        final_arr = np.zeros((height, width, 4), dtype=np.uint8)
        # 居中
        sy, sx = (height - h_obj)//2, (width - w_obj)//2
        # 边界保护
        end_y = min(sy + h_obj, height)
        end_x = min(sx + w_obj, width)
        real_h = end_y - sy
        real_w = end_x - sx
        
        final_arr[sy:end_y, sx:end_x] = obj[:real_h, :real_w]
        
    final_arr = final_arr.astype(np.float32) / 255.0
    # 灰底混合
    final_arr = final_arr[:, :, :3] * final_arr[:, :, 3:4] + 0.5 * (1 - final_arr[:, :, 3:4])
    return Image.fromarray((final_arr * 255).astype(np.uint8))

def load_stage1_pipeline(base_model, ckpt_path, device, dtype):
    print(f"\n--- Loading Stage 1 (Geometry) from {ckpt_path} ---")
    pipe = MVAdapterI2MVSDPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=DDPMScheduler,
    )
    pipe.init_custom_adapter(num_views=6)
    
    # 使用被我们 patch 过的 torch.load
    print("Loading checkpoint file...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    
    # 分离权重
    # 注意：Lightning 可能会把参数存为 system.pipeline.unet 或 system.unet
    unet_sd = {k:v for k,v in sd.items() if "unet" in k}
    enc_sd = {k:v for k,v in sd.items() if "cond_encoder" in k}
    
    smart_load_weights(pipe.unet, unet_sd, "UNet")
    smart_load_weights(pipe.cond_encoder, enc_sd, "Encoder")
    
    pipe.to(device=device, dtype=dtype)
    
    # 显式转换
    if hasattr(pipe, "cond_encoder"):
        pipe.cond_encoder.to(device=device, dtype=dtype)
        
    return pipe

# ==========================================
# 3. 主流程
# ==========================================
def run_stage1(args):
    device = "cuda"
    dtype = torch.float16
    
    # 1. 准备相机 (距离1.8，如果之前校准过请修改)
    azimuths = [0, 45, 90, 180, 270, 315]
    cameras = get_orthogonal_camera(
        elevation_deg=[0]*6, distance=[1.8]*6, 
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[x - 90 for x in azimuths], device=device
    )
    plucker = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [1.1]*6, 512)
    control_images = ((plucker + 1.0) / 2.0).clamp(0, 1).to(device=device, dtype=dtype)
    
    # 2. 加载模型
    pipe = load_stage1_pipeline(args.base_model, args.ckpt, device, dtype)
    
    # 3. 推理
    raw_img = Image.open(args.image_path)
    proc_img = preprocess_image(raw_img)
    
    print("\nRunning Inference...")
    # 设置 Seed
    generator = torch.Generator(device).manual_seed(42)
    
    normals = pipe(
        prompt="high quality", height=512, width=512, 
        num_inference_steps=30, guidance_scale=3.0, num_images_per_prompt=6,
        control_image=control_images, reference_image=proc_img, reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        generator=generator
    ).images
    
    # 4. 拼图展示
    W, H = 256, 256
    combo = Image.new('RGB', (W * 7, H), (255, 255, 255))
    
    # 左边放原图
    combo.paste(proc_img.resize((W, H)), (0, 0))
    
    # 右边放 6 张法线
    for i in range(6):
        combo.paste(normals[i].resize((W, H)), ((i+1)*W, 0))
        
    draw = ImageDraw.Draw(combo)
    fnt = ImageFont.load_default()
    draw.text((10, 10), "Input", fill="white", font=fnt)
    angles = ["0 (Front)", "45", "90 (Side)", "180 (Back)", "270", "315"]
    for i, ang in enumerate(angles):
        draw.text(((i+1)*W + 10, 10), ang, fill="white", font=fnt)
        
    save_path = os.path.join(args.output_dir, f"stage1_res_{os.path.basename(args.image_path)}")
    os.makedirs(args.output_dir, exist_ok=True)
    combo.save(save_path)
    print(f"✅ Saved result to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Input image")
    parser.add_argument("--ckpt", type=str, required=True, help="Stage 1 checkpoint path")
    parser.add_argument("--output_dir", type=str, default="stage1_results")
    parser.add_argument("--base_model", type=str, default="Manojb/stable-diffusion-2-1-base")
    args = parser.parse_args()
    
    run_stage1(args)