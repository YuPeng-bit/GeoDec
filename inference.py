import argparse
import os
import sys
import torch
import numpy as np
import gc
from PIL import Image, ImageDraw, ImageFont
from rembg import remove  # [新增] 自动抠图库

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_controlnet_sd import MVAdapterI2MVControlNetSDPipeline
except ImportError:
    print("Error: Could not import pipelines.")
    sys.exit(1)

from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils.mesh_utils import get_orthogonal_camera
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from diffusers import DDPMScheduler, ControlNetModel

# ==========================================
# 1. 核心工具函数
# ==========================================
def force_cast_pipeline(pipe, device, dtype):
    """强制模型组件对齐设备和精度"""
    pipe.to(device)
    if hasattr(pipe, "unet") and pipe.unet: pipe.unet.to(device=device, dtype=dtype)
    if hasattr(pipe, "vae") and pipe.vae: pipe.vae.to(device=device, dtype=dtype)
    if hasattr(pipe, "text_encoder") and pipe.text_encoder: pipe.text_encoder.to(device=device, dtype=dtype)
    if hasattr(pipe, "cond_encoder") and pipe.cond_encoder: pipe.cond_encoder.to(device=device, dtype=dtype)
    if hasattr(pipe, "controlnet") and pipe.controlnet: pipe.controlnet.to(device=device, dtype=dtype)
    return pipe

def smart_load_weights(model, state_dict, model_name="Model"):
    """智能权重加载"""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    prefixes = ["system.unet.", "system.cond_encoder.", "module.", "unet.", "cond_encoder."]
    
    new_state_dict = {}
    matched = 0
    for k in model_keys:
        if k in ckpt_keys:
            new_state_dict[k] = state_dict[k]
            matched += 1
            continue
        for p in prefixes:
            if p + k in ckpt_keys:
                new_state_dict[k] = state_dict[p+k]
                matched += 1
                break
    
    match_rate = matched / len(model_keys) if len(model_keys) > 0 else 0
    print(f"[{model_name}] Weights Match Rate: {match_rate:.2%}")
    model.load_state_dict(new_state_dict, strict=False)

def process_custom_image(image_path, size=512, remove_bg=True):
    """
    全自动预处理：读取 -> (可选)抠图 -> 居中 -> 灰底 -> Resize
    """
    print(f"Processing input image: {image_path}")
    raw_img = Image.open(image_path).convert("RGBA")
    
    # 1. 自动抠图
    if remove_bg:
        print("  - Running background removal...")
        raw_img = remove(raw_img) # rembg magic
    
    # 2. 裁剪透明边缘并居中 (Standard Preprocess)
    image = raw_img
    arr = np.array(image)
    alpha = arr[..., 3] > 0
    y, x = np.where(alpha)
    
    if len(y) == 0 or len(x) == 0:
        print("  - Warning: Empty image after rembg!")
        final_arr = np.zeros((size, size, 4), dtype=np.uint8)
        final_arr[:] = [127, 127, 127, 255]
    else:
        y0, y1 = y.min(), y.max() + 1
        x0, x1 = x.min(), x.max() + 1
        obj_crop = arr[y0:y1, x0:x1]
        h_obj, w_obj = obj_crop.shape[:2]
        
        # 保持比例缩放，留一点边距 (0.9)
        scale = min((size * 0.9) / h_obj, (size * 0.9) / w_obj)
        new_h, new_w = int(h_obj * scale), int(w_obj * scale)
        
        # 使用 PIL Resize 质量更好
        crop_pil = Image.fromarray(obj_crop).resize((new_w, new_h), Image.Resampling.BILINEAR)
        crop_arr = np.array(crop_pil)
        
        start_h = (size - new_h) // 2
        start_w = (size - new_w) // 2
        
        final_arr = np.zeros((size, size, 4), dtype=np.uint8)
        final_arr[start_h:start_h+new_h, start_w:start_w+new_w] = crop_arr
    
    # 3. 合成灰底 (127)
    final_arr = final_arr.astype(np.float32) / 255.0
    # RGB * Alpha + BG * (1 - Alpha)
    final_arr = final_arr[:, :, :3] * final_arr[:, :, 3:4] + (1 - final_arr[:, :, 3:4]) * 0.5
    final_arr = (final_arr * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(final_arr)

# ==========================================
# 主逻辑
# ==========================================
def run_custom_inference(args):
    device = args.device
    dtype = torch.float16
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 处理输入图片
    ref_img = process_custom_image(args.image_path, remove_bg=not args.no_rembg)
    ref_img.save(os.path.join(args.output_dir, "processed_input.png"))
    
    # 2. 准备相机
    azimuths = [0, 45, 90, 180, 270, 315]
    cameras = get_orthogonal_camera(elevation_deg=[0]*6, distance=[1.8]*6, left=-0.55, right=0.55, bottom=-0.55, top=0.55, azimuth_deg=[x-90 for x in azimuths], device=device)
    plucker = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [1.1]*6, 512)
    control_images = ((plucker + 1.0) / 2.0).clamp(0, 1).to(device=device, dtype=dtype)

    # ---------------------------------------------------------
    # PHASE 1: Stage 1 Geometry
    # ---------------------------------------------------------
    print("\n[1/2] Running Stage 1 (Generating Normals)...")
    pipe_s1 = MVAdapterI2MVSDPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    pipe_s1.scheduler = ShiftSNRScheduler.from_scheduler(pipe_s1.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=DDPMScheduler)
    pipe_s1.init_custom_adapter(num_views=6)
    
    ckpt_s1 = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    sd_s1 = ckpt_s1["state_dict"] if "state_dict" in ckpt_s1 else ckpt_s1
    smart_load_weights(pipe_s1.unet, {k:v for k,v in sd_s1.items() if "unet" in k}, "Stage1-UNet")
    smart_load_weights(pipe_s1.cond_encoder, {k:v for k,v in sd_s1.items() if "cond_encoder" in k}, "Stage1-Encoder")
    pipe_s1 = force_cast_pipeline(pipe_s1, device, dtype)
    
    normals = pipe_s1(
        prompt="high quality", height=512, width=512, num_inference_steps=30, guidance_scale=3.0, num_images_per_prompt=6,
        control_image=control_images, reference_image=ref_img, reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast"
    ).images
    
    del pipe_s1; gc.collect(); torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PHASE 2: Stage 2 Ours (Fine-tuned)
    # ---------------------------------------------------------
    print("\n[2/2] Running Stage 2 (Ours Fine-tuned)...")
    cnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe_s2 = MVAdapterI2MVControlNetSDPipeline.from_pretrained(args.base_model, controlnet=cnet, torch_dtype=dtype, safety_checker=None)
    pipe_s2.scheduler = ShiftSNRScheduler.from_scheduler(pipe_s2.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=DDPMScheduler)
    pipe_s2.init_custom_adapter(num_views=6)
    
    print(f"Loading Fine-tuned weights from: {args.stage2_ckpt}")
    ckpt_s2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    sd_s2 = ckpt_s2["state_dict"] if "state_dict" in ckpt_s2 else ckpt_s2
    smart_load_weights(pipe_s2.unet, {k:v for k,v in sd_s2.items() if "unet" in k}, "Stage2-UNet")
    smart_load_weights(pipe_s2.cond_encoder, {k:v for k,v in sd_s2.items() if "cond_encoder" in k}, "Stage2-Encoder")
    pipe_s2 = force_cast_pipeline(pipe_s2, device, dtype)
    
    # 转换为 ControlNet 输入格式
    norm_inputs = [n.convert("RGB") for n in normals]
    
    ours_imgs = pipe_s2(
        prompt="high quality", height=512, width=512, num_inference_steps=30, guidance_scale=3.0, num_images_per_prompt=6,
        control_image=control_images, reference_image=ref_img, 
        normal_maps=norm_inputs, 
        controlnet_conditioning_scale=args.control_scale, # 推荐 0.7
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast"
    ).images

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    W, H = 512, 512
    # 创建一个 6列 x 1行 的画布
    combo = Image.new('RGB', (W * 6, H), (255, 255, 255))

    # Paste Images
    for i in range(6):
        # 修正坐标：
        # X轴: i * W (从0开始贴)
        # Y轴: 0 (贴在第一行)
        resized_img = ours_imgs[i].resize((W, H))
        combo.paste(resized_img, (i * W, 0))
    
    save_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.output_dir, f"{save_name}.jpg")
    combo.save(save_path)
    print(f"\n✅ Result saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to your custom image (jpg/png)")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    parser.add_argument("--control_scale", type=float, default=0.7, help="ControlNet strength for Ours")
    parser.add_argument("--no_rembg", action="store_true", help="Skip background removal (if image is already preprocessed)")
    
    # Defaults
    parser.add_argument("--base_model", type=str, default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--controlnet", type=str, default="thibaud/controlnet-sd21-normalbae-diffusers")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    run_custom_inference(args)