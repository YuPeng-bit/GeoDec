import argparse
import os
import sys
import torch
import numpy as np
import gc
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
from mvadapter.pipelines.pipeline_mvadapter_i2mv_controlnet_sd import MVAdapterI2MVControlNetSDPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils.mesh_utils import get_orthogonal_camera
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from diffusers import DDPMScheduler, ControlNetModel

# ==========================================
# 1. 智能权重加载函数
# ==========================================
def smart_load_weights(model, state_dict, model_name="Model"):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    prefixes_to_try = [
        "system.unet.", "system.cond_encoder.", "module.", "unet.", "cond_encoder."
    ]
    
    new_state_dict = {}
    matched_keys = 0
    
    for k in model_keys:
        if k in ckpt_keys:
            new_state_dict[k] = state_dict[k]
            matched_keys += 1
            continue
        found = False
        for p in prefixes_to_try:
            k_pre = p + k
            if k_pre in ckpt_keys:
                new_state_dict[k] = state_dict[k_pre]
                matched_keys += 1
                found = True
                break
        if not found:
            for ck in ckpt_keys:
                for p in prefixes_to_try:
                    if ck.startswith(p) and ck[len(p):] == k:
                        new_state_dict[k] = state_dict[ck]
                        matched_keys += 1
                        found = True
                        break
                if found: break

    match_rate = matched_keys / len(model_keys) if len(model_keys) > 0 else 0.0
    print(f"[{model_name}] Match Rate: {match_rate:.2%}")
    model.load_state_dict(new_state_dict, strict=False)

# ==========================================
# 2. 统一预处理
# ==========================================
def preprocess_image(image: Image.Image, height: int = 512, width: int = 512):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.array(image)
    alpha = arr[..., 3] > 0
    y, x = np.where(alpha)
    
    if len(y) == 0 or len(x) == 0:
        final_arr = np.zeros((height, width, 4), dtype=np.uint8)
        final_arr[:] = [127, 127, 127, 255]
    else:
        y0, y1 = y.min(), y.max() + 1
        x0, x1 = x.min(), x.max() + 1
        obj_crop = arr[y0:y1, x0:x1]
        h_obj, w_obj = obj_crop.shape[:2]
        
        start_h = (height - h_obj) // 2
        start_w = (width - w_obj) // 2
        
        final_arr = np.zeros((height, width, 4), dtype=np.uint8)
        end_h = min(start_h + h_obj, height)
        end_w = min(start_w + w_obj, width)
        final_arr[start_h:end_h, start_w:end_w] = obj_crop[:(end_h-start_h), :(end_w-start_w)]
    
    final_arr = final_arr.astype(np.float32) / 255.0
    final_arr = final_arr[:, :, :3] * final_arr[:, :, 3:4] + (1 - final_arr[:, :, 3:4]) * 0.5
    final_arr = (final_arr * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(final_arr)

# ==========================================
# 3. 加载 Stage 1
# ==========================================
def load_stage1_pipeline(base_model, ckpt_path, device, dtype):
    print(f"\n--- Loading Stage 1 (Geometry) ---")
    pipe = MVAdapterI2MVSDPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=DDPMScheduler,
    )
    pipe.init_custom_adapter(num_views=6)
    
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    smart_load_weights(pipe.unet, state_dict, "Stage1-UNet")
    smart_load_weights(pipe.cond_encoder, state_dict, "Stage1-CondEncoder")
    
    pipe.to(device=device, dtype=dtype)
    if hasattr(pipe, "cond_encoder"):
        pipe.cond_encoder.to(device=device, dtype=dtype)
        
    return pipe

# ==========================================
# 4. 加载 Stage 2 (修复 FP16)
# ==========================================
def load_stage2_pipeline(base_model, adapter_path, controlnet_path, device, dtype):
    print(f"\n--- Loading Stage 2 (Texture) ---")
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
    pipe = MVAdapterI2MVControlNetSDPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=DDPMScheduler,
    )
    pipe.init_custom_adapter(num_views=6)
    pipe.load_custom_adapter(adapter_path, weight_name="mvadapter_i2mv_sd21.safetensors")
    
    pipe.to(device=device, dtype=dtype)
    
    # [核心修复] 显式转换 cond_encoder 为 fp16
    if hasattr(pipe, "cond_encoder"):
        print("Stage 2: Explicitly casting cond_encoder to dtype...")
        pipe.cond_encoder.to(device=device, dtype=dtype)
        
    return pipe

# ==========================================
# 5. 主流程
# ==========================================
def run_joint_inference(args):
    device = args.device
    dtype = torch.float16
    
    azimuth_deg = [0, 45, 90, 180, 270, 315]
    cameras = get_orthogonal_camera(
        elevation_deg=[0]*6, distance=[1.8]*6,
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[x - 90 for x in azimuth_deg], device=device
    )
    plucker_embeds = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [1.1]*6, 512)
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1).to(dtype=dtype)
    
    # --- Step 1 ---
    pipe1 = load_stage1_pipeline(args.base_model, args.stage1_ckpt, device, dtype)
    raw_img = Image.open(args.image_path)
    proc_img = preprocess_image(raw_img, 512, 512)
    
    print("\nRunning Stage 1 (Normals)...")
    gen_kwargs = {"generator": torch.Generator(device).manual_seed(args.seed)} if args.seed != -1 else {}
    normals = pipe1(
        prompt="high quality", height=512, width=512,
        num_inference_steps=30, guidance_scale=3.0, num_images_per_prompt=6,
        control_image=control_images, reference_image=proc_img,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        **gen_kwargs
    ).images
    
    debug_dir = os.path.join(args.output_dir, "debug_stage1")
    os.makedirs(debug_dir, exist_ok=True)
    for i, n in enumerate(normals): n.save(os.path.join(debug_dir, f"normal_{i}.png"))
    
    del pipe1
    torch.cuda.empty_cache()
    gc.collect()
    
    # --- Step 2 ---
    pipe2 = load_stage2_pipeline(args.base_model, args.stage2_adapter, args.controlnet, device, dtype)
    proc_normals = [n.convert("RGB") for n in normals]
    
    print("\nRunning Stage 2 (RGB)...")
    rgbs = pipe2(
        prompt="high quality", height=512, width=512,
        num_inference_steps=30, guidance_scale=3.0, num_images_per_prompt=6,
        control_image=control_images, reference_image=proc_img,
        normal_maps=proc_normals, controlnet_conditioning_scale=0.6,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        **gen_kwargs
    ).images
    
    save_path = os.path.join(args.output_dir, f"joint_v4_{os.path.basename(args.image_path)}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    W, H = 256, 256
    combo = Image.new('RGB', (W * 7, H * 2), (255, 255, 255))
    combo.paste(proc_img.resize((W, H)), (0, 0))
    combo.paste(proc_img.resize((W, H)), (0, H))
    
    for i in range(6):
        combo.paste(proc_normals[i].resize((W, H)), ((i+1)*W, 0))
        combo.paste(rgbs[i].resize((W, H)), ((i+1)*W, H))
        
    combo.save(save_path)
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="joint_results_v4")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--stage2_adapter", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--controlnet", type=str, default="thibaud/controlnet-sd21-normalbae-diffusers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_joint_inference(args)