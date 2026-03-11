import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path to import mvadapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvadapter.pipelines.pipeline_mvadapter_i2mv_controlnet_sd import MVAdapterI2MVControlNetSDPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils.mesh_utils import get_orthogonal_camera
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho

# ==========================================
# 核心预处理函数 (SD2.1 512x512, 居中, 灰底)
# ==========================================
def preprocess_image(image: Image.Image, height: int, width: int):
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    image = np.array(image)
    alpha = image[..., 3] > 0
    H, W = alpha.shape
    
    y, x = np.where(alpha)
    if len(y) == 0 or len(x) == 0:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
        x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    
    image_center = image[y0:y1, x0:x1]
    
    H_c, W_c, _ = image_center.shape
    if H_c > W_c:
        W_new = int(W_c * (height * 0.9) / H_c)
        H_new = int(height * 0.9)
    else:
        H_new = int(H_c * (width * 0.9) / W_c)
        W_new = int(width * 0.9)
    
    image_center = np.array(Image.fromarray(image_center).resize((W_new, H_new), Image.Resampling.BILINEAR))
    
    start_h = (height - H_new) // 2
    start_w = (width - W_new) // 2
    
    final_image = np.zeros((height, width, 4), dtype=np.uint8)
    final_image[start_h : start_h + H_new, start_w : start_w + W_new] = image_center
    
    final_image = final_image.astype(np.float32) / 255.0
    # 合成灰色背景 (0.5)
    final_image = final_image[:, :, :3] * final_image[:, :, 3:4] + (1 - final_image[:, :, 3:4]) * 0.5
    final_image = (final_image * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(final_image)

def prepare_pipeline(
    base_model,
    adapter_path,
    controlnet_path,
    num_views,
    device,
    dtype,
    scheduler_type="ddpm"
):
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter: {adapter_path}")
    print(f"Loading controlnet: {controlnet_path}")

    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)

    # Load Pipeline
    pipe = MVAdapterI2MVControlNetSDPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None # Disable safety checker for speed
    )

    # Scheduler
    scheduler_class = DDPMScheduler
    if scheduler_type == "lcm":
        scheduler_class = LCMScheduler
    
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    
    # Initialize & Load MV-Adapter
    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_i2mv_sd21.safetensors"
    )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)
    pipe.controlnet.to(device=device, dtype=dtype)
    
    # Enable VAE slicing for memory efficiency
    pipe.enable_vae_slicing()

    return pipe

def get_data_paths(data_root, uid):
    # Construct paths based on Objaverse structure
    # Reference (RGB): texture_rand_easylight_objaverse/xx/uid/color_0000.webp
    # GT Normals: texture_ortho10view_easylight_objaverse/xx/uid/normal_{0000,0004,0001,0002,0003,0005}.webp
    # GT RGB: texture_ortho10view_easylight_objaverse/xx/uid/color_{0000,0004,0001,0002,0003,0005}.webp
    
    sub_dir = uid[:2]
    
    # Paths
    rand_dir = os.path.join(data_root, "texture_rand_easylight_objaverse", sub_dir, uid)
    ortho_dir = os.path.join(data_root, "texture_ortho10view_easylight_objaverse", sub_dir, uid)
    
    # Reference image (random view 0)
    ref_path = os.path.join(rand_dir, "color_0000.webp")
    
    # 6 canonical views indices
    # Order: Front, Front-Left, Left, Back, Right, Front-Right
    # Indices: 0000, 0004, 0001, 0002, 0003, 0005
    indices = ["0000", "0004", "0001", "0002", "0003", "0005"]
    
    gt_rgb_paths = [os.path.join(ortho_dir, f"color_{idx}.webp") for idx in indices]
    gt_normal_paths = [os.path.join(ortho_dir, f"normal_{idx}.webp") for idx in indices]
    
    return ref_path, gt_rgb_paths, gt_normal_paths

# 辅助函数：合成灰色背景
def composite_gray(img_pil):
    if img_pil.mode != 'RGBA':
        img_pil = img_pil.convert('RGBA')
    # 使用纯灰色背景 (127, 127, 127)
    bg = Image.new('RGBA', img_pil.size, (127, 127, 127, 255))
    return Image.alpha_composite(bg, img_pil).convert('RGB')

def make_grid_comparison(ref_img, pred_imgs, gt_imgs, normal_imgs, uid, save_path):
    W, H = 256, 256
    num_views = 6
    cols = num_views + 1
    rows = 3
    
    grid_img = Image.new('RGB', (cols * W, rows * H), (255, 255, 255))
    
    # 1. 预处理参考图 (已经预处理过了，直接 Resize)
    ref_vis = ref_img.resize((W, H))
    
    # 2. 预处理预测图 (已经是 RGB 灰底，直接 Resize)
    preds_vis = [img.resize((W, H)) for img in pred_imgs]
    
    # 3. [修复点] 预处理 GT RGB (先合成灰底，再 Resize)
    gts_vis = []
    for p in gt_imgs:
        raw_gt = Image.open(p)
        # 关键步骤：处理透明通道，防止拉花
        comp_gt = composite_gray(raw_gt) 
        gts_vis.append(comp_gt.resize((W, H)))
        
    # 4. [修复点] 预处理 GT Normal (同上)
    normals_vis = []
    for p in normal_imgs:
        raw_norm = Image.open(p)
        comp_norm = composite_gray(raw_norm)
        normals_vis.append(comp_norm.resize((W, H)))
    
    # --- 绘图 ---
    # Row 1: Reference | Predictions
    grid_img.paste(ref_vis, (0, 0))
    for i, img in enumerate(preds_vis):
        grid_img.paste(img, ((i + 1) * W, 0))
        
    # Row 2: Empty | GT RGB
    for i, img in enumerate(gts_vis):
        grid_img.paste(img, ((i + 1) * W, H))
        
    # Row 3: Empty | GT Normal
    for i, img in enumerate(normals_vis):
        grid_img.paste(img, ((i + 1) * W, 2 * H))
        
    draw = ImageDraw.Draw(grid_img)
    try: font = ImageFont.truetype("arial.ttf", 20)
    except: font = ImageFont.load_default()
    
    draw.text((10, 10), "Ref Input", fill="red", font=font)
    draw.text((W + 10, 10), "Prediction (Stage 2)", fill="red", font=font)
    draw.text((W + 10, H + 10), "Ground Truth RGB", fill="red", font=font)
    draw.text((W + 10, 2 * H + 10), "Input Normals (GT)", fill="red", font=font)
    
    grid_img.save(save_path)
    print(f"Saved comparison to {save_path}")

def run_inference(args):
    device = args.device
    
    # 1. Load Pipeline
    pipe = prepare_pipeline(
        base_model="stabilityai/stable-diffusion-2-1-base",
        adapter_path="huanngzh/mv-adapter",
        controlnet_path="thibaud/controlnet-sd21-normalbae-diffusers",
        num_views=6,
        device=device,
        dtype=torch.float16
    )
    
    # 2. Select Sample
    with open(args.split_json, 'r') as f:
        all_ids = json.load(f)
    
    if args.sample_id:
        uid = args.sample_id
    else:
        random.seed(args.seed)
        uid = random.choice(all_ids)
    
    print(f"Running inference for UID: {uid}")
    
    # 3. Load Data
    ref_path, gt_rgb_paths, gt_normal_paths = get_data_paths(args.data_root, uid)
    
    if not os.path.exists(ref_path):
        print(f"Error: Data not found for {uid}")
        return

    # 4. Preprocess Inputs
    # IMPORTANT: Both Reference and Normals must undergo the SAME preprocessing 
    # to ensure geometry alignment (centering, cropping, resizing).
    
    raw_ref = Image.open(ref_path)
    processed_ref = preprocess_image(raw_ref, 512, 512)
    
    # Preprocess normals
    # ControlNet expects images. We process them individually.
    processed_normals = []
    for p in gt_normal_paths:
        raw_norm = Image.open(p)
        # Apply same centering/cropping logic
        processed_norm = preprocess_image(raw_norm, 512, 512)
        processed_normals.append(processed_norm)
    
    # 5. Prepare Cameras (Plucker Embeddings)
    # MV-Adapter standard canonical views
    azimuth_deg = [0, 45, 90, 180, 270, 315]
    cameras = get_orthogonal_camera(
        elevation_deg=[0] * 6,
        distance=[1.8] * 6,
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[x - 90 for x in azimuth_deg],
        device=device,
    )
    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * 6, 512
    )
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1) # Plucker for MV-Adapter

    # 6. Run Pipeline
    pipe_kwargs = {}
    if args.seed != -1:
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(args.seed)

    images = pipe(
        prompt="high quality", 
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=3.0,
        num_images_per_prompt=6,
        
        # MV-Adapter Inputs
        control_image=control_images, # Plucker
        control_conditioning_scale=1.0,
        reference_image=processed_ref, # RGB Ref
        reference_conditioning_scale=1.0,
        
        # ControlNet Inputs
        normal_maps=processed_normals, # List of 6 PIL Images
        controlnet_conditioning_scale=args.control_scale, # Tune this!
        
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        **pipe_kwargs,
    ).images

    # 7. Visualization
    output_path = os.path.join(args.output_dir, f"{uid}_stage2.jpg")
    make_grid_comparison(processed_ref, images, gt_rgb_paths, gt_normal_paths, uid, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/xyp/MV-Adapter/mvadapter/data")
    parser.add_argument("--split_json", type=str, default="/data/xyp/MV-Adapter/mvadapter/data/objaverse_rest.json")
    parser.add_argument("--output_dir", type=str, default="stage2_results")
    parser.add_argument("--sample_id", type=str, help="Specific Object ID to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control_scale", type=float, default=1.0, help="ControlNet strength")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_inference(args)