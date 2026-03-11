import argparse
import os
import sys
import torch
import numpy as np
import gc
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

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
from metrics import MVEvaluator

# ==========================================
# 工具函数
# ==========================================
def force_cast_pipeline(pipe, device, dtype):
    pipe.to(device)
    if hasattr(pipe, "unet") and pipe.unet: pipe.unet.to(device=device, dtype=dtype)
    if hasattr(pipe, "vae") and pipe.vae: pipe.vae.to(device=device, dtype=dtype)
    if hasattr(pipe, "text_encoder") and pipe.text_encoder: pipe.text_encoder.to(device=device, dtype=dtype)
    if hasattr(pipe, "cond_encoder") and pipe.cond_encoder: pipe.cond_encoder.to(device=device, dtype=dtype)
    if hasattr(pipe, "controlnet") and pipe.controlnet: pipe.controlnet.to(device=device, dtype=dtype)
    return pipe

def smart_load_weights(model, state_dict, model_name="Model"):
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

def preprocess_image(image: Image.Image, height: int = 512, width: int = 512):
    if image.mode != "RGBA": image = image.convert("RGBA")
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
# 数据集
# ==========================================
class ComparisonDataset(Dataset):
    def __init__(self, root_dir, split_json, seed=42, num_samples=50):
        import random, json
        self.rand_dir = os.path.join(root_dir, "texture_rand_easylight_objaverse")
        self.ortho_dir = os.path.join(root_dir, "texture_ortho10view_easylight_objaverse")
        with open(split_json, 'r') as f: all_ids = json.load(f)
        random.seed(seed)
        self.sample_ids = sorted(random.sample([str(x) for x in all_ids], min(num_samples, len(all_ids))))
        self.gt_indices = ["0000", "0004", "0001", "0002", "0003", "0005"]

    def __len__(self): return len(self.sample_ids)
    def __getitem__(self, idx):
        uid = self.sample_ids[idx]
        sub = uid[:2]
        ref_path = os.path.join(self.rand_dir, sub, uid, "color_0000.webp")
        gt_paths = [os.path.join(self.ortho_dir, sub, uid, f"color_{i}.webp") for i in self.gt_indices]
        return {"uid": uid, "ref_path": ref_path, "gt_paths": gt_paths}

# ==========================================
# 主评测逻辑
# ==========================================
def run_comparison(args):
    device = args.device
    dtype = torch.float16
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    dataset = ComparisonDataset(args.data_root, args.split_json, args.seed, args.num_samples)
    evaluator = MVEvaluator(device=device)
    
    # 结果容器
    results = {
        "baseline": {"psnr": [], "ssim": [], "lpips": []},
        "ours_finetuned": {"psnr": [], "ssim": [], "lpips": []}
    }
    
    # 相机条件
    azimuths = [0, 45, 90, 180, 270, 315]
    cameras = get_orthogonal_camera(elevation_deg=[0]*6, distance=[1.2]*6, left=-0.55, right=0.55, bottom=-0.55, top=0.55, azimuth_deg=[x-90 for x in azimuths], device=device)
    plucker = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [1.1]*6, 512)
    control_images = ((plucker + 1.0) / 2.0).clamp(0, 1).to(device=device, dtype=dtype)

    # ---------------------------------------------------------
    # PHASE 1: Baseline (Original)
    # ---------------------------------------------------------
    print("\n[Phase 1] Running Baseline...")
    pipe_base = MVAdapterI2MVSDPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    pipe_base.scheduler = ShiftSNRScheduler.from_scheduler(pipe_base.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=DDPMScheduler)
    pipe_base.init_custom_adapter(num_views=6)
    pipe_base.load_custom_adapter(args.baseline_adapter, weight_name="mvadapter_i2mv_sd21.safetensors")
    pipe_base = force_cast_pipeline(pipe_base, device, dtype)
    
    baseline_preds = {}
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        try:
            if not os.path.exists(sample['ref_path']): continue
            ref_img = preprocess_image(Image.open(sample['ref_path']))
            imgs = pipe_base(
                prompt="high quality", height=512, width=512, num_inference_steps=30, guidance_scale=3.0, num_images_per_prompt=6,
                control_image=control_images, reference_image=ref_img, reference_conditioning_scale=1.0,
                negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast"
            ).images
            scores = evaluator.compute(imgs, sample['gt_paths'])
            for k in results["baseline"]: results["baseline"][k].append(scores[k])
            baseline_preds[sample['uid']] = imgs
        except Exception as e: print(f"Err baseline {sample['uid']}: {e}")
            
    del pipe_base; gc.collect(); torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # PHASE 2: Generate Normals (Using Stage 1 Model)
    # ---------------------------------------------------------
    print("\n[Phase 2] Generating Geometry (Stage 1)...")
    pipe_s1 = MVAdapterI2MVSDPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    pipe_s1.scheduler = ShiftSNRScheduler.from_scheduler(pipe_s1.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=DDPMScheduler)
    pipe_s1.init_custom_adapter(num_views=6)
    
    ckpt_s1 = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    sd_s1 = ckpt_s1["state_dict"] if "state_dict" in ckpt_s1 else ckpt_s1
    smart_load_weights(pipe_s1.unet, {k:v for k,v in sd_s1.items() if "unet" in k}, "Stage1-UNet")
    smart_load_weights(pipe_s1.cond_encoder, {k:v for k,v in sd_s1.items() if "cond_encoder" in k}, "Stage1-Encoder")
    pipe_s1 = force_cast_pipeline(pipe_s1, device, dtype)
    
    temp_normals = {}
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        try:
            if not os.path.exists(sample['ref_path']): continue
            ref_img = preprocess_image(Image.open(sample['ref_path']))
            normals = pipe_s1(
                prompt="high quality", height=512, width=512, num_inference_steps=30, guidance_scale=3.0, num_images_per_prompt=6,
                control_image=control_images, reference_image=ref_img, reference_conditioning_scale=1.0,
                negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast"
            ).images
            temp_normals[sample['uid']] = normals
        except: pass
        
    del pipe_s1; gc.collect(); torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # PHASE 3: Ours Fine-tuned (Stage 2)
    # ---------------------------------------------------------
    print("\n[Phase 3] Running Ours Fine-tuned Stage 2...")
    cnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe_s2 = MVAdapterI2MVControlNetSDPipeline.from_pretrained(args.base_model, controlnet=cnet, torch_dtype=dtype, safety_checker=None)
    pipe_s2.scheduler = ShiftSNRScheduler.from_scheduler(pipe_s2.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=DDPMScheduler)
    pipe_s2.init_custom_adapter(num_views=6)
    
    # [关键修改] 加载刚刚微调好的 Stage 2 权重
    print(f"Loading Fine-tuned Stage 2 weights from: {args.stage2_ckpt}")
    ckpt_s2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    sd_s2 = ckpt_s2["state_dict"] if "state_dict" in ckpt_s2 else ckpt_s2
    smart_load_weights(pipe_s2.unet, {k:v for k,v in sd_s2.items() if "unet" in k}, "Stage2-UNet")
    smart_load_weights(pipe_s2.cond_encoder, {k:v for k,v in sd_s2.items() if "cond_encoder" in k}, "Stage2-Encoder")
    
    pipe_s2 = force_cast_pipeline(pipe_s2, device, dtype)
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        uid = sample['uid']
        try:
            if uid not in temp_normals: continue
            ref_img = preprocess_image(Image.open(sample['ref_path']))
            norm_imgs = [n.convert("RGB") for n in temp_normals[uid]]
            
            # 使用训练时设定的 control scale 0.7
            ours_imgs = pipe_s2(
                prompt="high quality", height=512, width=512, num_inference_steps=30, guidance_scale=3.0, num_images_per_prompt=6,
                control_image=control_images, reference_image=ref_img, normal_maps=norm_imgs, 
                controlnet_conditioning_scale=0.7, # 你的直觉值
                negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast"
            ).images
            
            scores = evaluator.compute(ours_imgs, sample['gt_paths'])
            for k in results["ours_finetuned"]: results["ours_finetuned"][k].append(scores[k])
            
            # Visualization
            if i < 200 and uid in baseline_preds:
                W, H = 256, 256
                combo = Image.new('RGB', (W * 7, H * 4), (255, 255, 255))
                combo.paste(ref_img.resize((W, H)), (0, 0))
                gts = [preprocess_image(Image.open(p)).resize((W, H)) for p in sample['gt_paths']]
                
                # Rows: Baseline | Ours | GT | Normals
                for j, img in enumerate(baseline_preds[uid]): combo.paste(img.resize((W, H)), ((j+1)*W, H))
                for j, img in enumerate(ours_imgs): combo.paste(img.resize((W, H)), ((j+1)*W, H*2))
                for j, img in enumerate(gts): combo.paste(img, ((j+1)*W, H*3))
                
                draw = ImageDraw.Draw(combo)
                fnt = ImageFont.load_default()
                draw.text((10, H+10), "Baseline", fill="black", font=fnt)
                draw.text((10, H*2+10), "Ours FT", fill="black", font=fnt)
                draw.text((10, H*3+10), "GT", fill="black", font=fnt)
                combo.save(os.path.join(vis_dir, f"{uid}_compare_v3.jpg"))
        except Exception as e: print(f"Err s2 {uid}: {e}")

    # Report
    print("\n=== Final Report ===")
    final = {}
    for m in results:
        if len(results[m]['psnr']) > 0:
            final[m] = {k: np.mean(v) for k, v in results[m].items()}
            print(f"[{m.upper()}] PSNR: {final[m]['psnr']:.4f} | SSIM: {final[m]['ssim']:.4f} | LPIPS: {final[m]['lpips']:.4f}")
    
    with open(os.path.join(args.output_dir, "metrics_v3.json"), 'w') as f: json.dump(final, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Path to FINE-TUNED Stage 2 .ckpt")
    parser.add_argument("--data_root", type=str, default="/data/xyp/MV-Adapter/mvadapter/data")
    parser.add_argument("--split_json", type=str, default="/data/xyp/MV-Adapter/mvadapter/data/objaverse_rest.json")
    parser.add_argument("--output_dir", type=str, default="final_benchmark_v3")
    parser.add_argument("--base_model", type=str, default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--baseline_adapter", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--controlnet", type=str, default="thibaud/controlnet-sd21-normalbae-diffusers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_comparison(args)