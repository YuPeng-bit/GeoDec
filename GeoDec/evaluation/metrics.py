import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from torchvision import transforms
from PIL import Image
import numpy as np

class MVEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = lpips.LPIPS(net='alex').to(device)
        
        self.to_tensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
    def composite_on_gray(self, img):
        """
        严格对齐 inference_i2mv_sdxl.py 中的背景处理逻辑:
        RGB * Alpha + (1 - Alpha) * 0.5
        """
        img = img.convert("RGBA")
        
        # 转换为 numpy 进行精确计算，这比 PIL 的 alpha_composite 更能对齐模型的预处理
        arr = np.array(img).astype(np.float32) / 255.0
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3:4]
        
        # 背景设为 0.5 (即灰色)
        bg_color = 0.5
        
        # 混合公式
        composite = rgb * alpha + (1 - alpha) * bg_color
        
        # 转回 uint8
        composite = (composite * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(composite)

    @torch.no_grad()
    def compute_single(self, pred_pil, gt_pil_rgba):
        # 1. 处理 GT: 使用相同的灰色合成逻辑
        gt_rgb = self.composite_on_gray(gt_pil_rgba)
        
        # 2. 处理 Pred: 
        # 推理出来的图已经是 RGB 且是灰底（经过 preprocess_image），通常不需要动。
        # 但如果是 RGBA，则进行合成。
        if pred_pil.mode == 'RGBA':
            pred_rgb = self.composite_on_gray(pred_pil)
        else:
            pred_rgb = pred_pil.convert("RGB")
            
        # 3. 转 Tensor 计算
        pred = self.to_tensor(pred_rgb).unsqueeze(0).to(self.device)
        gt = self.to_tensor(gt_rgb).unsqueeze(0).to(self.device)
        
        return {
            "lpips": self.lpips_metric(pred * 2 - 1, gt * 2 - 1).item(),
            "psnr": self.psnr_metric(pred, gt).item(),
            "ssim": self.ssim_metric(pred, gt).item()
        }

    # ... compute 方法保持不变 ...
    @torch.no_grad()
    def compute(self, pred_pil_list, gt_path_list):
        """批量计算 (兼容旧接口)"""
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        
        for pred_img, gt_path in zip(pred_pil_list, gt_path_list):
            gt_img = Image.open(gt_path)
            scores = self.compute_single(pred_img, gt_img)
            
            metrics["psnr"].append(scores["psnr"])
            metrics["ssim"].append(scores["ssim"])
            metrics["lpips"].append(scores["lpips"])
            
        return {
            "psnr": sum(metrics["psnr"]) / len(metrics["psnr"]),
            "ssim": sum(metrics["ssim"]) / len(metrics["ssim"]),
            "lpips": sum(metrics["lpips"]) / len(metrics["lpips"])
        }