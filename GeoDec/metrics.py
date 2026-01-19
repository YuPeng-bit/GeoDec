import torch
import numpy as np
from PIL import Image
import os
import lpips
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

class MVEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        # 初始化 LPIPS 模型 (AlexNet 是主流选择，也就是论文里常用的)
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.lpips_fn.eval()
        except Exception as e:
            print(f"Warning: Failed to initialize LPIPS: {e}")
            self.lpips_fn = None

    def preprocess(self, data, size=(512, 512)):
        """
        统一将输入转换为 (1, 3, H, W) 的 Tensor，范围 [0, 1]
        关键修复：对透明图片进行灰底混合 (Alpha Blending on Gray)
        """
        img_tensor = None

        # --- Case 1: 路径 (str) ---
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Image not found: {data}")
            pil_img = Image.open(data)
        
        # --- Case 2: PIL Image ---
        elif isinstance(data, Image.Image):
            pil_img = data
        
        # --- Case 3: Tensor ---
        elif isinstance(data, torch.Tensor):
            img_tensor = data.clone()
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.float()
            # 假设 Tensor 已经是 [0, 1] 且已经处理好背景，直接返回
            return img_tensor.to(self.device)
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

        # === 统一处理 PIL Image (关键背景修复) ===
        # 1. Resize
        pil_img = pil_img.resize(size, Image.Resampling.BILINEAR)
        
        # 2. Alpha Blending (如果是 RGBA，混合到灰底)
        if pil_img.mode == 'RGBA':
            # 创建灰底画布 (127, 127, 127)
            background = Image.new("RGBA", size, (127, 127, 127, 255))
            # 混合: Alpha Composite
            composite = Image.alpha_composite(background, pil_img)
            pil_img = composite.convert("RGB")
        else:
            # 如果本身没 Alpha，直接转 RGB
            pil_img = pil_img.convert("RGB")

        # 3. To Tensor [0, 1]
        arr = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
        
        return img_tensor.to(self.device)

    @torch.no_grad()
    def compute(self, preds, gts):
        """
        计算一批图片或单个图片的平均指标
        """
        psnr_list = []
        ssim_list = []
        lpips_list = []

        # 统一处理为 List 以便逐个对比
        # 如果是 Tensor Batch，拆分
        if isinstance(preds, torch.Tensor):
            preds_list = [preds[i] for i in range(preds.shape[0])]
        else:
            preds_list = preds
            
        if isinstance(gts, torch.Tensor):
            gts_list = [gts[i] for i in range(gts.shape[0])]
        else:
            gts_list = gts

        # 长度对齐
        min_len = min(len(preds_list), len(gts_list))
        
        for i in range(min_len):
            # 预处理 (会自动处理背景)
            p = self.preprocess(preds_list[i])
            g = self.preprocess(gts_list[i])
            
            # Metric 1: PSNR
            psnr_val = peak_signal_noise_ratio(p, g, data_range=1.0)
            psnr_list.append(psnr_val.item())
            
            # Metric 2: SSIM
            ssim_val = structural_similarity_index_measure(p, g, data_range=1.0)
            ssim_list.append(ssim_val.item())
            
            # Metric 3: LPIPS
            if self.lpips_fn is not None:
                # LPIPS 需要 [-1, 1]
                p_norm = p * 2.0 - 1.0
                g_norm = g * 2.0 - 1.0
                lpips_val = self.lpips_fn(p_norm, g_norm)
                lpips_list.append(lpips_val.item())

        return {
            "psnr": np.mean(psnr_list) if psnr_list else 0.0,
            "ssim": np.mean(ssim_list) if ssim_list else 0.0,
            "lpips": np.mean(lpips_list) if lpips_list else 0.0
        }