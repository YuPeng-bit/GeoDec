import os
import sys
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np
from diffusers import DDPMScheduler, UNet2DConditionModel, ControlNetModel
from diffusers.models import AutoencoderKL
from diffusers.training_utils import compute_snr
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

# 尝试导入 Pipelines
try:
    from ..pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
    from ..pipelines.pipeline_mvadapter_i2mv_controlnet_sd import MVAdapterI2MVControlNetSDPipeline
except ImportError:
    pass

from ..schedulers.scheduling_shift_snr import ShiftSNRScheduler
from ..utils.core import find
from ..utils.typing import *
from .base import BaseSystem
from .utils import encode_prompt, vae_encode

# [新增] 尝试导入 Evaluator，如果不在路径里则尝试添加根目录
try:
    from metrics import MVEvaluator
except ImportError:
    sys.path.append(os.getcwd())
    try:
        from metrics import MVEvaluator
    except ImportError:
        print("Warning: Could not import MVEvaluator. Metrics will not be computed.")
        MVEvaluator = None

class MVAdapterImageSDSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):

        # Model / Adapter
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        pretrained_vae_name_or_path: Optional[str] = None
        pretrained_adapter_name_or_path: Optional[str] = None
        pretrained_unet_name_or_path: Optional[str] = None
        
        # ControlNet
        pretrained_controlnet_name_or_path: Optional[str] = None
        
        init_adapter_kwargs: Dict[str, Any] = field(default_factory=dict)

        use_fp16_vae: bool = True
        use_fp16_clip: bool = True

        # Training
        trainable_modules: List[str] = field(default_factory=list)
        train_cond_encoder: bool = True
        prompt_drop_prob: float = 0.0
        image_drop_prob: float = 0.0
        cond_drop_prob: float = 0.0
        
        # [新增] Control Scale
        train_control_scale: float = 0.7 

        gradient_checkpointing: bool = False

        # Noise sampler
        noise_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
        noise_offset: float = 0.0
        input_perturbation: float = 0.0
        snr_gamma: Optional[float] = 5.0
        prediction_type: Optional[str] = None
        shift_noise: bool = False
        shift_noise_mode: str = "interpolated"
        shift_noise_scale: float = 1.0

        # Evaluation
        eval_seed: int = 42
        eval_num_inference_steps: int = 30
        eval_guidance_scale: float = 3.0 # 通常设为 3.0 效果更好
        eval_height: int = 512
        eval_width: int = 512
        
        optimizer: Dict[str, Any] = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()

        # 1. Prepare pipeline kwargs
        pipeline_kwargs = {}
        if self.cfg.pretrained_vae_name_or_path is not None:
            pipeline_kwargs["vae"] = AutoencoderKL.from_pretrained(
                self.cfg.pretrained_vae_name_or_path
            )
        if self.cfg.pretrained_unet_name_or_path is not None:
            pipeline_kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                self.cfg.pretrained_unet_name_or_path
            )

        # 2. Init Pipeline (根据是否用 ControlNet 自动选择类)
        self.controlnet = None
        if self.cfg.pretrained_controlnet_name_or_path:
            print(f"Loading ControlNet from {self.cfg.pretrained_controlnet_name_or_path}")
            self.controlnet = ControlNetModel.from_pretrained(self.cfg.pretrained_controlnet_name_or_path)
            self.controlnet.requires_grad_(False)
            self.controlnet.eval()
            if self.cfg.use_fp16_vae:
                 self.controlnet.to(dtype=torch.float16)
            
            # 使用带 ControlNet 的 Pipeline
            pipeline = MVAdapterI2MVControlNetSDPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path, 
                controlnet=self.controlnet,
                **pipeline_kwargs
            )
        else:
            # 使用原版 Pipeline
            pipeline = MVAdapterI2MVSDPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path, **pipeline_kwargs
            )

        # 3. Init Custom Adapter
        init_adapter_kwargs = OmegaConf.to_container(self.cfg.init_adapter_kwargs)
        if "self_attn_processor" in init_adapter_kwargs:
            self_attn_processor = init_adapter_kwargs["self_attn_processor"]
            if self_attn_processor is not None and isinstance(self_attn_processor, str):
                self_attn_processor = find(self_attn_processor)
                init_adapter_kwargs["self_attn_processor"] = self_attn_processor
        pipeline.init_custom_adapter(**init_adapter_kwargs)

        # 4. Load Pretrained Adapter Weights
        if self.cfg.pretrained_adapter_name_or_path:
            if os.path.isfile(self.cfg.pretrained_adapter_name_or_path):
                pretrained_path = os.path.dirname(self.cfg.pretrained_adapter_name_or_path)
                adapter_name = os.path.basename(self.cfg.pretrained_adapter_name_or_path)
            else:
                pretrained_path = self.cfg.pretrained_adapter_name_or_path
                adapter_name = "mvadapter_i2mv_sd21.safetensors"
                
            pipeline.load_custom_adapter(pretrained_path, weight_name=adapter_name)

        # 5. Setup Scheduler
        noise_scheduler = DDPMScheduler.from_config(
            pipeline.scheduler.config, **self.cfg.noise_scheduler_kwargs
        )
        if self.cfg.shift_noise:
            noise_scheduler = ShiftSNRScheduler.from_scheduler(
                noise_scheduler,
                shift_mode=self.cfg.shift_noise_mode,
                shift_scale=self.cfg.shift_noise_scale,
                scheduler_class=DDPMScheduler,
            )
        pipeline.scheduler = noise_scheduler

        # 6. Bind Attributes
        # 注意：这里我们使用了通用的 self.pipeline，它可能是普通版也可能是 ControlNet 版
        self.pipeline = pipeline 
        self.vae = self.pipeline.vae.to(
            dtype=torch.float16 if self.cfg.use_fp16_vae else torch.float32
        )
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder.to(
            dtype=torch.float16 if self.cfg.use_fp16_clip else torch.float32
        )
        self.feature_extractor = self.pipeline.feature_extractor

        self.cond_encoder = self.pipeline.cond_encoder
        self.unet = self.pipeline.unet
        self.noise_scheduler = self.pipeline.scheduler
        self.inference_scheduler = DDPMScheduler.from_config(
            self.noise_scheduler.config
        )
        self.pipeline.scheduler = self.inference_scheduler
        if self.cfg.prediction_type is not None:
            self.noise_scheduler.register_to_config(
                prediction_type=self.cfg.prediction_type
            )

        # 7. Unfreeze Logic
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.cond_encoder.requires_grad_(self.cfg.train_cond_encoder)

        trainable_count = 0
        if self.cfg.trainable_modules and len(self.cfg.trainable_modules) > 0:
            for name, param in self.unet.named_parameters():
                if any(tm in name for tm in self.cfg.trainable_modules):
                    param.requires_grad_(True)
                    trainable_count += 1
        
        print(f"====== System Configuration ======")
        print(f"Trainable UNet Parameters: {trainable_count}")
        print(f"Training Control Scale: {self.cfg.train_control_scale}")

        if self.cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.cfg.train_cond_encoder:
                self.cond_encoder.enable_gradient_checkpointing()
        
        # [新增] 初始化 Evaluator (只在主进程初始化以节省显存，或者所有进程都跑但只在rank0打印)
        self.evaluator = None
        if MVEvaluator is not None:
            self.evaluator = MVEvaluator(device=self.device)

    def forward(
        self,
        noisy_latents: Tensor,
        conditioning_pixel_values: Tensor,
        timesteps: Tensor,
        ref_latents: Tensor,
        prompts: List[str],
        num_views: int,
        **kwargs,
    ) -> Dict[str, Any]:
        bsz = noisy_latents.shape[0]
        b_samples = bsz // num_views
        num_batch_images = num_views

        # 1. Text Encoding
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            input_ids = self.tokenizer(
                prompts,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].to(noisy_latents.device)
            encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[0].float()

        # 2. Drops
        prompt_drop_mask = torch.rand(b_samples, device=noisy_latents.device) < self.cfg.prompt_drop_prob
        image_drop_mask = torch.rand(b_samples, device=noisy_latents.device) < self.cfg.image_drop_prob
        cond_drop_mask = torch.rand(b_samples, device=noisy_latents.device) < self.cfg.cond_drop_prob
        prompt_drop_mask = prompt_drop_mask | cond_drop_mask
        image_drop_mask = image_drop_mask | cond_drop_mask

        encoder_hidden_states[prompt_drop_mask] = 0.0

        # 3. Reference Encoding
        with torch.no_grad():
            ref_timesteps = torch.zeros_like(timesteps[:b_samples])
            ref_hidden_states = {}
            self.unet(
                ref_latents,
                ref_timesteps,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs={
                    "cache_hidden_states": ref_hidden_states,
                    "use_mv": False,
                    "use_ref": False,
                },
                return_dict=False,
            )
            for k, v in ref_hidden_states.items():
                v_ = v
                v_[image_drop_mask] = 0.0
                ref_hidden_states[k] = v_.repeat_interleave(num_batch_images, dim=0)

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_batch_images, dim=0)

        # 4. Condition Encoder (Plucker)
        plucker_embeds = conditioning_pixel_values[:, :6, :, :]
        conditioning_features = self.cond_encoder(plucker_embeds)

        # 5. ControlNet Forward
        down_block_res_samples = None
        mid_block_res_sample = None
        
        if self.controlnet is not None:
            if conditioning_pixel_values.shape[1] >= 9:
                normal_maps = conditioning_pixel_values[:, 6:9, :, :]
            else:
                normal_maps = torch.zeros_like(conditioning_pixel_values[:, :3, :, :])
            
            # 使用 Config 中的 scale
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=normal_maps.to(dtype=self.controlnet.dtype),
                conditioning_scale=self.cfg.train_control_scale, 
                return_dict=False,
            )

        # 6. UNet Forward
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_intrablock_additional_residuals=conditioning_features,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            cross_attention_kwargs={
                "ref_hidden_states": ref_hidden_states,
                "num_views": num_views,
                "use_mv": True,
                "use_ref": True,
            },
        ).sample

        return {"noise_pred": noise_pred}

    def training_step(self, batch, batch_idx):
        num_views = batch["num_views"]
        vae_max_slice = 8
        
        # VAE Encoding (Target)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            latents = []
            for i in range(0, batch["rgb"].shape[0], vae_max_slice):
                latents.append(
                    vae_encode(
                        self.vae,
                        batch["rgb"][i : i + vae_max_slice].to(self.vae.dtype) * 2 - 1,
                        sample=True,
                        apply_scale=True,
                    ).float()
                )
            latents = torch.cat(latents, dim=0)

        # VAE Encoding (Ref)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            ref_latents = vae_encode(
                self.vae,
                batch["reference_rgb"].to(self.vae.dtype) * 2 - 1,
                sample=True,
                apply_scale=True,
            ).float()

        bsz = latents.shape[0]
        b_samples = bsz // num_views

        # Noise & Timesteps
        noise = torch.randn_like(latents)
        if self.cfg.noise_offset is not None:
            noise += self.cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        noise_mask = batch.get("noise_mask", torch.ones((bsz,), dtype=torch.bool, device=latents.device))
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (b_samples,), device=latents.device, dtype=torch.long
        )
        timesteps = timesteps.repeat_interleave(num_views)
        timesteps[~noise_mask] = 0

        if self.cfg.input_perturbation is not None:
            new_noise = noise + self.cfg.input_perturbation * torch.randn_like(noise)
            noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noisy_latents[~noise_mask] = latents[~noise_mask]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {self.noise_scheduler.config.prediction_type}")

        conditioning_pixel_values = batch["source_rgb"]
        
        model_pred = self(
            noisy_latents, conditioning_pixel_values, timesteps, ref_latents, **batch
        )["noise_pred"]

        model_pred = model_pred[noise_mask]
        target = target[noise_mask]

        if self.cfg.snr_gamma is None:
            loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                snr = snr + 1
            mse_loss_weights = (
                torch.stack([snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred, target, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        self.log("train/loss", loss, prog_bar=True)
        self.check_train(batch)
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass
        
    def configure_optimizers(self):
        unet_params = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        cond_params = list(filter(lambda p: p.requires_grad, self.cond_encoder.parameters()))
        
        params_to_optimize = []
        if len(unet_params) > 0:
            params_to_optimize.append({"params": unet_params, "lr": self.cfg.optimizer["args"]["lr"]})
        if len(cond_params) > 0:
            lr = self.cfg.optimizer.get("params", {}).get("cond_encoder", {}).get("lr", self.cfg.optimizer["args"]["lr"])
            params_to_optimize.append({"params": cond_params, "lr": lr})

        if not params_to_optimize:
            return torch.optim.AdamW([torch.tensor(0.0, requires_grad=True)], lr=0.0)

        optimizer = torch.optim.AdamW(params_to_optimize, **self.cfg.optimizer["args"])
        return optimizer

    # Visualization & Validation Helpers
    def get_input_visualizations(self, batch):
        return [
            {
                "type": "rgb",
                "img": rearrange(batch["source_rgb"], "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(batch["reference_rgb"], "B C H W -> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(batch["rgb"], "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]),
                "kwargs": {"data_format": "HWC"},
            },
        ]

    def get_output_visualizations(self, batch, outputs):
        images = [
            {
                "type": "rgb",
                "img": rearrange(batch["source_rgb"], "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(batch["rgb"], "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(batch["reference_rgb"], "B C H W -> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(outputs, "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]),
                "kwargs": {"data_format": "HWC"},
            },
        ]
        return images

    def generate_images(self, batch, **kwargs):
        # [核心] 如果有ControlNet，则传入 normal_maps
        gen_kwargs = {}
        if self.controlnet is not None:
            # 假设 source_rgb 通道 6:9 是法线
            if batch["source_rgb"].shape[1] >= 9:
                gen_kwargs["normal_maps"] = batch["source_rgb"][:, 6:9]
            else:
                gen_kwargs["normal_maps"] = torch.zeros_like(batch["source_rgb"][:, :3])
            
            gen_kwargs["controlnet_conditioning_scale"] = self.cfg.train_control_scale

        return self.pipeline(
            prompt=batch["prompts"],
            control_image=batch["source_rgb"][:, :6], # Plucker
            num_images_per_prompt=batch["num_views"],
            generator=torch.Generator(device=self.device).manual_seed(self.cfg.eval_seed),
            num_inference_steps=self.cfg.eval_num_inference_steps,
            guidance_scale=self.cfg.eval_guidance_scale,
            height=self.cfg.eval_height,
            width=self.cfg.eval_width,
            reference_image=batch["reference_rgb"],
            output_type="pt",
            **gen_kwargs
        ).images

    def on_save_checkpoint(self, checkpoint):
        if self.global_rank == 0:
            self.pipeline.save_custom_adapter(
                os.path.dirname(self.get_save_dir()),
                "custom_adapter.safetensors",
                safe_serialization=True,
                include_keys=self.cfg.trainable_modules,
            )

    def on_check_train(self, batch):
        self.save_image_grid(
            f"it{self.true_global_step}-train.jpg",
            self.get_input_visualizations(batch),
            name="train_step_input",
            step=self.true_global_step,
        )

    # [核心新增] 带 Metric 计算的 Validation Step
    def validation_step(self, batch, batch_idx):
        # 生成图片 (Tensor B*V, C, H, W)
        pred_imgs = self.generate_images(batch)
        
        # 计算 Metrics (仅当有 Evaluator 时)
        if self.evaluator is not None:
            # GT (Ground Truth)
            gt_imgs = batch["rgb"] # (B*V, C, H, W)
            
            # 由于 batch 里可能是 (0,1) 范围，我们确保都转成 evaluator 需要的格式
            # MVEvaluator 通常处理 PIL 或 [0,1] Tensor
            # 我们先转为 list of tensors
            
            # 计算平均指标
            metrics = self.evaluator.compute(pred_imgs, gt_imgs)
            
            # Log metrics (自动求所有 batch 的平均)
            self.log("val/psnr", metrics["psnr"], prog_bar=True, sync_dist=True)
            self.log("val/ssim", metrics["ssim"], prog_bar=True, sync_dist=True)
            self.log("val/lpips", metrics["lpips"], prog_bar=True, sync_dist=True)

        # 保存可视化图片 (限制数量，避免磁盘爆炸)
        if self.cfg.check_val_limit_rank > 0 and self.global_rank < self.cfg.check_val_limit_rank:
            # 只有前几个 batch 才保存图，比如前 2 个 batch
            if batch_idx < 2: 
                self.save_image_grid(
                    f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}.jpg",
                    self.get_output_visualizations(batch, pred_imgs),
                    name=f"validation_step_output_{self.global_rank}_{batch_idx}",
                    step=self.true_global_step,
                )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self.generate_images(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test-{self.global_rank}_{batch_idx}.jpg",
            self.get_output_visualizations(batch, out),
            name=f"test_step_output_{self.global_rank}_{batch_idx}",
            step=self.true_global_step,
        )

    def on_test_end(self):
        pass