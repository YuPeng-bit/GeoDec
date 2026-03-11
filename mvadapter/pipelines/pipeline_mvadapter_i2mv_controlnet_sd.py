# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, ControlNetModel, MultiControlNetModel, UNet2DConditionModel, T2IAdapter
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ..loaders import CustomAdapterMixin
from ..models.attention_processor import (
    DecoupledMVRowSelfAttnProcessor2_0,
    set_unet_2d_condition_attn_processor,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class MVAdapterI2MVControlNetSDPipeline(StableDiffusionPipeline, CustomAdapterMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        
        self.register_modules(controlnet=controlnet)
        
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)
        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        return init_latents

    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        
        # MV-Adapter Params
        mv_scale: float = 1.0,
        control_image: Optional[PipelineImageInput] = None, 
        control_conditioning_scale: Optional[float] = 1.0,
        control_conditioning_factor: float = 1.0,
        reference_image: Optional[PipelineImageInput] = None,
        reference_conditioning_scale: Optional[float] = 1.0,
        
        # ControlNet Params
        normal_maps: Optional[PipelineImageInput] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        
        **kwargs,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, height, width, None, negative_prompt, prompt_embeds, negative_prompt_embeds, None, None, None
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=self.clip_skip,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ControlNet Image (Normal Maps)
        if normal_maps is not None:
            controlnet_image = self.prepare_control_image(
                image=normal_maps,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=1,
                device=device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        else:
            controlnet_image = None

        # MV-Adapter: Reference Image
        if reference_image is not None:
            reference_image_tensor = self.image_processor.preprocess(reference_image)
            reference_latents = self.prepare_image_latents(
                reference_image_tensor,
                timesteps[:1].repeat(batch_size * num_images_per_prompt),
                batch_size,
                1,
                prompt_embeds.dtype,
                device,
                generator,
                add_noise=False,
            )

            ref_timesteps = torch.zeros_like(timesteps[0])
            ref_hidden_states = {}
            
            with torch.no_grad():
                self.unet(
                    reference_latents,
                    ref_timesteps,
                    encoder_hidden_states=prompt_embeds[-1:],
                    cross_attention_kwargs={
                        "cache_hidden_states": ref_hidden_states,
                        "use_mv": False,
                        "use_ref": False,
                    },
                    return_dict=False,
                )
                
            ref_hidden_states = {
                k: v.repeat_interleave(num_images_per_prompt, dim=0)
                for k, v in ref_hidden_states.items()
            }
            if self.do_classifier_free_guidance:
                ref_hidden_states = {
                    k: torch.cat([torch.zeros_like(v), v], dim=0)
                    for k, v in ref_hidden_states.items()
                }
        else:
            ref_hidden_states = {}

        cross_attention_kwargs = {
            "num_views": num_images_per_prompt,
            "mv_scale": mv_scale,
            "ref_hidden_states": {k: v.clone() for k, v in ref_hidden_states.items()},
            "ref_scale": reference_conditioning_scale,
            **(self.cross_attention_kwargs or {}),
        }

        # MV-Adapter: Plucker Embedding
        if control_image is not None:
            control_image_feature = control_image.to(device=device, dtype=latents.dtype)
            
            adapter_state = self.cond_encoder(control_image_feature)
            for i, state in enumerate(adapter_state):
                adapter_state[i] = state * control_conditioning_scale
            
            # [CRITICAL FIX] Handle CFG for Adapter State
            if self.do_classifier_free_guidance:
                adapter_state = [torch.cat([state] * 2, dim=0) for state in adapter_state]
        else:
            adapter_state = None

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if adapter_state is not None and i < int(num_inference_steps * control_conditioning_factor):
                    down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
                else:
                    down_intrablock_additional_residuals = None

                if controlnet_image is not None:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_image,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    # ... Helper methods for CustomAdapterMixin ...
    def _init_custom_adapter(
        self,
        num_views: int = 1,
        self_attn_processor: Any = DecoupledMVRowSelfAttnProcessor2_0,
        cond_in_channels: int = 6,
        copy_attn_weights: bool = True,
        zero_init_module_keys: List[str] = [],
    ):
        self.cond_encoder = T2IAdapter(
            in_channels=cond_in_channels,
            channels=self.unet.config.block_out_channels,
            num_res_blocks=self.unet.config.layers_per_block,
            downscale_factor=8,
        )

        set_unet_2d_condition_attn_processor(
            self.unet,
            set_self_attn_proc_func=lambda name, hs, cad, ap: self_attn_processor(
                query_dim=hs, inner_dim=hs, num_views=num_views, name=name, use_mv=True, use_ref=True
            ),
            set_cross_attn_proc_func=lambda name, hs, cad, ap: self_attn_processor(
                query_dim=hs, inner_dim=hs, num_views=num_views, name=name, use_mv=False, use_ref=False
            ),
        )

        if copy_attn_weights:
            state_dict = self.unet.state_dict()
            for key in state_dict.keys():
                if "_mv" in key:
                    compatible_key = key.replace("_mv", "").replace("processor.", "")
                elif "_ref" in key:
                    compatible_key = key.replace("_ref", "").replace("processor.", "")
                else:
                    compatible_key = key

                is_zero_init_key = any([k in key for k in zero_init_module_keys])
                if is_zero_init_key:
                    state_dict[key] = torch.zeros_like(state_dict[compatible_key])
                else:
                    state_dict[key] = state_dict[compatible_key].clone()
            self.unet.load_state_dict(state_dict)

    def _load_custom_adapter(self, state_dict):
        self.unet.load_state_dict(state_dict, strict=False)
        self.cond_encoder.load_state_dict(state_dict, strict=False)

    def _save_custom_adapter(self, include_keys: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None):
        def include_fn(k):
            is_included = False
            if include_keys is not None:
                is_included = is_included or any([key in k for key in include_keys])
            if exclude_keys is not None:
                is_included = is_included and not any([key in k for key in exclude_keys])
            return is_included

        state_dict = {k: v for k, v in self.unet.state_dict().items() if include_fn(k)}
        state_dict.update(self.cond_encoder.state_dict())
        return state_dict