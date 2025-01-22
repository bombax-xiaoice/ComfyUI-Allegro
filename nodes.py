import os
import torch
import folder_paths
import torchvision.transforms
from comfy.utils import ProgressBar, calculate_parameters, weight_dtype
from comfy.cli_args import args
from comfy import model_management
import latent_preview
import comfy.latent_formats
import random
import math
import typing
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

import sys
comfy_path = os.path.dirname(folder_paths.__file__)
sys.path.append(f'{comfy_path}/custom_nodes/ComfyUI-Allegro')
print(sys.path)

from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer
from allegro.pipelines.pipeline_allegro import AllegroPipeline
from allegro.pipelines.pipeline_allegro_ti2v import AllegroTI2VPipeline
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel
from allegro.models.transformers.transformer_3d_allegro_ti2v import AllegroTransformerTI2V3DModel

script_directory = os.path.dirname(os.path.abspath(__file__))

class LoadAllegroModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default":"models/"},),
                "transformer_path": ("STRING", {"default":""},),
                "vae_path": ("STRING", {"default": ""}),
                "text_encoder_path": ("STRING", {"default": ""}),
                "tokenizer_path": ("STRING", {"default": ""}),
            }
        }
    CATEGORY = "Allegro"
    RETURN_TYPES = ("AllegroPIPE","VAE",)
    RETURN_NAMES = ("pipe","vae",)
    FUNCTION = "run"
    
    def run(self, model_path, transformer_path, vae_path, text_encoder_path, tokenizer_path):
        if not os.path.exists(transformer_path) or not os.path.exists(vae_path) or not os.path.exists(text_encoder_path) or not os.path.exists(text_encoder_path) or os.path.exists(tokenizer_path):
            if os.path.isabs(model_path) and os.path.exists(model_path):
                modelfullpath = model_path
            else:
                modelfullpath = os.path.join(script_directory, model_path)
                if not os.path.exists(modelfullpath):
                    modelfullpath = os.path.join(script_directory, "models/")
                    if not os.path.exists(modelfullpath):
                        from huggingface_hub import snapshot_download
                        snapshot_download("rhymes-ai/Allegro", local_dir=modelfullpath, local_dir_use_symlinks=False)
            transformer_path = os.path.join(modelfullpath, "transformer") if not os.path.exists(transformer_path) else transformer_path
            vae_path = os.path.join(modelfullpath, "vae") if not os.path.exists(vae_path) else vae_path
            text_encoder_path = os.path.join(modelfullpath, "text_encoder") if not os.path.exists(text_encoder_path) else text_encoder_path
            tokenizer_path = os.path.join(modelfullpath, "tokenizer") if not os.path.exists(tokenizer_path) else tokenizer_path
        pbar = ProgressBar(3)

        vae = AllegroAutoencoderKL3D.from_pretrained(vae_path, torch_dtype=torch.float32)
        vae.eval()
        if vae.device != model_management.vae_offload_device():
            vae = vae.to(model_management.vae_offload_device())
        pbar.update(1)
        
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
        text_encoder.eval()
        if text_encoder.device != model_management.text_encoder_offload_device():
            text_encoder = text_encoder.to(model_management.text_encoder_offload_device())
        pbar.update(1)
        
        scheduler = EulerAncestralDiscreteScheduler()
        transformer = AllegroTransformer3DModel.from_pretrained(transformer_path, torch_dtype=torch.bfloat16)
        transformer.eval()
        if transformer.device != model_management.unet_offload_device():
            transformer = transformer.to(model_management.unet_offload_device())
        pipe = AllegroPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer)
        pbar.update(1)
    
        return (pipe,vae,)

class AllegroTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("AllegroPIPE",),
                "positive_prompt": ("STRING",{"multiline": True, "dynamicPrompts": True, "default":""},),
            },
            "optional": {
                "negative_prompt": ("STRING",{"multiline": True, "dynamicPrompts": True, "default":""},),
            }
        }

    CATEGORY = "Allegro"
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("positive","negative",)
    FUNCTION = "run"
    
    def run(self, pipe, positive_prompt, negative_prompt):
        olddevice = pipe.text_encoder.device
        positive_prompt_template = "(masterpiece), (best quality), (ultra-detailed), (unwatermarked), {} emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"
        negative_prompt_default = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        positive_prompt = positive_prompt_template.format(positive_prompt.lower().strip())
        negative_prompt = negative_prompt if negative_prompt.strip() else negative_prompt_default
        
        if pipe.text_encoder.device != model_management.text_encoder_device():
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            try:
                pipe.text_encoder = pipe.text_encoder.to(device = model_management.text_encoder_device())
            except:
                pipe.text_encoder = pipe.text_encoder.to(device = torch.device('cpu'))

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            positive_prompt,
            True,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            device=pipe.text_encoder.device,
            clean_caption=True,
            max_sequence_length=512,
        )

        if pipe.text_encoder.device != model_management.text_encoder_offload_device():
            pipe.text_encoder = pipe.text_encoder.to(device = model_management.text_encoder_offload_device())
        
        return({"embeds": prompt_embeds,"attention_mask": prompt_attention_mask.bool()},{"embeds":negative_prompt_embeds,"attention_mask": negative_prompt_attention_mask.bool()})

class AllegroSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("AllegroPIPE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "frames": ("INT", {"default":88,}),
                "width": ("INT", {"default":1280,}),
                "height": ("INT", {"default":720,}),
                "steps": ("INT", {"default":100, "min": 1, "max": 200, "step": 1}),
                "guidance": ("FLOAT", {"default":7.5, "min": 0.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default":0}),
                "low_vram_mode": ("BOOLEAN", {"default":False}),
            },
            "optional": {
                "latents": ("LATENT",),
            }
        }
    CATEGORY = "Allegro"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "run"

    def run(self, pipe, positive, negative, frames, width, height, steps, guidance, seed, low_vram_mode, latents=None):
        latentsdevice = latents["samples"].device if latents and "samples" in latents and hasattr(latents["samples"],'device') else None
        latentsdtype = latents["samples"].dtype if latents and "samples" and "samples" in latents and hasattr(latents["samples"],'dtype') in latents else None
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype(device, supported_dtypes=[torch.bfloat16,])
        olddtype = pipe.transformer.dtype
        if pipe.transformer.device != device or pipe.transformer.dtype != dtype:
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            if low_vram_mode:
                pipe.transformer = pipe.transformer.to(dtype = dtype)
            else:
                pipe.transformer = pipe.transformer.to(device = device, dtype = dtype)
                
        if latents!=None and isinstance(latents, dict) and "samples" in latents and latents["samples"]!=None and (latents["samples"].device != device or latents["samples"].dtype != dtype):
            latents["samples"] = latents["samples"].to(device = device, dtype = dtype)
        if positive['embeds'].device != device or positive['embeds'].dtype != dtype:
            positive['embeds'] = positive['embeds'].to(device = device, dtype = dtype)
        if positive['attention_mask'].device != device or positive['attention_mask'].dtype != dtype:
            positive['attention_mask'] = positive['attention_mask'].to(device = device, dtype = dtype)
        if negative['embeds'].device != device or negative['embeds'].dtype != dtype:
            negative['embeds'] = negative['embeds'].to(device = device, dtype = dtype)
        if negative['attention_mask'].device != device or negative['attention_mask'].dtype != dtype:
            negative['attention_mask'] = negative['attention_mask'].to(device = device, dtype = dtype)
        
        try:
            setattr(pipe, 'load_device', device)
            setattr(pipe, 'model', typing.NewType('PseudoModel',typing.Generic))
            setattr(pipe.model, 'latent_format', comfy.latent_formats.SD15())
            callback = latent_preview.prepare_callback(pipe, steps)
        except:
            callback = None
        
        output = pipe(
            prompt = None,
            negative_prompt = None,
            prompt_embeds = positive['embeds'],
            prompt_attention_mask = positive['attention_mask'],
            negative_prompt_embeds = negative['embeds'],
            negative_prompt_attention_mask = negative['attention_mask'],
            num_frames=frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=512,
            generator = torch.Generator(device).manual_seed(seed),
            latents = latents['samples'].transpose(0, 1).unsqueeze(0) if latents!=None and "samples" in latents and latents["samples"]!=None else None,
            output_type = "latents",
            callback = lambda s,t,l:callback(s,l[0,:,random.randint(0, l.shape[-3]-1),:,:].unsqueeze(0),t,steps) if callback else None,
            device = device,
        ).video[0]
        
        if pipe.transformer.device != model_management.unet_offload_device() or pipe.transformer.dtype != olddtype:
            pipe.transformer = pipe.transformer.to(device = model_management.unet_offload_device(), dtype = olddtype)

        if latents!=None and isinstance(latents, dict) and "samples" in latents and latents["samples"]!=None and (latents["samples"].device != latentsdevice or latents["samples"].dtype != latentsdtype):
            latents["samples"] = latents["samples"].to(device = latentsdevice, dtype = latentsdtype)
        if output!=None and output.device != latentsdevice or output.dtype != latentsdtype:
            output = output.to(device = latentsdevice, dtype = latentsdtype)
        
        return ({"samples":output},)
    

class AllegroEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "batch": ("INT",{"default":1,"min":1,"max":16}),
            }
        }
    CATEGORY = "Allegro"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "run"
    
    def run(self, vae, images, batch):
        imagedevice = images.device
        imagedtype = images.dtype
        
        olddtype = vae.dtype
        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device, allowed_dtypes=[torch.bfloat16,])
        if vae.device != device or vae.dtype != dtype:
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            if hasattr(vae, 'encoder') and hasattr(vae, 'quant_conv'):
                vae.encoder = vae.encoder.to(device = device, dtype = dtype)
                vae.quant_conv = vae.quant_conv.to(device = device, dtype = dtype)
            else:
                vae = vae.to(device = device, dtype = dtype)
        
        if images.device != device or images.dtype != dtype:
            images = images.to(device = device, dtype = dtype)
        
        pbar = ProgressBar( (math.floor((images.shape[-3] - vae.kernel[0]//4) / (vae.stride[0]//4)) + 1) * (math.floor((images.shape[-2] - vae.kernel[1]//8) / (vae.stride[1]//8)) + 1) * (math.floor((images.shape[-1] - vae.kernel[2]//8) / (vae.stride[2]//8)) + 1) )
        latents = vae.encode(images.permute(0,3,1,2), local_batch_size=batch, callback=lambda s,t,l:pbar.update_absolute(s,total=t))

        if images.device != imagedevice or images.dtype != imagedtype:
            images = images.to(device = imagedevice, dtype = imagedtype)
        if latents.device != imagedevice or latents.dtype != imagedtype:
            latents = latents.to(device = imagedevice, dtype = imagedtype)
        
        if device != model_management.vae_offload_device() or dtype != olddtype:
            if hasattr(vae, 'encoder') and hasattr(vae, 'quant_conv'):
                vae.encoder = vae.encoder.to(device=model_management.vae_offload_device(), dtype=olddtype)
                vae.quant_conv = vae.quant_conv.to(device=model_management.vae_offload_device(), dtype=olddtype)
            else:
                vae = vae.to(device=model_management.vae_offload_device(), dtype=olddtype)

        return ({'samples':latents},)

class AllegroDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "vae": ("VAE",),
                "batch": ("INT",{"default":1,"min":1,"max":16}),
            }
        }
    CATEGORY = "Allegro"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"

    def run(self, vae, latents, batch):
        latentsdevice = latents["samples"].device
        latentsdtype = latents["samples"].dtype
        #sd = pipe.state_dict()
        #parameters = calculate_parameters(sd, 'first_stage_model.decoder.') + calculate_parameters(sd, 'first_stage_model.post_quant_conv.')
        olddtype = vae.dtype
        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device, allowed_dtypes=[torch.bfloat16,])
        if vae.device != device or vae.dtype != dtype:
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            if hasattr(vae, 'decoder') and hasattr(vae, 'post_quant_conv'):
                vae.decoder = vae.decoder.to(device = device, dtype = dtype)                
                vae.post_quant_conv = vae.post_quant_conv.to(device = device, dtype = dtype)
            else:
                vae = vae.to(device = device, dtype = dtype)

        if latents["samples"].device != device or latents["samples"].dtype != dtype:
            latents["samples"] = latents["samples"].to(device = device, dtype = dtype)
        
        pbar = ProgressBar( (math.floor((latents["samples"].shape[-3] - vae.kernel[0]//4) / (vae.stride[0]//4)) + 1) * (math.floor((latents["samples"].shape[-2] - vae.kernel[1]//8) / (vae.stride[1]//8)) + 1) * (math.floor((latents["samples"].shape[-1] - vae.kernel[2]//8) / (vae.stride[2]//8)) + 1) )
        if args.preview_method != latent_preview.LatentPreviewMethod.NoPreviews:
            callback = lambda s,t,l:pbar.update_absolute(s, total=t, preview=("JPEG", latent_preview.preview_to_image(l[0,:,random.randint(0,l.shape[-3]-1),:,:].permute(1,2,0)), args.preview_size))
        else:
            callback = lambda s,t,l:pbar.update_absolute(s, total=t)
        images = vae.decode(latents["samples"].unsqueeze(0) / vae.scale_factor, local_batch_size=batch, callback=callback).sample
        images = (images / 2.0 + 0.5).clamp(0,1).permute(0, 1, 3, 4, 2).squeeze(0).contiguous()

        if latents["samples"].device != latentsdevice or latents["samples"].dtype != latentsdtype:
            latents["samples"] = latents["samples"].to(device = latentsdevice, dtype = latentsdtype)
        if images.device != latentsdevice or images.dtype != torch.float32:
            images = images.to(device = latentsdevice, dtype = torch.float32)

        if device != model_management.vae_offload_device() or dtype != olddtype:
            if hasattr(vae, 'decoder') and hasattr(vae, 'post_quant_conv'):
                vae.decoder = vae.decoder.to(device = model_management.vae_offload_device(), dtype = olddtype)
                vae.post_quant_conv = vae.post_quant_conv.to(device = model_management.vae_offload_device(), dtype = olddtype)
            else:
                vae = vae.to(device = model_management.vae_offload_device(), dtype = olddtype)
        
        return (images,)

class LoadAllegroTI2VModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default":"ti2v_models/"},),
                "transformer_path": ("STRING", {"default":""},),
                "vae_path": ("STRING", {"default": ""}),
                "text_encoder_path": ("STRING", {"default": ""}),
                "tokenizer_path": ("STRING", {"default": ""}),
            }
        }
    CATEGORY = "Allegro"
    RETURN_TYPES = ("AllegroPIPE","VAE",)
    RETURN_NAMES = ("pipe","vae",)
    FUNCTION = "run"

    def run(self, model_path, transformer_path, vae_path, text_encoder_path, tokenizer_path):
        if not os.path.exists(transformer_path) or not os.path.exists(vae_path) or not os.path.exists(text_encoder_path) or not os.path.exists(text_encoder_path) or os.path.exists(tokenizer_path):
            if os.path.isabs(model_path) and os.path.exists(model_path):
                modelfullpath = model_path
            else:
                modelfullpath = os.path.join(script_directory, model_path)
                if not os.path.exists(modelfullpath):
                    modelfullpath = os.path.join(script_directory, "ti2v_models/")
                    if not os.path.exists(modelfullpath):
                        from huggingface_hub import snapshot_download
                        snapshot_download("rhymes-ai/Allegro-TI2V", local_dir=modelfullpath, local_dir_use_symlinks=False)
            transformer_path = os.path.join(modelfullpath, "transformer") if not os.path.exists(transformer_path) else transformer_path
            vae_path = os.path.join(modelfullpath, "vae") if not os.path.exists(vae_path) else vae_path
            text_encoder_path = os.path.join(modelfullpath, "text_encoder") if not os.path.exists(text_encoder_path) else text_encoder_path
            tokenizer_path = os.path.join(modelfullpath, "tokenizer") if not os.path.exists(tokenizer_path) else tokenizer_path
        pbar = ProgressBar(3)

        vae = AllegroAutoencoderKL3D.from_pretrained(vae_path, torch_dtype=torch.float32)
        vae.eval()
        if vae.device != model_management.vae_offload_device():
            vae = vae.to(model_management.vae_offload_device())
        pbar.update(1)

        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
        text_encoder.eval()
        if text_encoder.device != model_management.text_encoder_offload_device():
            text_encoder = text_encoder.to(model_management.text_encoder_offload_device())
        pbar.update(1)

        scheduler = EulerAncestralDiscreteScheduler()
        transformer = AllegroTransformerTI2V3DModel.from_pretrained(transformer_path, torch_dtype=torch.bfloat16)
        transformer.eval()
        if transformer.device != model_management.unet_offload_device():
            transformer = transformer.to(model_management.unet_offload_device())
        pipe = AllegroTI2VPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer)
        pbar.update(1)

        return (pipe,vae,)

class AllegroTI2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("AllegroPIPE",),
                "ref_latents": ("LATENT",),
                "ref_masks": ("MASK",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "frames": ("INT", {"default":88,}),
                "width": ("INT", {"default":1280,}),
                "height": ("INT", {"default":720,}),
                "steps": ("INT", {"default":100, "min": 1, "max": 200, "step": 1}),
                "guidance": ("FLOAT", {"default":8, "min": 0.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default":0}),
                "low_vram_mode": ("BOOLEAN", {"default":False}),
            },
        }
    CATEGORY = "Allegro"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "run"

    def run(self, pipe, ref_latents, ref_masks, positive, negative, frames, width, height, steps, guidance, seed, low_vram_mode):
        latentsdevice = ref_latents["samples"].device if ref_latents and "samples" in ref_latents and hasattr(ref_latents["samples"],'device') else None
        latentsdtype = ref_latents["samples"].dtype if ref_latents and "samples" in ref_latents and hasattr(ref_latents["samples"],'dtype') else None
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype(device, supported_dtypes=[torch.bfloat16,])
        olddtype = pipe.transformer.dtype
        if pipe.transformer.device != device or pipe.transformer.dtype != dtype:
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            if low_vram_mode:
                pipe.transformer = pipe.transformer.to(dtype = dtype)
            else:
                pipe.transformer = pipe.transformer.to(device = device, dtype = dtype)

        if ref_latents!=None and isinstance(ref_latents, dict) and "samples" in ref_latents and ref_latents["samples"]!=None and (ref_latents["samples"].device != device or ref_latents["samples"].dtype != dtype):
            ref_latents["samples"] = ref_latents["samples"].to(device = device, dtype = dtype)
        if ref_masks!=None and isinstance(ref_masks, torch.Tensor) and (ref_masks.device != device or ref_masks.dtype != dtype):
            ref_masks = ref_masks.to(device = device, dtype = dtype)
        if positive['embeds'].device != device or positive['embeds'].dtype != dtype:
            positive['embeds'] = positive['embeds'].to(device = device, dtype = dtype)
        if positive['attention_mask'].device != device or positive['attention_mask'].dtype != dtype:
            positive['attention_mask'] = positive['attention_mask'].to(device = device, dtype = dtype)
        if negative['embeds'].device != device or negative['embeds'].dtype != dtype:
            negative['embeds'] = negative['embeds'].to(device = device, dtype = dtype)
        if negative['attention_mask'].device != device or negative['attention_mask'].dtype != dtype:
            negative['attention_mask'] = negative['attention_mask'].to(device = device, dtype = dtype)

        try:
            setattr(pipe, 'load_device', device)
            setattr(pipe, 'model', typing.NewType('PseudoModel',typing.Generic))
            setattr(pipe.model, 'latent_format', comfy.latent_formats.SD15())
            callback = latent_preview.prepare_callback(pipe, steps)
        except:
            callback = None

        output = pipe(
            prompt = None,
            negative_prompt = None,
            prompt_embeds = positive['embeds'],
            prompt_attention_mask = positive['attention_mask'],
            negative_prompt_embeds = negative['embeds'],
            negative_prompt_attention_mask = negative['attention_mask'],
            num_frames=frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=512,
            generator = torch.Generator(device).manual_seed(seed),
            latents = None,
            output_type = "latents",
            callback = lambda s,t,l:callback(s,l[0,:,random.randint(0, l.shape[-3]-1),:,:].unsqueeze(0),t,steps) if callback else None,
            device = device,
            conditional_images = None,
            conditional_images_indices = None,
            masked_video = ref_latents["samples"],
            mask = ref_masks,
        ).video[0]

        if pipe.transformer.device != model_management.unet_offload_device() or pipe.transformer.dtype != olddtype:
            pipe.transformer = pipe.transformer.to(device = model_management.unet_offload_device(), dtype = olddtype)

        if ref_latents!=None and isinstance(ref_latents, dict) and "samples" in ref_latents and ref_latents["samples"]!=None and (ref_latents["samples"].device != latentsdevice or ref_latents["samples"].dtype != latentsdtype):
            ref_latents["samples"] = ref_latents["samples"].to(device = latentsdevice, dtype = latentsdtype)
        if ref_masks!=None and isinstance(ref_latents, torch.Tensor) and (ref_masks.device != latentsdevice or ref_masks.dtype != latentsdtype):
            ref_masks = ref_masks.to(device = latentsdevice, dtype = latentsdtype)
        if output!=None and output.device != latentsdevice or output.dtype != latentsdtype:
            output = output.to(device = latentsdevice, dtype = latentsdtype)

        return ({"samples":output},)

class AllegroTI2VEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("AllegroPIPE",),
                "ref_images": ("IMAGE",),
                "frames": ("INT",{"default":88}),
                "indices": ("STRING",{"default":""}),
                "batch": ("INT",{"default":1,"min":1,"max":16}),
                "seed": ("INT", {"default":0}),
            },
        }
    CATEGORY = "Allegro"
    RETURN_TYPES = ("LATENT","MASK","INT","INT","INT",)
    RETURN_NAMES = ("ref_latents","ref_masks","frames","width","height")
    FUNCTION = "run"

    def run(self, pipe, ref_images, frames, indices, batch, seed):
        imagedevice = ref_images.device
        imagedtype = ref_images.dtype

        vae = pipe.vae
        olddtype = vae.dtype
        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device, allowed_dtypes=[torch.bfloat16,])
        if vae.device != device or vae.dtype != dtype:
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            if hasattr(vae, 'encoder') and hasattr(vae, 'quant_conv'):
                vae.encoder = vae.encoder.to(device = device, dtype = dtype)
                vae.quant_conv = vae.quant_conv.to(device = device, dtype = dtype)
            else:
                vae = vae.to(device = device, dtype = dtype)

        if ref_images.shape[0] > frames:
            ref_images = ref_images[[int(round(i*(ref_images.shape[0]-1)/(frames-1))) for i in range(frames)]:,:,:]
            ref_images_indices = list(range(frames))
        else:
            ref_images_indices = [0] if ref_images.shape[0]==1 else [0, -1] if ref_images.shape[0]==2 else [int(round(i*(frames-1)/(ref_images.shape[0]-1))) for i in range(ref_images.shape[0])]
            if indices:
                for k, token in enumerate(indices.replace(' ','').split(',')):
                    if token and (str.isdigit(token) or token[0]=='-' and str.isdigit(token[1:])):
                        index = int(token)
                        if -frames < index < frames:
                            ref_images_indices[k] = index
        if ref_images.device != device or ref_images.dtype != dtype:
            ref_images = ref_images.to(device = device, dtype = dtype)

        pbar = ProgressBar( (math.floor((ref_images.shape[-4] - vae.kernel[0]//4) / (vae.stride[0]//4)) + 1) * (math.floor((ref_images.shape[-3] - vae.kernel[1]//8) / (vae.stride[1]//8)) + 1) * (math.floor((ref_images.shape[-2] - vae.kernel[2]//8) / (vae.stride[2]//8)) + 1) )
        mask, masked_video = pipe.prepare_mask_masked_video(
            conditional_images = ref_images.permute(0,3,1,2), #T,H,W,C->T,C,H,W
            conditional_images_indices = ref_images_indices,
            num_frames = frames,
            batch_size = 1,
            height = ref_images.shape[-3],
            width = ref_images.shape[-2],
            num_images_per_prompt=1,
            generator=torch.Generator(device).manual_seed(seed),
            device=device,
            callback=lambda s,t,l:pbar.update_absolute(s,total=t),
            local_batch_size=batch,
        )
        #latents = vae.encode(images, callback=lambda s,t,l:pbar.update_absolute(s,total=t))

        if ref_images.device != imagedevice or ref_images.dtype != imagedtype:
            ref_images = ref_images.to(device = imagedevice, dtype = imagedtype)
        if masked_video.device != imagedevice or masked_video.dtype != imagedtype:
            masked_video = masked_video.to(device = imagedevice, dtype = imagedtype)
        if mask.device != imagedevice or mask.dtype != imagedtype:
            mask = mask.to(device = imagedevice, dtype = imagedtype)

        if device != model_management.vae_offload_device() or dtype != olddtype:
            if hasattr(vae, 'encoder') and hasattr(vae, 'quant_conv'):
                vae.encoder = vae.encoder.to(device=model_management.vae_offload_device(), dtype=olddtype)
                vae.quant_conv = vae.quant_conv.to(device=model_management.vae_offload_device(), dtype=olddtype)
            else:
                vae = vae.to(device=model_management.vae_offload_device(), dtype=olddtype)

        return ({'samples':masked_video}, mask, frames, ref_images.shape[-2], ref_images.shape[-3])

NODE_CLASS_MAPPINGS = {
    "LoadAllegroModel":LoadAllegroModel,
    "AllegroSampler":AllegroSampler,
    "AllegroDecoder":AllegroDecoder,
    "AllegroEncoder":AllegroEncoder,
    "AllegroTextEncoder":AllegroTextEncoder,
    "AllegroTI2VSampler":AllegroTI2VSampler,
    "AllegroTI2VEncoder":AllegroTI2VEncoder,
    "LoadAllegroTI2VModel":LoadAllegroTI2VModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAllegroModel":"(Down)Load Allegro Model",
    "AllegroSampler":"Allegro Sampler",
    "AllegroDecoder":"Allegro Decoder",
    "AllegroEncoder":"Allegro Encoder",
    "AllegroTextEncoder":"Allegro Text Encoder",
    "AllegroTI2VSampler":"Allegro TextImage2Video Sampler",
    "AllegroTI2VEncoder":"Allegro TextImage2Video Encoder",
    "LoadAllegroTI2VModel":"(Down)Load Allegro TextImage2Video Model",
}
