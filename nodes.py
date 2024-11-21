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
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel

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
        positive_prompt = positive_prompt_template.format(positive_prompt.strip())
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
            callback = lambda s,t,l:callback(s,l[0,:,random.randint(0, l.shape[-3]),:,:].unsqueeze(0),t,steps) if callback else None,
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
            }
        }
    CATEGORY = "Allegro"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "run"
    
    def run(self, vae, images):
        imagedevice = images.device
        imagedtype = images.dtype
        
        olddtype = vae.dtype
        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device, allowed_dtypes=[torch.bfloat32,])
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
        
        pbar = ProgressBar( (math.floor((latents["samples"].shape[-3] - vae.kernel[0]//4) / (vae.stride[0]//4)) + 1) * (math.floor((latents["samples"].shape[-2] - vae.kernel[1]//8) / (vae.stride[1]//8)) + 1) * (math.floor((latents["samples"].shape[-1] - vae.kernel[2]//8) / (vae.stride[2]//8)) + 1) )
        latents = vae.encode(images, callback=lambda s,t,l:pbar.update_absolute(s,total=t))

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
            }
        }
    CATEGORY = "Allegro"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"

    def run(self, vae, latents):
        latentsdevice = latents["samples"].device
        latentsdtype = latents["samples"].dtype
        #sd = pipe.state_dict()
        #parameters = calculate_parameters(sd, 'first_stage_model.decoder.') + calculate_parameters(sd, 'first_stage_model.post_quant_conv.')
        olddtype = vae.dtype
        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device, allowed_dtypes=[torch.float32,])
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
            callback = lambda s,t,l:pbar.update_absolute(s, total=t, preview=("JPEG", latent_preview.preview_to_image(l[0,:,random.randint(0,l.shape[-3]),:,:].permute(1,2,0)), args.preview_size))
        else:
            callback = lambda s,t,l:pbar.update_absolute(s, total=t)
        images = vae.decode(latents["samples"].unsqueeze(0) / vae.scale_factor, callback=callback).sample
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

NODE_CLASS_MAPPINGS = {
    "LoadAllegroModel":LoadAllegroModel,
    "AllegroSampler":AllegroSampler,
    "AllegroDecoder":AllegroDecoder,
    "AllegroEncoder":AllegroEncoder,
    "AllegroTextEncoder":AllegroTextEncoder,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAllegroModel":"(Down)Load Allegro Model",
    "AllegroSampler":"Allegro Sampler",
    "AllegroDecoder":"Allegro Decoder",
    "AllegroEncoder":"Allegro Encoder",
    "AllegroTextEncoder":"Allegro Text Encoder",
}
