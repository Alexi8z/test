import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFilter
import numpy as np
from modules import scripts

class CheckpointNsfwScript(scripts.Script):
    def title(self):
        return "Checkpoint NSFW Check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        """
        images: torch tensor [B,C,H,W] 0-1
        p.allow_nsfw: if True, bypass NSFW check (NSFW channel)
        p.nsfw_sensitivity: float 0-1, higher = more strict
        """
        images = kwargs['images']

        # Bypass and sensitivity
        disable_safety = getattr(p, 'allow_nsfw', False)
        sensitivity = getattr(p, 'nsfw_sensitivity', 0.6)

        # Use your NSFW-trained checkpoint
        nsfw_model_path = "E:\sd.webui\webui\models\Stable-diffusion\JANKUV5NSFWTrainedNoobai_v50.safetensors"

        # Load the pipeline once
        pipe = StableDiffusionPipeline.from_ckpt(nsfw_model_path, torch_dtype=torch.float16)
        pipe.to("cuda")  # or "cpu"

        images_np = images.cpu().permute(0,2,3,1).numpy()

        nsfw_flags = []
        for img in images_np:
            if disable_safety:
                nsfw_flags.append(False)
                continue

            # Convert image to latent for your NSFW checkpoint
            latent = pipe.vae.encode(torch.tensor(img*2-1).permute(2,0,1).unsqueeze(0).to("cuda"))
            # Generate prediction for "NSFW" vs "SFW"
            score = pipe(prompt="nsfw content", image=latent, guidance_scale=1.0).safety_score
            nsfw_flags.append(score > sensitivity)

        # Apply blur on flagged images
        for i, flag in enumerate(nsfw_flags):
            if flag:
                img = Image.fromarray((images_np[i]*255).astype(np.uint8))
                blurred = img.filter(ImageFilter.GaussianBlur(radius=15))
                images_np[i] = np.array(blurred)/255.0

        # Convert back to torch
        images[:] = torch.from_numpy(images_np).permute(0,3,1,2)
