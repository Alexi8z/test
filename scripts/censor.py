import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image, ImageFilter
from modules import scripts, shared
import numpy as np

# Anime-focused NSFW model
safety_model_id = "giacomoarienti/nsfw-classifier"

safety_feature_extractor = None
safety_checker = None

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]

def check_safety(x_image, disable_safety=False, sensitivity=0.6):
    global safety_feature_extractor, safety_checker
    if disable_safety:
        return x_image, [False]*len(x_image)

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(
        images=x_image,
        clip_input=safety_checker_input.pixel_values
    )

    # Apply sensitivity threshold (0–1)
    has_nsfw_concept = [bool(hc > sensitivity) for hc in has_nsfw_concept]

    return x_checked_image, has_nsfw_concept

def censor_batch(x, disable_safety=False, sensitivity=0.6):
    """
    x: torch tensor [B,C,H,W] range 0-1
    NSFW images are replaced with a blurred version of themselves.
    """
    x_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked, nsfw_flags = check_safety(x_numpy, disable_safety=disable_safety, sensitivity=sensitivity)

    for i, flag in enumerate(nsfw_flags):
        if flag:
            img = Image.fromarray((x_checked[i]*255).astype(np.uint8))
            blurred = img.filter(ImageFilter.GaussianBlur(radius=15))
            x_checked[i] = np.array(blurred)/255.0

    return torch.from_numpy(x_checked).permute(0, 3, 1, 2)

class AnimeNsfwCheckScript(scripts.Script):
    def title(self):
        return "Anime NSFW Check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs['images']

        # NSFW channel bypass and sensitivity control
        disable_safety = getattr(p, 'allow_nsfw', False)
        sensitivity = getattr(p, 'nsfw_sensitivity', 0.6)

        images[:] = censor_batch(images, disable_safety=disable_safety, sensitivity=sensitivity)[:]


