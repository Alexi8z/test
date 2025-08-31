import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
from modules import scripts

# Default SD safety checker
safety_model_id = "CompVis/stable-diffusion-safety-checker"

# Anime NSFW prompt keywords
NSFW_KEYWORDS = [
    "nude", "nsfw", "pussy", "breasts", "hentai", "sex", "nipples",
    "1girl naked", "1boy naked", "naked", "erotic", "cum", "dick",
    "pussy", "lewd", "cumming", "sex", "porn", "open jacket", "no top",
    "no panties", "bare chest"
]

safety_feature_extractor = None
safety_checker = None

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]

def check_safety(x_image, prompt="", disable_safety=False, sensitivity=0.5):
    """
    x_image: numpy array batch HWC [0-1]
    prompt: string prompt for keyword checks
    disable_safety: True if NSFW filtering should be skipped (e.g. NSFW channel)
    sensitivity: float [0-1], threshold for safety checker
    """
    global safety_feature_extractor, safety_checker
    if disable_safety:
        return [False]*len(x_image)

    # Keyword-based detection
    prompt_flag = any(kw.lower() in prompt.lower() for kw in NSFW_KEYWORDS)

    # Initialize safety checker
    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    inputs = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=inputs.pixel_values)

    # Apply sensitivity threshold and combine with prompt keyword check
    has_nsfw_concept = [bool(hc > sensitivity) or prompt_flag for hc in has_nsfw_concept]

    return x_checked_image, has_nsfw_concept

def censor_batch(x, prompt="", disable_safety=False, sensitivity=0.5):
    """
    x: torch tensor [B,C,H,W] range 0-1
    Replaces NSFW images with strong blur + bold red NSFW text
    """
    x_np = x.cpu().permute(0,2,3,1).numpy()
    x_checked, nsfw_flags = check_safety(x_np, prompt=prompt, disable_safety=disable_safety, sensitivity=sensitivity)

    for i, flag in enumerate(nsfw_flags):
        if flag:
            img = Image.fromarray((x_checked[i]*255).astype(np.uint8))
            # Strong blur
            blurred = img.filter(ImageFilter.GaussianBlur(radius=40))

            # Draw bold red NSFW text
            draw = ImageDraw.Draw(blurred)
            try:
                font_size = max(80, blurred.width // 5)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            text = "NSFW"
            text_width, text_height = draw.textsize(text, font=font)
            x_pos = (blurred.width - text_width) // 2
            y_pos = (blurred.height - text_height) // 2

            # Bold by drawing multiple offsets
            offsets = [(-2,-2),(2,-2),(-2,2),(2,2),(0,0)]
            for ox, oy in offsets:
                draw.text((x_pos+ox, y_pos+oy), text, fill=(255,0,0), font=font)

            x_np[i] = np.array(blurred)/255.0

    return torch.from_numpy(x_np).permute(0,3,1,2)

class AnimeNsfwCheckScript(scripts.Script):
    def title(self):
        return "Anime NSFW Check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs['images']

        # NSFW toggle and sensitivity
        disable_safety = getattr(p, 'allow_nsfw', False)
        sensitivity = getattr(p, 'nsfw_sensitivity', 0.5)
        prompt = getattr(p, 'prompt', "")

        images[:] = censor_batch(images, prompt=prompt, disable_safety=disable_safety, sensitivity=sensitivity)[:]
