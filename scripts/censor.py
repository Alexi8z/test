import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
from modules import scripts

# SD Safety checker
safety_model_id = "CompVis/stable-diffusion-safety-checker"

# NSFW keyword list
NSFW_KEYWORDS = [
    "nude", "nsfw", "pussy", "breasts", "hentai", "sex", "nipples",
    "1girl naked", "1boy naked", "naked", "erotic", "cum", "dick",
    "lewd", "cumming", "porn"
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
    Returns NSFW flags but does NOT replace the image with black.
    """
    global safety_feature_extractor, safety_checker
    if disable_safety:
        return [False]*len(x_image)

    # Prompt keyword detection
    prompt_flag = any(kw.lower() in prompt.lower() for kw in NSFW_KEYWORDS)

    # Initialize model if needed
    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    inputs = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    _, has_nsfw_concept = safety_checker(images=x_image, clip_input=inputs.pixel_values)

    # Combine keyword flag
    has_nsfw_concept = [bool(hc or prompt_flag) for hc in has_nsfw_concept]

    return has_nsfw_concept

def censor_batch(x, prompt="", disable_safety=False, sensitivity=0.5):
    """
    Blur NSFW images and overlay red "NSFW" text.
    """
    x_np = x.cpu().permute(0,2,3,1).numpy()
    nsfw_flags = check_safety(x_np, prompt=prompt, disable_safety=disable_safety, sensitivity=sensitivity)

    for i, flag in enumerate(nsfw_flags):
        if flag:
            img = Image.fromarray((x_np[i]*255).astype(np.uint8))
            blurred = img.filter(ImageFilter.GaussianBlur(radius=40))

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

            # Draw bold by multiple offsets
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

        disable_safety = getattr(p, 'allow_nsfw', False)
        sensitivity = getattr(p, 'nsfw_sensitivity', 0.5)
        prompt = getattr(p, 'prompt', "")

        images[:] = censor_batch(images, prompt=prompt, disable_safety=disable_safety, sensitivity=sensitivity)[:]
