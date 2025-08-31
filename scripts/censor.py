import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
from modules import scripts

# NSFW classifier (general, works for anime too in combination with prompt keywords)
NSFW_MODEL_ID = "prithivMLmods/vit-mini-explicit-content"

# List of explicit anime keywords for prompt-based filtering
NSFW_KEYWORDS = [
    "nude", "nsfw", "pussy", "breasts", "hentai", "sex", "nipples",
    "1girl naked", "1boy naked", "naked", "erotic", "cum", "dick",
    "pussy", "lewd", "cumming", "sex", "porn"
]

# Initialize global model
safety_feature_extractor = None
safety_model = None

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]

def check_safety(images_np, prompt="", disable_safety=False, sensitivity=0.5):
    """
    images_np: numpy array [B,H,W,C] range 0-1
    prompt: string prompt for additional keyword check
    disable_safety: bypass NSFW check (NSFW channel)
    sensitivity: float [0-1], threshold for NSFW classifier
    """
    global safety_feature_extractor, safety_model

    if disable_safety:
        return [False] * len(images_np)

    # Keyword-based detection
    prompt_flag = any(kw.lower() in prompt.lower() for kw in NSFW_KEYWORDS)

    # Initialize classifier
    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(NSFW_MODEL_ID)
        safety_model = AutoModelForImageClassification.from_pretrained(NSFW_MODEL_ID)

    nsfw_flags = []
    for img in images_np:
        # Convert to PIL and preprocess
        pil_img = Image.fromarray((img*255).astype(np.uint8))
        inputs = safety_feature_extractor(images=pil_img, return_tensors="pt")
        outputs = safety_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        nsfw_score = probs[0][1].item()  # Assuming class 1 = NSFW
        nsfw_flags.append(nsfw_score > sensitivity or prompt_flag)

    return nsfw_flags

def censor_batch(x, prompt="", disable_safety=False, sensitivity=0.5, replacement="blur"):
    """
    x: torch tensor [B,C,H,W] range 0-1
    """
    x_np = x.cpu().permute(0,2,3,1).numpy()
    nsfw_flags = check_safety(x_np, prompt=prompt, disable_safety=disable_safety, sensitivity=sensitivity)

    for i, flag in enumerate(nsfw_flags):
        if flag:
            if replacement == "blur":
                img = Image.fromarray((x_np[i]*255).astype(np.uint8))
                # Stronger blur
                blurred = img.filter(ImageFilter.GaussianBlur(radius=30))

                # Draw bold red "NSFW" text
                draw = ImageDraw.Draw(blurred)
                try:
                    font = ImageFont.truetype("arial.ttf", max(40, blurred.width // 10))
                except:
                    font = ImageFont.load_default()
                text = "NSFW"
                text_width, text_height = draw.textsize(text, font=font)
                x_pos = (blurred.width - text_width) // 2
                y_pos = (blurred.height - text_height) // 2

                # Make text bolder by drawing multiple offsets
                offsets = [(-1,-1),(1,-1),(-1,1),(1,1),(0,0)]
                for ox, oy in offsets:
                    draw.text((x_pos+ox, y_pos+oy), text, fill=(255,0,0), font=font)

                x_np[i] = np.array(blurred)/255.0
            elif isinstance(replacement, (Image.Image, np.ndarray, torch.Tensor)):
                if isinstance(replacement, torch.Tensor):
                    x_np[i] = replacement.cpu().permute(1,2,0).numpy()
                elif isinstance(replacement, Image.Image):
                    x_np[i] = np.array(replacement)/255.0
                elif isinstance(replacement, np.ndarray):
                    x_np[i] = replacement

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

        images[:] = censor_batch(images, prompt=prompt, disable_safety=disable_safety, sensitivity=sensitivity, replacement="blur")[:]




