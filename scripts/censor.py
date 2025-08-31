import torch
import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from modules import scripts

import requests
from io import BytesIO

# ----- Settings -----
safety_model_id = "CompVis/stable-diffusion-safety-checker"
NSFW_THRESHOLD = 0.95 # set higher to be less sensitive
CUSTOM_IMAGE_URL = "https://cdn.discordapp.com/attachments/1294630750900977695/1411673768773816370/ewapolmkwgrniom.jpg?ex=68b58326&is=68b431a6&hm=9a6f7782b24246d0650a7ca25d687782a876efc0588fd07eedd7a2921104e6a3&"

# ----- Global variables -----
safety_feature_extractor = None
safety_checker = None

# Load custom image once
response = requests.get(CUSTOM_IMAGE_URL)
custom_image_pil = Image.open(BytesIO(response.content)).convert("RGB")

# ----- Helper functions -----
def numpy_to_pil(images):
    """Convert numpy array(s) to PIL Image(s)."""
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def check_safety(x_image, threshold=NSFW_THRESHOLD):
    """Run safety checker and return checked images + NSFW flags."""
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    inputs = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(
        images=x_image,
        clip_input=inputs.pixel_values
    )

    # Apply threshold to reduce sensitivity
    if isinstance(has_nsfw_concept, torch.Tensor):
        has_nsfw_concept = has_nsfw_concept.float() > threshold

    return x_checked_image, has_nsfw_concept

def censor_batch(x):
    """Replace flagged images with custom image."""
    x_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, has_nsfw_concept = check_safety(x_numpy)

    for i, flagged in enumerate(has_nsfw_concept):
        if flagged:
            # Resize custom image to match flagged image
            custom_resized = np.array(custom_image_pil.resize(
                (x_numpy.shape[2], x_numpy.shape[1])
            ), dtype=np.float32) / 255.0
            x_checked_image[i] = custom_resized

    return torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

# ----- Script integration -----
class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW Check with Custom Replacement"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs['images']
        images[:] = censor_batch(images)[:]

