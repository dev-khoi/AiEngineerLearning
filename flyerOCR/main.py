from transformers import AutoModel, AutoProcessor, pipeline
from PIL import Image
import torch
import os
from accelerate import Accelerator

device = Accelerator().device

# path to flyer_1
# a loop is needed if perform OCR on multiple pages of the flyer
base_dir = os.path.dirname(__file__)
img_path = os.path.join(base_dir, "flyerImages", "flyer_1.png")
image = Image.open(img_path).convert("RGB")

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
PROMPT = "Extract text from the image"
# device = Accelerator().device
# not needed because of device_map="auto" in model loading, which automatically handles device placement

# Load model and processor with trust_remote_code
# Handles non-text inputs OR multi-modal inputs
# AutoTokenizer is for text-based

pipe = pipeline("image-to-text", model=MODEL_NAME, trust_remote_code=True)

result = pipe(image=image, text="Describe the image", max_new_tokens=50)

print(result)

text = result[0]["generated_text"]
