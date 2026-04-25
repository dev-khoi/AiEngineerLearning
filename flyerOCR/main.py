from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch
import os
from accelerate import Accelerator

# path to flyer_1
# a loop is needed if perform OCR on multiple pages of the flyer
base_dir = os.path.dirname(__file__)
img_path = os.path.join(base_dir, "flyerImages", "flyer_1.png")

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
PROMPT = "Extract text from the image"
# device = Accelerator().device
# not needed because of device_map="auto" in model loading, which automatically handles device placement

# Load model and processor with trust_remote_code
# Handles non-text inputs OR multi-modal inputs
# AutoTokenizer is for text-based
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

# Load and process image
image = Image.open(img_path).convert("RGB")

inputs = processor(images=image, text=PROMPT, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024)

text = processor.decode(outputs[0], skip_special_tokens=True)

print(text)
