from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import torch

# Load processor and model
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")

# Load your image
image = Image.open("image-captioner/tabby-cat.jpg").convert("RGB")

# Preprocess image
inputs = processor(images=image, return_tensors="pt")

# Generate caption
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print("Caption:", caption)