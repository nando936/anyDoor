"""Test TrOCR on handwritten image"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: python test_trocr.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if image_path.startswith('\\\\'):
    image_path = image_path.replace('\\', '/')

print("Loading TrOCR model (this may take a moment)...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

print(f"Processing image: {image_path}")
image = Image.open(image_path).convert('RGB')

pixel_values = processor(image, return_tensors='pt').pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n=== TrOCR Output ===")
print(text)
print("=== End TrOCR Output ===")
