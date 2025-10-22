"""Debug OCR to see all text elements and their positions"""
import os
import sys
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_VISION_API_KEY')

def call_vision_api(image_path):
    """Call Google Vision API for text detection"""
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request_body = {
        "requests": [{
            "image": {"content": image_data},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, json=request_body)
    result = response.json()
    return result['responses'][0].get('textAnnotations', [])

image_path = "//vmware-host/Shared Folders/D/OneDrive/customers/raised panel/handwritten orders/page1.jpg"
annotations = call_vision_api(image_path)

print(f"Total annotations: {len(annotations)}")
print("\nFirst 50 individual text elements:")
for i, ann in enumerate(annotations[1:51], 1):
    text = ann.get('description', '')
    vertices = ann['boundingPoly']['vertices']
    y = sum(v.get('y', 0) for v in vertices) / len(vertices)

    # Check if looks like measurement
    has_dash = '-' in text
    has_x = 'x' in text.lower() or 'X' in text
    starts_digit = text[0].isdigit() if text else False

    print(f"{i:3d}. y={int(y):4d} '{text:20s}' dash={has_dash} x={has_x} digit={starts_digit}")
