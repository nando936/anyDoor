"""Debug: Show how OCR breaks up headers"""
import os
import sys
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv('GOOGLE_VISION_API_KEY')

def call_vision_api(image_path):
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

    url = f"https://vision.googleapis.com/v1/images:annotate?key={google_api_key}"
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

print("Headers and related text elements:\n")
for ann in annotations[1:]:
    text = ann['description']
    text_lower = text.lower()
    y = sum(v.get('y', 0) for v in ann['boundingPoly']['vertices']) / 4

    # Show elements that might be headers
    if any(word in text_lower for word in ['pantry', 'living', 'room', 'white', 'oak', 'drawer', 'draver', 'door', 'dour']):
        print(f"y={int(y):4d}  '{text}'")
