"""Test OCR on header area to see field layout"""
import cv2
import os
import base64
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv('GOOGLE_VISION_API_KEY')

# Read image
image_path = '//vmware-host/Shared Folders/suarez group qb/customers/raised panel/True Custom/all_pages/page_1.png'
if image_path.startswith('\\\\'):
    image_path = image_path.replace('\\', '/')

image = cv2.imread(image_path)

# Crop header area (approx top 600 pixels to include all fields)
header_crop = image[0:600, :]

# Encode image
_, buffer = cv2.imencode('.png', header_crop)
image_b64 = base64.b64encode(buffer).decode('utf-8')

# Call Vision API
url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
payload = {
    'requests': [{
        'image': {'content': image_b64},
        'features': [{'type': 'TEXT_DETECTION'}]
    }]
}

response = requests.post(url, json=payload)
result = response.json()

if 'responses' in result and result['responses']:
    annotations = result['responses'][0].get('textAnnotations', [])
    if annotations:
        full_text = annotations[0]['description']
        print('=== HEADER OCR TEXT ===')
        print(full_text)
        print('\n=== LINES ===')
        for i, line in enumerate(full_text.split('\n')):
            print(f'{i}: {line}')
