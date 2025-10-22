"""OCR a single image"""
import sys
import base64
import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_VISION_API_KEY')

if len(sys.argv) < 2:
    print("Usage: python ocr_single.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if image_path.startswith('\\\\'):
    image_path = image_path.replace('\\', '/')

with open(image_path, 'rb') as f:
    image_data = base64.standard_b64encode(f.read()).decode('utf-8')

url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
request_body = {
    'requests': [{
        'image': {'content': image_data},
        'features': [{'type': 'DOCUMENT_TEXT_DETECTION'}]
    }]
}

response = requests.post(url, json=request_body)
result = response.json()

if 'responses' in result and len(result['responses']) > 0:
    annotations = result['responses'][0].get('textAnnotations', [])
    if annotations:
        print(annotations[0].get('description', ''))
