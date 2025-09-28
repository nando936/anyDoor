"""
Debug raw OCR output for page 1
"""
import base64
import requests
import os
import json

api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_1.png"

# Read and encode image
with open(image_path, 'rb') as f:
    image_content = base64.b64encode(f.read()).decode('utf-8')

# Call Vision API
url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
request = {
    "requests": [{
        "image": {"content": image_content},
        "features": [{"type": "TEXT_DETECTION"}]
    }]
}

response = requests.post(url, json=request)

if response.status_code == 200:
    result = response.json()
    if 'responses' in result and result['responses']:
        annotations = result['responses'][0].get('textAnnotations', [])

        # First annotation is full text
        print("FULL TEXT FROM OCR:")
        print("-" * 60)
        if annotations:
            print(annotations[0]['description'])
            print("-" * 60)

            # Look for items around the problem area
            print("\nINDIVIDUAL ITEMS CONTAINING '20' or '34' or '3/16':")
            for ann in annotations[1:]:
                text = ann['description']
                if '20' in text or '34' in text or '3/16' in text or '.' in text:
                    vertices = ann['boundingPoly']['vertices']
                    x = sum(v.get('x', 0) for v in vertices) / 4
                    y = sum(v.get('y', 0) for v in vertices) / 4
                    print(f"  '{text}' at ({x:.0f}, {y:.0f})")
else:
    print(f"Error: {response.status_code}")
    print(response.text)