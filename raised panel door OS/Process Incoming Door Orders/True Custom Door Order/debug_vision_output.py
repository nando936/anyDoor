"""
Debug script to see exactly what Vision API detects
"""
import os
import sys
import json
import base64
import requests
import cv2
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_VISION_API_KEY')

image_path = sys.argv[1] if len(sys.argv) > 1 else None
if not image_path:
    print("Usage: python debug_vision_output.py <image_path>")
    sys.exit(1)

# Convert path
if image_path.startswith('\\\\'):
    image_path = image_path.replace('\\', '/')

# Read image
image = cv2.imread(image_path)
_, buffer = cv2.imencode('.png', image)
content = base64.b64encode(buffer).decode('utf-8')

# Call Vision API
url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
request = {
    "requests": [{
        "image": {"content": content},
        "features": [{"type": "TEXT_DETECTION"}]
    }]
}

response = requests.post(url, json=request)
result = response.json()
annotations = result['responses'][0].get('textAnnotations', [])

# Get text elements with positions
text_elements = []
for text in annotations[1:]:  # Skip first (full text)
    vertices = text['boundingPoly']['vertices']
    x_coords = [v.get('x', 0) for v in vertices]
    y_coords = [v.get('y', 0) for v in vertices]

    text_elements.append({
        'text': text['description'],
        'x': sum(x_coords) / len(x_coords),
        'y': sum(y_coords) / len(y_coords),
    })

# Find elements near y=370-380 (second row area)
print("=== Text elements around row 2 (y=370-380) ===")
for elem in text_elements:
    if 370 <= elem['y'] <= 380:
        print(f"Text: '{elem['text']:10s}' at x={elem['x']:.1f}, y={elem['y']:.1f}")

print("\n=== All elements with '2' or '12' ===")
for elem in text_elements:
    if elem['text'] in ['2', '12']:
        print(f"Text: '{elem['text']:10s}' at x={elem['x']:.1f}, y={elem['y']:.1f}")
