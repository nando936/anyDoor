#!/usr/bin/env python3
"""
Test script to check what Google Vision API detects in the preprocessed image
Uses the same API parameters as the real detection code
"""
import cv2
import base64
import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GOOGLE_VISION_API_KEY')
if not api_key:
    print("[ERROR] GOOGLE_VISION_API_KEY not found in environment")
    exit(1)

# Load the preprocessed image
image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-4_phase1_preprocessed.png"
image = cv2.imread(image_path)

if image is None:
    print(f"ERROR: Could not load image from {image_path}")
    exit(1)

print(f"Loaded preprocessed image: {image.shape[1]}x{image.shape[0]} pixels")

# Encode image to base64 (same as text_detection.py line 166)
_, buffer = cv2.imencode('.png', image)
content = base64.b64encode(buffer).decode('utf-8')

# Call Google Vision API (same as text_detection.py lines 178-183)
print("\nCalling Google Vision API...")
url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
payload = {
    'requests': [{
        'image': {'content': content},
        'features': [{'type': 'TEXT_DETECTION'}]
    }]
}

response = requests.post(url, json=payload)

if response.status_code != 200:
    print(f"[ERROR] API request failed: {response.status_code}")
    print(response.text)
    exit(1)

data = response.json()

# Check for errors
if 'error' in data.get('responses', [{}])[0]:
    print(f"[ERROR] Vision API error: {data['responses'][0]['error']}")
    exit(1)

annotations = data.get('responses', [{}])[0].get('textAnnotations', [])

print(f"Google Vision API found {len(annotations)} text annotations")

if len(annotations) == 0:
    print("\n[WARNING] No text detected by Vision API!")
    exit(0)

# First annotation is full page text, skip it
# annotations[1:] are individual text items with bounding boxes
print(f"Individual text items: {len(annotations) - 1}")

# M7 position from debug output: (1115, 666)
# The "9" should be below M7, roughly at X=1100-1200, Y=700-850
TARGET_X_MIN = 1000
TARGET_X_MAX = 1200
TARGET_Y_MIN = 670
TARGET_Y_MAX = 900

print(f"\nLooking for text near M7 (x={TARGET_X_MIN}-{TARGET_X_MAX}, y={TARGET_Y_MIN}-{TARGET_Y_MAX})")
print("="*80)

# Filter text items with numbers (same as text_detection.py line 228)
items_with_numbers = []
items_near_9 = []

for i, annotation in enumerate(annotations[1:], start=1):
    text = annotation.get('description', '')

    # Check if text contains digit
    has_digit = any(c.isdigit() for c in text)
    if not has_digit:
        continue

    # Get bounding box
    vertices = annotation.get('boundingPoly', {}).get('vertices', [])
    if len(vertices) < 4:
        continue

    # Calculate center (same as text_detection.py lines 289-291)
    x_coords = [v.get('x', 0) for v in vertices]
    y_coords = [v.get('y', 0) for v in vertices]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    item = {
        'index': i,
        'text': text,
        'center': (int(center_x), int(center_y)),
        'bounds': {
            'left': min(x_coords),
            'right': max(x_coords),
            'top': min(y_coords),
            'bottom': max(y_coords)
        }
    }

    items_with_numbers.append(item)

    # Check if near target location
    if (TARGET_X_MIN <= center_x <= TARGET_X_MAX and
        TARGET_Y_MIN <= center_y <= TARGET_Y_MAX):
        items_near_9.append(item)

print(f"\nTotal text items with numbers: {len(items_with_numbers)}")
print(f"Text items near the '9' location: {len(items_near_9)}")

# Check if "9" was found
found_9 = any(item['text'].strip() == '9' for item in items_with_numbers)

print("\n" + "="*80)
if found_9:
    print("✓ THE '9' WAS FOUND BY GOOGLE VISION API!")
    nine_items = [item for item in items_with_numbers if item['text'].strip() == '9']
    for item in nine_items:
        print(f"  Text: '{item['text']}'")
        print(f"  Position: {item['center']}")
        print(f"  Index: #{item['index']}")
else:
    print("✗ THE '9' WAS NOT FOUND BY GOOGLE VISION API")
print("="*80)

# Show all text items found
print("\nALL TEXT ITEMS WITH NUMBERS (sorted by Y position):")
print("-"*80)
for item in sorted(items_with_numbers, key=lambda x: x['center'][1]):
    near_marker = " ← NEAR '9' LOCATION" if item in items_near_9 else ""
    print(f"'{item['text']:15s}' at ({item['center'][0]:4d}, {item['center'][1]:4d}){near_marker}")

if items_near_9:
    print("\n" + "="*80)
    print("TEXT ITEMS NEAR THE '9' LOCATION:")
    print("="*80)
    for item in items_near_9:
        print(f"Text: '{item['text']}'")
        print(f"Position: {item['center']}")
        print(f"Bounds: left={item['bounds']['left']:.0f}, right={item['bounds']['right']:.0f}, "
              f"top={item['bounds']['top']:.0f}, bottom={item['bounds']['bottom']:.0f}")
        print()

# Save detailed results to JSON
output_json = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/test_google_vision_9_results.json"
with open(output_json, 'w') as f:
    json.dump({
        'total_annotations': len(annotations),
        'items_with_numbers': len(items_with_numbers),
        'items_near_9_location': len(items_near_9),
        'found_9': found_9,
        'all_items': items_with_numbers,
        'items_near_9': items_near_9
    }, f, indent=2)

print(f"\n[SAVED] Detailed results: {output_json}")
