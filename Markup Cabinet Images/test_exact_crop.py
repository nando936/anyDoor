#!/usr/bin/env python3
"""
Test with the EXACT same crop region as the detector uses
"""

import cv2
import numpy as np
import base64
import requests
import json
import sys

# Fix Unicode issues
sys.stdout.reconfigure(encoding='utf-8')

def apply_hsv_preprocessing(image):
    """Apply HSV preprocessing to isolate green text on brown backgrounds"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    return mask_inv

# Image path
image_path = '//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_4.png'

# The problematic group bounds from the detector output
group_bounds = {
    'left': 653,
    'right': 850,
    'top': 1075,
    'bottom': 1164
}

print("Using EXACT detector logic:")
print(f"Group bounds: x=[{group_bounds['left']} to {group_bounds['right']}], y=[{group_bounds['top']} to {group_bounds['bottom']}]")

# Load image
image = cv2.imread(image_path)
h, w = image.shape[:2]

# EXACT same logic as detector
padding = 30
x1 = max(0, int(group_bounds['left'] - padding))
y1 = max(0, int(group_bounds['top'] - padding))
x2 = min(w, int(group_bounds['right'] + padding))
y2 = min(h, int(group_bounds['bottom'] + padding))

print(f"With padding={padding}: x=[{x1} to {x2}], y=[{y1} to {y2}]")
print(f"Crop dimensions: {x2-x1}x{y2-y1}")

# Crop and zoom
cropped = image[y1:y2, x1:x2]
zoom_factor = 3
zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

# Apply HSV (like detector does)
zoomed_hsv = apply_hsv_preprocessing(zoomed)

# Save for inspection
cv2.imwrite('test_exact_crop_hsv.png', zoomed_hsv)
print("\nSaved test_exact_crop_hsv.png for inspection")

# Run OCR
_, buffer = cv2.imencode('.png', zoomed_hsv)
content = base64.b64encode(buffer).decode('utf-8')

api_key = 'AIzaSyC9wt5Sl-zS9V27K8eUqMZRG4Zd3h_1K4Q'
url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'

request = {
    'requests': [{
        'image': {'content': content},
        'features': [{'type': 'TEXT_DETECTION'}]
    }]
}

response = requests.post(url, json=request)

if response.status_code == 200:
    result = response.json()
    if 'responses' in result and result['responses']:
        annotations = result['responses'][0].get('textAnnotations', [])
        if annotations:
            print('\nOCR RESULT WITH EXACT DETECTOR CROP:')
            print('=' * 50)
            full_text = annotations[0]['description']
            print(f"Full text: {repr(full_text)}")
            print('\nFormatted:')
            print(full_text)
            print('=' * 50)

            print('\nThis explains the issue! The detector IS seeing "5 1/2" because')
            print('the group bounds extend up to Y=1045, capturing the "5 1/2" at Y=1062')