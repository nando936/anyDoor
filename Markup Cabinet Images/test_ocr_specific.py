#!/usr/bin/env python3
"""
Test OCR on the specific debug image to see what it returns
"""

import base64
import requests
import json
import sys

# Fix Unicode issues
sys.stdout.reconfigure(encoding='utf-8')

# Read the specific debug image
image_path = '//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/debug_zoom_5_1-2_WRONG_pos751x1119_zoomed_hsv.png'

print(f"Testing OCR on: {image_path}")
print("=" * 80)

with open(image_path, 'rb') as f:
    content = base64.b64encode(f.read()).decode('utf-8')

# Google Vision API
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
            print('FULL TEXT FROM GOOGLE VISION:')
            print('-' * 40)
            full_text = annotations[0]['description']
            # Show with escaped newlines to be clear
            print(f"Raw: {repr(full_text)}")
            print()
            print("Formatted:")
            print(full_text)
            print('-' * 40)

            print('\nLINES IN FULL TEXT:')
            lines = full_text.split('\n')
            for i, line in enumerate(lines, 1):
                print(f"Line {i}: '{line}'")

            print('\nINDIVIDUAL TEXT ITEMS:')
            for i, ann in enumerate(annotations[1:], 1):
                text = ann['description']
                vertices = ann.get('boundingPoly', {}).get('vertices', [])
                if vertices:
                    x_avg = sum(v.get('x', 0) for v in vertices) / 4
                    y_avg = sum(v.get('y', 0) for v in vertices) / 4
                    print(f'{i:2d}. "{text}" at position ({x_avg:.0f}, {y_avg:.0f})')
                else:
                    print(f'{i:2d}. "{text}"')
        else:
            print("No text detected")
else:
    print(f'Error: {response.status_code}')