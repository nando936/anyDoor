"""
Debug why the second 23" width isn't being detected on page 3
"""
import cv2
import numpy as np
import requests
import base64
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GOOGLE_VISION_API_KEY')

# Load the image
image_path = r"\\vmware-host\Shared Folders\suarez group qb\customers\raised panel\Measures-2025-09-08(17-08)\all_pages\page_3.png"
image = cv2.imread(image_path)
h, w = image.shape[:2]

print("="*60)
print("SEARCHING FOR ALL '23' MEASUREMENTS IN PAGE 3")
print("="*60)

# Run OCR on full image
success, buffer = cv2.imencode('.png', image)
if not success:
    raise Exception("Failed to encode image")

# Use Vision API directly
url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
request_json = {
    "requests": [{
        "image": {
            "content": base64.b64encode(buffer).decode('utf-8')
        },
        "features": [{
            "type": "TEXT_DETECTION",
            "maxResults": 50
        }]
    }]
}

response = requests.post(url, json=request_json)
response_data = response.json()

if 'error' in response_data:
    raise Exception(f"API Error: {response_data['error']}")

texts = response_data['responses'][0].get('textAnnotations', [])

print("\nFull page OCR - Looking for '23':")
print("-"*40)

# Find all instances of "23"
found_23 = []
for text in texts[1:]:  # Skip first which is full text
    if text.get('description', '').strip() == "23":
        vertices = text.get('boundingPoly', {}).get('vertices', [])
        if vertices:
            x = (vertices[0].get('x', 0) + vertices[2].get('x', 0)) // 2
            y = (vertices[0].get('y', 0) + vertices[2].get('y', 0)) // 2
            found_23.append({
                'text': text['description'],
                'x': x,
                'y': y,
                'bounds': [(v.get('x', 0), v.get('y', 0)) for v in vertices]
            })
            print(f"  Found '23' at position ({x}, {y})")

print(f"\nTotal '23' found in initial OCR: {len(found_23)}")

# Let's also look for any text that might be misread as something else
print("\n" + "="*60)
print("CHECKING RIGHT SIDE OF IMAGE (where second 23 should be)")
print("="*60)

# Focus on right side where the sink is
right_side_x_start = 900  # Approximate x coordinate where right opening starts

print(f"\nAll text found on right side (x > {right_side_x_start}):")
print("-"*40)

right_side_texts = []
for text in texts[1:]:
    vertices = text.bounding_poly.vertices
    x = (vertices[0].x + vertices[2].x) // 2
    y = (vertices[0].y + vertices[2].y) // 2

    if x > right_side_x_start:
        right_side_texts.append({
            'text': text.description,
            'x': x,
            'y': y
        })

# Sort by y position (top to bottom)
right_side_texts.sort(key=lambda t: t['y'])

for t in right_side_texts:
    print(f"  '{t['text']}' at ({t['x']}, {t['y']})")

# Check if there's anything that could be "23" misread
print("\n" + "="*60)
print("POSSIBLE MISREADS OF '23':")
print("="*60)

possible_misreads = ['2', '3', '23', '73', '28', '22', '25', '23"', '"23', '23.', '.23']
for text in texts[1:]:
    if any(text.description.strip().startswith(p) or text.description.strip().endswith(p)
           for p in possible_misreads):
        vertices = text.bounding_poly.vertices
        x = (vertices[0].x + vertices[2].x) // 2
        y = (vertices[0].y + vertices[2].y) // 2
        if x > right_side_x_start and 1100 < y < 1250:  # Approximate y range for widths
            print(f"  Possible misread: '{text.description}' at ({x}, {y})")

# Let's also check what's in the horizontal measurements list
print("\n" + "="*60)
print("ANALYZING HORIZONTAL MEASUREMENT REGION")
print("="*60)

# The horizontal measurements are typically in a certain y-range
horizontal_y_range = (1100, 1250)  # Approximate range where width measurements appear

horizontal_texts = []
for text in texts[1:]:
    vertices = text.bounding_poly.vertices
    x = (vertices[0].x + vertices[2].x) // 2
    y = (vertices[0].y + vertices[2].y) // 2

    if horizontal_y_range[0] < y < horizontal_y_range[1]:
        horizontal_texts.append({
            'text': text.description,
            'x': x,
            'y': y
        })

horizontal_texts.sort(key=lambda t: t['x'])  # Sort left to right

print(f"All text in horizontal measurement region (y between {horizontal_y_range[0]} and {horizontal_y_range[1]}):")
for t in horizontal_texts:
    print(f"  '{t['text']}' at ({t['x']}, {t['y']})")

# Save a debug image marking where we expect the second "23"
debug_image = image.copy()

# Mark the found "23"
for f in found_23:
    cv2.circle(debug_image, (f['x'], f['y']), 20, (0, 255, 0), 3)
    cv2.putText(debug_image, "Found 23", (f['x']-50, f['y']-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Mark where we expect the second "23" (approximate)
expected_x = 1100  # Approximate x for right opening
expected_y = 1150  # Approximate y for width measurement
cv2.circle(debug_image, (expected_x, expected_y), 30, (0, 0, 255), 3)
cv2.putText(debug_image, "Missing 23?", (expected_x-50, expected_y-40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Draw rectangle around right side area
cv2.rectangle(debug_image, (right_side_x_start, horizontal_y_range[0]),
              (w, horizontal_y_range[1]), (255, 0, 0), 2)

output_path = "debug_missing_23.png"
cv2.imwrite(output_path, debug_image)
print(f"\n[OK] Debug image saved to {output_path}")