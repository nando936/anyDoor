"""
Debug zoom verification in detail for page 1
"""
import cv2
import numpy as np
import base64
import requests
import os
import re

image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_1.png"
api_key = os.getenv('GOOGLE_CLOUD_API_KEY')

# Coordinates from the debug output
x = 916  # avg_x for the group
y = 1573  # avg_y for the group

# Crop and zoom as the function does
img = cv2.imread(image_path)
crop_size = 200
x_min = max(0, int(x - crop_size))
x_max = min(img.shape[1], int(x + crop_size))
y_min = max(0, int(y - crop_size//2))
y_max = min(img.shape[0], int(y + crop_size//2))

cropped = img[y_min:y_max, x_min:x_max]
zoom_factor = 3
zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

# Save for inspection
cv2.imwrite("debug_zoom_region.png", zoomed)
print(f"Saved zoomed region to debug_zoom_region.png")

# Encode and run OCR
_, buffer = cv2.imencode('.png', zoomed)
zoomed_content = base64.b64encode(buffer).decode('utf-8')

url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
request = {
    "requests": [{
        "image": {"content": zoomed_content},
        "features": [{"type": "TEXT_DETECTION"}]
    }]
}

response = requests.post(url, json=request)

if response.status_code == 200:
    result = response.json()
    if 'responses' in result and result['responses']:
        annotations = result['responses'][0].get('textAnnotations', [])
        if annotations:
            # Get the full text from the zoomed region
            verified_text = annotations[0]['description'].strip()
            print(f"\nFull text from zoom: '{verified_text}'")

            # Check lines
            lines = verified_text.split('\n')
            print(f"\nLines found: {lines}")

            # Check for measurement patterns
            print("\nChecking for measurement patterns:")
            for line in lines:
                line = line.strip()
                print(f"  Line: '{line}'")

                # Check exact patterns from the function
                if re.match(r'^\d+$', line):
                    print(f"    -> Matches whole number pattern")
                elif re.match(r'^\d+\s+\d+/\d+', line):
                    print(f"    -> Matches measurement with fraction pattern")
                elif re.match(r'^\d+-\d+/\d+', line):
                    print(f"    -> Matches dash format pattern")
                elif re.match(r'^\d+/\d+/\d+$', line):
                    print(f"    -> Matches special fraction format")
                else:
                    # Check if it contains a measurement
                    if re.search(r'\d+\s+\d+/\d+', line):
                        print(f"    -> Contains measurement pattern but doesn't start with it")
else:
    print(f"Error: {response.status_code}")