"""
Test a potential fix for the second "20 1/16" detection issue.
The problem: zoom verification finds "20 1/16" but only returns the centered text "20".
Solution: When full text contains a valid measurement pattern, return it even if not perfectly centered.
"""
import cv2
import numpy as np
import requests
import base64
import json
import re
import os
from dotenv import load_dotenv
import sys

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

def improved_verify_measurement(image_path, x, y, group_text, api_key):
    """Improved verification that prioritizes complete measurements in zoomed text"""
    print(f"\n[IMPROVED VERIFICATION] for '{group_text}' at ({x:.0f}, {y:.0f})")

    # Load and crop image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    padding_x = 200
    padding_y = 50
    x1 = max(0, int(x - padding_x))
    y1 = max(0, int(y - padding_y))
    x2 = min(w, int(x + padding_x))
    y2 = min(h, int(y + padding_y))

    cropped = image[y1:y2, x1:x2]
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Run OCR
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
                full_text = annotations[0]['description']
                print(f"  Full zoomed text: '{full_text.replace(chr(10), ' | ')}'")

                # Priority 1: Look for complete measurements in full text
                # Common patterns for cabinet measurements
                patterns = [
                    r'\b(\d+\s+\d+/\d+)\b',  # "20 1/16"
                    r'\b(\d+-\d+/\d+)\b',     # "20-1/16"
                    r'\b(\d+)\b(?:\s+|\s*)(\d+/\d+)\b',  # "20" followed by "1/16"
                ]

                for pattern in patterns:
                    matches = re.finditer(pattern, full_text)
                    for match in matches:
                        if match.lastindex == 2:
                            # Two capture groups - combine them
                            measurement = f"{match.group(1)} {match.group(2)}"
                        else:
                            measurement = match.group(1)

                        # Sanity check - cabinet dimensions typically between 3" and 48"
                        first_num = int(re.match(r'(\d+)', measurement).group(1))
                        if 3 <= first_num <= 48:
                            print(f"  FOUND valid measurement in full text: '{measurement}'")
                            return measurement.replace('-', ' ')  # Normalize format

                # Priority 2: Check center-aligned text (original logic)
                zoomed_height, zoomed_width = zoomed.shape[:2]
                center_x = zoomed_width / 2
                center_y = zoomed_height / 2

                for ann in annotations[1:]:
                    vertices = ann.get('boundingPoly', {}).get('vertices', [])
                    if len(vertices) >= 4:
                        x_coords = [v.get('x', 0) for v in vertices]
                        y_coords = [v.get('y', 0) for v in vertices]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)

                        margin = 10
                        if (min_x - margin <= center_x <= max_x + margin and
                            min_y - margin <= center_y <= max_y + margin):
                            text_item = ann['description'].strip()

                            # Check if it's a valid measurement
                            if re.match(r'^\d+$', text_item):
                                num = int(text_item)
                                if 3 <= num <= 48:
                                    print(f"  Found centered number: '{text_item}'")
                                    # But still prefer complete measurement if available
                                    if re.search(rf'\b{text_item}\s+\d+/\d+\b', full_text):
                                        match = re.search(rf'\b({text_item}\s+\d+/\d+)\b', full_text)
                                        if match:
                                            print(f"  Upgraded to full measurement: '{match.group(1)}'")
                                            return match.group(1)
                                    return text_item

    print(f"  No improvement found, returning original: '{group_text}'")
    return group_text

def main():
    print("="*80)
    print("TESTING IMPROVED VERIFICATION FOR PAGE 3")
    print("="*80)

    API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')
    if not API_KEY:
        print("[ERROR] Please set GOOGLE_VISION_API_KEY")
        return

    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3.png"

    # Test the problematic second "20" group
    print("\nTesting the second '20' group that should be '20 1/16':")

    # Group center position from our debug
    group_x = 1186
    group_y = 1410
    group_text = "20 ラ 17 116"  # The garbled group text

    result = improved_verify_measurement(image_path, group_x, group_y, group_text, API_KEY)

    print(f"\n[RESULT] Improved verification returns: '{result}'")

    if result == "20 1/16":
        print("[SUCCESS] Second '20 1/16' would be correctly detected!")
    else:
        print(f"[ISSUE] Still not detecting properly, got: '{result}'")

    # Also test with just "20" as input
    print("\nTesting with just '20' as input text:")
    result2 = improved_verify_measurement(image_path, 1172, 1416, "20", API_KEY)
    print(f"\n[RESULT] For standalone '20': '{result2}'")

if __name__ == "__main__":
    main()