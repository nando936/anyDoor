"""
Debug why zoom verification isn't fixing the second "20 1/16" measurement.
Focus on position (1172, 1416) where the second "20" is detected.
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

# Load environment variables
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

def verify_measurement_with_zoom_debug(image_path, x, y, text, api_key):
    """Debug version of verify_measurement_with_zoom"""
    print(f"\n[ZOOM VERIFICATION] for '{text}' at ({x:.0f}, {y:.0f})")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("  ERROR: Cannot load image")
        return text

    h, w = image.shape[:2]

    # Create a padded crop area
    padding_x = 200  # Wider horizontal padding
    padding_y = 50   # Vertical padding
    x1 = max(0, int(x - padding_x))
    y1 = max(0, int(y - padding_y))
    x2 = min(w, int(x + padding_x))
    y2 = min(h, int(y + padding_y))

    print(f"  Crop region: ({x1}, {y1}) to ({x2}, {y2})")

    # Crop the region
    cropped = image[y1:y2, x1:x2]

    # Zoom in 3x
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Save for inspection
    cv2.imwrite(f"debug_zoom_verify_{text.replace(' ', '_').replace('/', '-')}_at_{int(x)}_{int(y)}.png", zoomed)

    # Encode for Vision API
    _, buffer = cv2.imencode('.png', zoomed)
    zoomed_content = base64.b64encode(buffer).decode('utf-8')

    # Run OCR on the zoomed region
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
                # Get full text
                full_text = annotations[0]['description']
                print(f"  Full zoomed text: '{full_text.replace(chr(10), ' | ')}'")

                # Calculate center of zoomed image
                zoomed_height, zoomed_width = zoomed.shape[:2]
                center_x = zoomed_width / 2
                center_y = zoomed_height / 2

                print(f"  Zoomed image center: ({center_x:.0f}, {center_y:.0f})")

                # Find text items near the center
                valid_texts = []
                for ann in annotations[1:]:  # Skip full text
                    vertices = ann.get('boundingPoly', {}).get('vertices', [])
                    if len(vertices) >= 4:
                        x_coords = [v.get('x', 0) for v in vertices]
                        y_coords = [v.get('y', 0) for v in vertices]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)

                        # Check if center point is within this text's bounding box
                        margin = 10
                        if (min_x - margin <= center_x <= max_x + margin and
                            min_y - margin <= center_y <= max_y + margin):
                            text_item = ann['description'].strip()
                            valid_texts.append(text_item)
                            print(f"    Text at center: '{text_item}'")

                # Check for measurement patterns in valid texts
                for text_item in valid_texts:
                    if re.match(r'^\d+$', text_item) or \
                       re.match(r'^\d+\s+\d+/\d+$', text_item) or \
                       re.match(r'^\d+-\d+/\d+$', text_item) or \
                       re.match(r'^\d+/\d+/\d+$', text_item):
                        print(f"    FOUND measurement pattern: '{text_item}'")
                        return text_item

                # Also check if "20 1/16" appears in full text even if not centered
                if '20 1/16' in full_text:
                    print(f"    FOUND '20 1/16' in full text (not necessarily centered)")
                    # Try to extract it
                    match = re.search(r'20\s+1/16', full_text)
                    if match:
                        return '20 1/16'

    print(f"  No clear measurement found, keeping original: '{text}'")
    return text

def main():
    print("="*80)
    print("PAGE 3 ZOOM VERIFICATION DEBUG")
    print("="*80)

    # Get API key
    API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')
    if not API_KEY:
        print("[ERROR] Please set GOOGLE_VISION_API_KEY")
        return

    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3.png"

    # Test positions based on our findings:
    # 1. First "20 1/16" that works: group center at (495, 1309)
    # 2. Second "20" with garbled group: group center at (1186, 1410)

    print("\n[TEST 1] First '20 1/16' group (should work):")
    result1 = verify_measurement_with_zoom_debug(image_path, 495, 1309, "20 1/16", API_KEY)
    print(f"  Result: '{result1}'")

    print("\n[TEST 2] Second '20' group with garbled text:")
    # Test with the actual garbled group text
    result2 = verify_measurement_with_zoom_debug(image_path, 1186, 1410, "20 ラ 17 116", API_KEY)
    print(f"  Result: '{result2}'")

    print("\n[TEST 3] Just the second '20' alone:")
    result3 = verify_measurement_with_zoom_debug(image_path, 1172, 1416, "20", API_KEY)
    print(f"  Result: '{result3}'")

    # Also test slightly different positions around the second "20"
    print("\n[TEST 4] Scanning around the second '20' position:")
    offsets = [
        (0, 0, "center"),
        (-50, 0, "left"),
        (50, 0, "right"),
        (0, -30, "above"),
        (0, 30, "below"),
    ]

    for dx, dy, label in offsets:
        test_x = 1172 + dx
        test_y = 1416 + dy
        print(f"\n  Testing {label} ({test_x}, {test_y}):")
        result = verify_measurement_with_zoom_debug(image_path, test_x, test_y, "20", API_KEY)
        if '1/16' in result:
            print(f"    SUCCESS: Found complete measurement: '{result}'")

    print("\n" + "="*80)
    print("SUMMARY:")
    print("This shows exactly what the zoom verification sees at each position")
    print("and why the second '20 1/16' might not be reconstructing properly.")

if __name__ == "__main__":
    main()