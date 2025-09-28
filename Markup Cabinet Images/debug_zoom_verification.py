#!/usr/bin/env python3
"""
Debug script to visualize what zoom verification sees for each measurement
Saves zoomed regions as images to understand OCR errors
"""

import cv2
import numpy as np
import base64
import requests
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix Unicode issues on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

def apply_hsv_preprocessing(image):
    """Apply HSV preprocessing to isolate green text on brown backgrounds"""
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask for green text
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert to make green text black on white background
    mask_inv = cv2.bitwise_not(mask)

    return mask_inv

def zoom_and_save_debug(image_path, x, y, text, output_folder=".", output_prefix="debug_zoom", group_bounds=None):
    """
    Zoom into a measurement location and save the zoomed image for debugging
    This shows exactly what the OCR sees during verification
    Uses EXACT same logic as measurement_based_detector.py
    """
    print(f"\n[DEBUG] Zooming on '{text}' at position ({x:.0f}, {y:.0f})")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        return None

    h, w = image.shape[:2]
    print(f"  Image dimensions: {w}x{h}")

    # EXACT same crop logic as measurement_based_detector.py
    if group_bounds:
        print(f"  Using group bounds: x=[{group_bounds['left']} to {group_bounds['right']}], y=[{group_bounds['top']} to {group_bounds['bottom']}]")
        padding = 30
        x1 = max(0, int(group_bounds['left'] - padding))
        y1 = max(0, int(group_bounds['top'] - padding))
        x2 = min(w, int(group_bounds['right'] + padding))
        y2 = min(h, int(group_bounds['bottom'] + padding))
    else:
        padding_x = 200
        padding_y = 50
        x1 = max(0, int(x - padding_x))
        y1 = max(0, int(y - padding_y))
        x2 = min(w, int(x + padding_x))
        y2 = min(h, int(y + padding_y))

    print(f"  Crop region: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Crop dimensions: {x2-x1}x{y2-y1}")

    # Crop the region
    cropped = image[y1:y2, x1:x2]

    # Save the cropped region
    crop_filename = f"{output_prefix}_{text.replace(' ', '_').replace('/', '-')}_pos{int(x)}x{int(y)}_cropped.png"
    crop_filepath = os.path.join(output_folder, crop_filename)
    cv2.imwrite(crop_filepath, cropped)
    print(f"  [OK] Saved cropped region: {crop_filepath}")

    # Apply HSV preprocessing (same as in measurement_based_detector.py)
    cropped_hsv = apply_hsv_preprocessing(cropped)

    # Save the HSV processed version
    hsv_filename = f"{output_prefix}_{text.replace(' ', '_').replace('/', '-')}_pos{int(x)}x{int(y)}_hsv.png"
    hsv_filepath = os.path.join(output_folder, hsv_filename)
    cv2.imwrite(hsv_filepath, cropped_hsv)
    print(f"  [OK] Saved HSV processed: {hsv_filepath}")

    # Zoom in 3x (same as verification function)
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
    zoomed_hsv = cv2.resize(cropped_hsv, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Add a red crosshair at the center to show exact target point
    zoomed_with_marker = zoomed.copy()
    center_x = (x - x1) * zoom_factor
    center_y = (y - y1) * zoom_factor

    # Draw crosshair
    cv2.line(zoomed_with_marker,
             (int(center_x - 20), int(center_y)),
             (int(center_x + 20), int(center_y)),
             (0, 0, 255), 2)
    cv2.line(zoomed_with_marker,
             (int(center_x), int(center_y - 20)),
             (int(center_x), int(center_y + 20)),
             (0, 0, 255), 2)

    # Add text label
    cv2.putText(zoomed_with_marker, f"Target: {text}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)
    cv2.putText(zoomed_with_marker, f"Original pos: ({int(x)}, {int(y)})",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    # Save the zoomed region with marker
    zoom_filename = f"{output_prefix}_{text.replace(' ', '_').replace('/', '-')}_pos{int(x)}x{int(y)}_zoomed.png"
    zoom_filepath = os.path.join(output_folder, zoom_filename)
    cv2.imwrite(zoom_filepath, zoomed_with_marker)
    print(f"  [OK] Saved zoomed region: {zoom_filepath}")

    # Save the HSV zoomed version
    zoom_hsv_filename = f"{output_prefix}_{text.replace(' ', '_').replace('/', '-')}_pos{int(x)}x{int(y)}_zoomed_hsv.png"
    zoom_hsv_filepath = os.path.join(output_folder, zoom_hsv_filename)
    cv2.imwrite(zoom_hsv_filepath, zoomed_hsv)
    print(f"  [OK] Saved HSV zoomed: {zoom_hsv_filepath}")

    return zoomed_hsv, crop_filepath, zoom_filepath

def run_ocr_on_zoomed(zoomed_image, api_key):
    """Run OCR on the zoomed image to see what it detects"""
    # Encode the zoomed image for Vision API
    _, buffer = cv2.imencode('.png', zoomed_image)
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
                # Get the full text from the zoomed region
                full_text = annotations[0]['description'].strip()
                print(f"\n  OCR detected in zoomed region:")
                print(f"  Full text: {full_text}")

                # Show individual text items
                print(f"\n  Individual items detected:")
                for i, ann in enumerate(annotations[1:], 1):
                    text = ann['description']
                    vertices = ann['boundingPoly']['vertices']
                    x_avg = sum(v.get('x', 0) for v in vertices) / 4
                    y_avg = sum(v.get('y', 0) for v in vertices) / 4
                    print(f"    {i}. '{text}' at ({x_avg:.0f}, {y_avg:.0f})")

                return full_text, annotations

    return None, None

def main():
    # Get API key from environment variable for security
    api_key = os.environ.get('GOOGLE_VISION_API_KEY')
    if not api_key:
        print("[ERROR] Please set GOOGLE_VISION_API_KEY environment variable")
        print("Example: export GOOGLE_VISION_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Target image and measurements to debug
    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_4.png"

    # Output folder - same as the page location
    output_folder = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/"

    # Check the problematic position that's being misidentified
    debug_targets = [
        {"text": "5_1-2_WRONG", "x": 751, "y": 1119},  # This is being detected as 5 1/2 but shouldn't be
    ]

    print("=" * 80)
    print("ZOOM VERIFICATION DEBUG")
    print("=" * 80)
    print(f"Image: {image_path}")

    for target in debug_targets:
        zoomed, crop_file, zoom_file = zoom_and_save_debug(
            image_path,
            target["x"],
            target["y"],
            target["text"],
            output_folder,
            "debug_zoom"
        )

        if zoomed is not None:
            # Run OCR on the HSV-processed zoomed image (same as actual detector)
            print(f"\n  Running OCR on HSV-processed image...")
            full_text, annotations = run_ocr_on_zoomed(zoomed, api_key)

            # Check what the verification logic would return
            if full_text:
                print(f"\n  Checking verification logic:")
                lines = full_text.split('\n')
                for line in lines:
                    line = line.strip()
                    print(f"    Line: '{line}'")

                    # Check if it matches single digit pattern (this might be the problem)
                    import re
                    if re.match(r'^\d+$', line):
                        print(f"    -> Would return: '{line}' (matches single digit pattern)")
                        break
                    elif re.match(r'^\d+\s+\d+/\d+$', line):
                        print(f"    -> Would return: '{line}' (matches fraction pattern)")
                        break

            print("-" * 80)

if __name__ == "__main__":
    main()