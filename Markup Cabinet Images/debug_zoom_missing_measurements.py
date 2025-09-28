#!/usr/bin/env python3
"""
Debug script to investigate missing measurements:
1. "51/2" at (684, 1117) - should be "5 1/2"
2. The missing second "23"
3. "20 ラ 17 116" at (1186, 1410) - being corrected to just "20"
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

# Load environment variables from .env file
load_dotenv()

def zoom_and_save_debug(image_path, x, y, text, output_prefix="debug_zoom"):
    """
    Zoom into a measurement location and save the zoomed image for debugging
    This shows exactly what the OCR sees during verification
    """
    print(f"\n[DEBUG] Zooming on '{text}' at position ({x:.0f}, {y:.0f})")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        return None

    h, w = image.shape[:2]
    print(f"  Image dimensions: {w}x{h}")

    # Create a padded crop area - same as in measurement_based_detector.py
    padding_x = 200  # Wider horizontal padding to capture complete measurements
    padding_y = 50   # Vertical padding
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
    cv2.imwrite(crop_filename, cropped)
    print(f"  [OK] Saved cropped region: {crop_filename}")

    # Zoom in 3x (same as verification function)
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Calculate center of zoomed image
    zoomed_height, zoomed_width = zoomed.shape[:2]
    center_x = zoomed_width / 2
    center_y = zoomed_height / 2

    # Add markers to show center point and center detection area
    zoomed_with_marker = zoomed.copy()

    # Draw center crosshair in red
    cv2.line(zoomed_with_marker,
             (int(center_x - 20), int(center_y)),
             (int(center_x + 20), int(center_y)),
             (0, 0, 255), 2)
    cv2.line(zoomed_with_marker,
             (int(center_x), int(center_y - 20)),
             (int(center_x), int(center_y + 20)),
             (0, 0, 255), 2)

    # Draw center detection area (10 pixel margin) in yellow
    margin = 10
    cv2.rectangle(zoomed_with_marker,
                  (int(center_x - margin), int(center_y - margin)),
                  (int(center_x + margin), int(center_y + margin)),
                  (0, 255, 255), 2)

    # Add text labels
    cv2.putText(zoomed_with_marker, f"Looking for: {text}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)
    cv2.putText(zoomed_with_marker, f"Original pos: ({int(x)}, {int(y)})",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    cv2.putText(zoomed_with_marker, f"Center: ({int(center_x)}, {int(center_y)})",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # Save the zoomed region with markers
    zoom_filename = f"{output_prefix}_{text.replace(' ', '_').replace('/', '-')}_pos{int(x)}x{int(y)}_zoomed.png"
    cv2.imwrite(zoom_filename, zoomed_with_marker)
    print(f"  [OK] Saved zoomed region: {zoom_filename}")

    return zoomed, center_x, center_y, crop_filename, zoom_filename

def run_ocr_on_zoomed(zoomed_image, center_x, center_y, api_key):
    """Run OCR on the zoomed image and check what's at the center"""
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
                valid_at_center = []

                for i, ann in enumerate(annotations[1:], 1):
                    text = ann['description']
                    vertices = ann['boundingPoly']['vertices']

                    # Get bounding box
                    x_coords = [v.get('x', 0) for v in vertices]
                    y_coords = [v.get('y', 0) for v in vertices]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    x_avg = sum(x_coords) / 4
                    y_avg = sum(y_coords) / 4

                    # Check if center point is within this text's bounding box
                    margin = 10
                    at_center = (min_x - margin <= center_x <= max_x + margin and
                                min_y - margin <= center_y <= max_y + margin)

                    center_marker = " [AT CENTER]" if at_center else ""
                    print(f"    {i}. '{text}' at ({x_avg:.0f}, {y_avg:.0f}) bounds=({min_x},{min_y})-({max_x},{max_y}){center_marker}")

                    if at_center:
                        valid_at_center.append(text)

                print(f"\n  Text items at center: {valid_at_center}")
                return full_text, annotations, valid_at_center

    return None, None, []

def main():
    # Get API key from environment variable
    api_key = os.environ.get('GOOGLE_VISION_API_KEY')
    if not api_key:
        print("[ERROR] Please set GOOGLE_VISION_API_KEY environment variable")
        print("Example: export GOOGLE_VISION_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Target image
    image_path = "page_3.png"

    # The problematic measurements to debug
    debug_targets = [
        {"text": "51/2", "x": 684, "y": 1117},  # Should be "5 1/2"
        {"text": "5_1/2", "x": 684, "y": 1117},  # Try with underscore for filename
        {"text": "20_17_116", "x": 1186, "y": 1410},  # Being corrected to just "20"
        # Let's also check areas around where second 23 might be
        {"text": "23_second?", "x": 1100, "y": 1148},  # Guessing it might be to the right
        {"text": "23_above?", "x": 472, "y": 1000},  # Or above the first one
        {"text": "23_below?", "x": 472, "y": 1300},  # Or below
    ]

    print("=" * 80)
    print("ZOOM VERIFICATION DEBUG - MISSING MEASUREMENTS")
    print("=" * 80)
    print(f"Image: {image_path}")
    print("\nInvestigating:")
    print("1. '51/2' at (684, 1117) - should be recognized as '5 1/2'")
    print("2. Missing second '23' width")
    print("3. '20 17 116' group being reduced to just '20'")

    for target in debug_targets:
        zoomed, center_x, center_y, crop_file, zoom_file = zoom_and_save_debug(
            image_path,
            target["x"],
            target["y"],
            target["text"],
            "debug_missing"
        )

        if zoomed is not None:
            # Run OCR to see what it detects
            full_text, annotations, valid_at_center = run_ocr_on_zoomed(
                zoomed, center_x, center_y, api_key
            )

            # Check what the verification logic would do
            if full_text:
                print(f"\n  Checking verification logic:")

                # Show what text is at the center
                if valid_at_center:
                    print(f"    Text at center: {valid_at_center}")
                    print(f"    Would keep/return one of these")
                else:
                    print(f"    No text at center - would keep original")

                # Check for measurement patterns in full text
                lines = full_text.split('\n')
                print(f"\n  Lines in full text:")
                for line in lines:
                    line = line.strip()
                    print(f"    '{line}'")

            print("-" * 80)

if __name__ == "__main__":
    main()