#!/usr/bin/env python3
"""
Debug script to visualize what's at the bounding box center for the "20 17 116" group
"""

import cv2
import numpy as np
import base64
import requests
import os
import sys
from dotenv import load_dotenv

# Fix Unicode output on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

def zoom_and_visualize(image_path, center_x, center_y, group_span, text):
    """
    Zoom into the bounding box center and visualize what's there
    """
    print(f"\n[DEBUG] Zooming on '{text}' at bbox center ({center_x:.0f}, {center_y:.0f})")

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
    x1 = max(0, int(center_x - padding_x))
    y1 = max(0, int(center_y - padding_y))
    x2 = min(w, int(center_x + padding_x))
    y2 = min(h, int(center_y + padding_y))

    print(f"  Crop region: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Crop dimensions: {x2-x1}x{y2-y1}")

    # Crop the region
    cropped = image[y1:y2, x1:x2]

    # Zoom in 3x (same as verification function)
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Calculate center of zoomed image
    zoomed_height, zoomed_width = zoomed.shape[:2]
    center_x_zoomed = zoomed_width / 2
    center_y_zoomed = zoomed_height / 2

    # Add visualization
    zoomed_with_marker = zoomed.copy()

    # Draw center crosshair in red
    cv2.line(zoomed_with_marker,
             (int(center_x_zoomed - 30), int(center_y_zoomed)),
             (int(center_x_zoomed + 30), int(center_y_zoomed)),
             (0, 0, 255), 2)
    cv2.line(zoomed_with_marker,
             (int(center_x_zoomed), int(center_y_zoomed - 30)),
             (int(center_x_zoomed), int(center_y_zoomed + 30)),
             (0, 0, 255), 2)

    # Draw center detection area (10 pixel margin) in yellow
    margin = 10
    cv2.rectangle(zoomed_with_marker,
                  (int(center_x_zoomed - margin), int(center_y_zoomed - margin)),
                  (int(center_x_zoomed + margin), int(center_y_zoomed + margin)),
                  (0, 255, 255), 2)

    # Draw the group span area (if provided) in green
    if group_span:
        # Convert group span coordinates to zoomed coordinates
        x_min, x_max, y_min, y_max = group_span
        # These are in original image coordinates, need to convert to cropped then zoomed
        x_min_crop = (x_min - x1) * zoom_factor
        x_max_crop = (x_max - x1) * zoom_factor
        y_min_crop = (y_min - y1) * zoom_factor
        y_max_crop = (y_max - y1) * zoom_factor

        cv2.rectangle(zoomed_with_marker,
                      (int(x_min_crop), int(y_min_crop)),
                      (int(x_max_crop), int(y_max_crop)),
                      (0, 255, 0), 2)

    # Add text labels
    cv2.putText(zoomed_with_marker, f"Group: '{text}'",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)
    cv2.putText(zoomed_with_marker, f"BBox center: ({int(center_x)}, {int(center_y)})",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    cv2.putText(zoomed_with_marker, "Yellow = center detection area",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    cv2.putText(zoomed_with_marker, "Green = group span",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    # Save the visualization
    output_filename = f"debug_bbox_center_{text.replace(' ', '_').replace('/', '-')}.png"
    cv2.imwrite(output_filename, zoomed_with_marker)
    print(f"  [OK] Saved visualization: {output_filename}")

    # Run OCR to see what's detected
    api_key = os.environ.get('GOOGLE_VISION_API_KEY')
    if api_key:
        # Encode the zoomed image for Vision API
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
                    # Get the full text
                    full_text = annotations[0]['description'].strip()
                    print(f"\n  OCR detected: {full_text}")

                    # Check what's at the center
                    print(f"\n  Text items and their positions:")
                    for i, ann in enumerate(annotations[1:], 1):
                        text_item = ann['description']
                        vertices = ann['boundingPoly']['vertices']

                        # Get bounding box
                        x_coords = [v.get('x', 0) for v in vertices]
                        y_coords = [v.get('y', 0) for v in vertices]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)

                        # Check if center point is within this text's bounding box
                        margin = 10
                        at_center = (min_x - margin <= center_x_zoomed <= max_x + margin and
                                    min_y - margin <= center_y_zoomed <= max_y + margin)

                        center_marker = " [AT CENTER]" if at_center else ""
                        print(f"    {i}. '{text_item}' at bounds=({min_x},{min_y})-({max_x},{max_y}){center_marker}")

    return zoomed_with_marker

def main():
    # Target image
    image_path = "page_3.png"

    # Test cases based on what we know
    test_cases = [
        {
            "text": "20 ラ 17 116",
            "center_x": 1173,  # Bbox center as calculated
            "center_y": 1404,
            "group_span": (1092, 1255, 1391, 1416)  # x_min, x_max, y_min, y_max
        },
        {
            "text": "20 1/16 (first)",
            "center_x": 495,  # Center of first 20 1/16
            "center_y": 1309,
            "group_span": None
        }
    ]

    print("=" * 80)
    print("BOUNDING BOX CENTER DEBUG")
    print("=" * 80)
    print(f"Image: {image_path}")

    for case in test_cases:
        zoom_and_visualize(
            image_path,
            case["center_x"],
            case["center_y"],
            case["group_span"],
            case["text"]
        )
        print("-" * 80)

if __name__ == "__main__":
    main()