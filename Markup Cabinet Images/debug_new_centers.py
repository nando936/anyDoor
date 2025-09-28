#!/usr/bin/env python3
"""
Debug script to visualize zoom areas with the new true bounding box centers
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

def create_zoom_debug_image(image_path, center_x, center_y, text, group_info=None):
    """
    Create a debug image showing the zoom area and what's detected
    """
    print(f"\n[DEBUG] Creating zoom debug for '{text}' at center ({center_x:.0f}, {center_y:.0f})")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        return None

    h, w = image.shape[:2]

    # Create a padded crop area - same as in measurement_based_detector.py
    padding_x = 200
    padding_y = 50
    x1 = max(0, int(center_x - padding_x))
    y1 = max(0, int(center_y - padding_y))
    x2 = min(w, int(center_x + padding_x))
    y2 = min(h, int(center_y + padding_y))

    print(f"  Crop region: ({x1}, {y1}) to ({x2}, {y2})")

    # Crop the region
    cropped = image[y1:y2, x1:x2]

    # Zoom in 3x
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Calculate center of zoomed image
    zoomed_height, zoomed_width = zoomed.shape[:2]
    center_x_zoomed = zoomed_width / 2
    center_y_zoomed = zoomed_height / 2

    # Create visualization
    zoomed_with_markers = zoomed.copy()

    # Draw center crosshair in red
    cv2.line(zoomed_with_markers,
             (int(center_x_zoomed - 40), int(center_y_zoomed)),
             (int(center_x_zoomed + 40), int(center_y_zoomed)),
             (0, 0, 255), 3)
    cv2.line(zoomed_with_markers,
             (int(center_x_zoomed), int(center_y_zoomed - 40)),
             (int(center_x_zoomed), int(center_y_zoomed + 40)),
             (0, 0, 255), 3)

    # Draw center detection area (10 pixel margin) in yellow
    margin = 10
    cv2.rectangle(zoomed_with_markers,
                  (int(center_x_zoomed - margin), int(center_y_zoomed - margin)),
                  (int(center_x_zoomed + margin), int(center_y_zoomed + margin)),
                  (0, 255, 255), 3)

    # Add text labels
    cv2.putText(zoomed_with_markers, f"Looking for: {text}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3)
    cv2.putText(zoomed_with_markers, f"Center: ({int(center_x)}, {int(center_y)})",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    if group_info:
        cv2.putText(zoomed_with_markers, f"Group span: x=[{group_info['x_min']:.0f}-{group_info['x_max']:.0f}]",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(zoomed_with_markers, f"            y=[{group_info['y_min']:.0f}-{group_info['y_max']:.0f}]",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    # Run OCR and add detected text
    api_key = os.environ.get('GOOGLE_VISION_API_KEY')
    if api_key:
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
                    full_text = annotations[0]['description'].strip()
                    cv2.putText(zoomed_with_markers, f"OCR sees: {full_text.replace(chr(10), ' ')}",
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 0, 0), 2)

                    # Draw bounding boxes for each text item
                    items_at_center = []
                    for i, ann in enumerate(annotations[1:]):
                        vertices = ann['boundingPoly']['vertices']
                        text_item = ann['description']

                        # Get bounds
                        x_coords = [v.get('x', 0) for v in vertices]
                        y_coords = [v.get('y', 0) for v in vertices]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)

                        # Check if at center
                        at_center = (min_x - margin <= center_x_zoomed <= max_x + margin and
                                    min_y - margin <= center_y_zoomed <= max_y + margin)

                        # Draw bounding box
                        color = (0, 255, 0) if at_center else (255, 255, 0)
                        thickness = 3 if at_center else 2
                        cv2.rectangle(zoomed_with_markers,
                                     (min_x, min_y), (max_x, max_y),
                                     color, thickness)

                        # Label the text
                        cv2.putText(zoomed_with_markers, text_item,
                                   (min_x, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7, color, 2)

                        if at_center:
                            items_at_center.append(text_item)

                    if items_at_center:
                        cv2.putText(zoomed_with_markers, f"At center: {', '.join(items_at_center)}",
                                    (10, 230), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 255, 0), 2)
                    else:
                        cv2.putText(zoomed_with_markers, "Nothing at center!",
                                    (10, 230), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 0, 255), 2)

    # Save the debug image
    filename = f"debug_new_center_{text.replace(' ', '_').replace('/', '-').replace(chr(12521), 'ra')}.png"
    cv2.imwrite(filename, zoomed_with_markers)
    print(f"  [OK] Saved debug image: {filename}")

    return zoomed_with_markers

def main():
    image_path = "page_3.png"

    # Test cases based on the measurements we're tracking
    test_cases = [
        {
            "text": "51/2",
            "center_x": 684,
            "center_y": 1117,
            "group_info": None
        },
        {
            "text": "20 ra 17 116 (old avg)",
            "center_x": 1186,  # Old average center
            "center_y": 1410,
            "group_info": {"x_min": 1172, "x_max": 1255, "y_min": 1416, "y_max": 1416}
        },
        {
            "text": "20 ra 17 116 (new bounds)",
            "center_x": 1154,  # New true bounds center
            "center_y": 1391,
            "group_info": {"x_min": 1018, "x_max": 1289, "y_min": 1319, "y_max": 1463}
        },
        {
            "text": "20 1/16 (first)",
            "center_x": 495,
            "center_y": 1309,
            "group_info": None
        },
        {
            "text": "23",
            "center_x": 472,
            "center_y": 1148,
            "group_info": None
        }
    ]

    print("=" * 80)
    print("DEBUG NEW ZOOM CENTERS")
    print("=" * 80)
    print(f"Image: {image_path}")

    for case in test_cases:
        create_zoom_debug_image(
            image_path,
            case["center_x"],
            case["center_y"],
            case["text"],
            case.get("group_info")
        )

if __name__ == "__main__":
    main()