#!/usr/bin/env python3
"""
Test Vision API on specific areas where we expect to find "23".
"""

import cv2
import numpy as np
import os
import sys
from google.cloud import vision
from google.api_core import client_options
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

def test_cropped_area(image_path, x_center, y_center, width=200, height=100, label=""):
    """Test Vision API on a cropped area around the specified center point."""

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return

    img_height, img_width = image.shape[:2]

    # Calculate crop bounds
    x1 = max(0, int(x_center - width/2))
    y1 = max(0, int(y_center - height/2))
    x2 = min(img_width, int(x_center + width/2))
    y2 = min(img_height, int(y_center + height/2))

    # Crop the image
    cropped = image[y1:y2, x1:x2]

    # Save the cropped area for inspection
    crop_filename = f"DEBUG_crop_area_{label}_{x_center}x{y_center}.png"
    cv2.imwrite(crop_filename, cropped)
    print(f"\n{'='*60}")
    print(f"Testing area {label} at ({x_center}, {y_center})")
    print(f"Cropped region: [{x1},{y1}] to [{x2},{y2}]")
    print(f"Saved crop to: {crop_filename}")

    # Initialize Vision API
    api_key = os.environ.get('GOOGLE_VISION_API_KEY') or os.environ.get('GOOGLE_CLOUD_API_KEY')
    if not api_key:
        print("Error: API key not set")
        return

    options = client_options.ClientOptions(api_key=api_key)
    client = vision.ImageAnnotatorClient(client_options=options)

    # Encode cropped image
    _, encoded = cv2.imencode('.png', cropped)
    content = encoded.tobytes()

    image_vision = vision.Image(content=content)

    # Detect text
    response = client.text_detection(image=image_vision)
    texts = response.text_annotations

    if not texts:
        print("No text detected in this area!")
        return

    # Skip first annotation (full text)
    text_items = texts[1:] if len(texts) > 1 else []

    print(f"Found {len(text_items)} text items:")
    for text in text_items:
        desc = text.description.strip()
        vertices = text.bounding_poly.vertices
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        center_x = sum(x_coords) // 4 + x1  # Adjust for crop offset
        center_y = sum(y_coords) // 4 + y1
        print(f"  '{desc}' at local ({center_x}, {center_y})")

    # Look specifically for "23" or "2" and "3"
    found_23 = False
    found_2 = False
    found_3 = False

    for text in text_items:
        desc = text.description.strip()
        if desc == "23":
            found_23 = True
            print(f"  [OK] Found '23'!")
        elif desc == "2":
            found_2 = True
        elif desc == "3":
            found_3 = True

    if found_2 and found_3:
        print(f"  [!] Found separate '2' and '3' - might need assembly")
    elif not found_23 and not found_2:
        print(f"  [X] No '23' or '2' found in this area")

def main():
    image_path = "page_3.png" if len(sys.argv) < 2 else sys.argv[1]

    print("TESTING SPECIFIC AREAS FOR '23' MEASUREMENTS")
    print("="*60)

    # Test areas based on old data positions
    # Old position 1: around (471, 1148)
    test_cropped_area(image_path, 471, 1148, 250, 150, "old_pos1")

    # Old position 2: around (1253, 1250) - but this is beyond image width!
    # Image is only 1224 pixels wide, so this position is invalid
    print(f"\n[!] Note: Old position 2 at (1253, 1250) is beyond image width (1224)")

    # New detection position: around (835, 834)
    test_cropped_area(image_path, 835, 834, 250, 150, "new_pos")

    # Also test the area where we see "2"
    test_cropped_area(image_path, 730, 545, 250, 150, "two_digit")

    # Test lower area where we might expect another 23
    test_cropped_area(image_path, 500, 1100, 300, 200, "lower_area")
    test_cropped_area(image_path, 800, 1100, 300, 200, "lower_right")

if __name__ == "__main__":
    main()