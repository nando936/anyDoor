#!/usr/bin/env python3
"""
Test script to check what Google Vision API detects on page 5 preprocessed image.
Zooms the entire image by 2x for better detection.
Specifically looking for the missing '9' and '6' single-digit measurements.
"""

import sys
import os
import base64
import requests
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def call_vision_api(image_data, api_key):
    """Call Vision API with image data."""
    # Encode image to base64
    _, buffer = cv2.imencode('.png', image_data)
    content = base64.b64encode(buffer).decode('utf-8')

    # Call Vision API
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request_data = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, json=request_data)

    if response.status_code != 200:
        return None, f"API failed: {response.status_code}"

    result = response.json()
    if 'responses' not in result or not result['responses']:
        return None, "No responses from Vision API"

    texts = result['responses'][0].get('textAnnotations', [])
    return texts, None


def test_vision_api_full_zoom(image_path, api_key):
    """Test Vision API detection on full image zoomed 2x."""

    print(f"Testing Vision API on: {image_path}")
    print("=" * 80)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Original image size: {w}x{h}px")

    # Zoom entire image by 2x
    zoomed = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    zoom_h, zoom_w = zoomed.shape[:2]
    print(f"Zoomed image size: {zoom_w}x{zoom_h}px (2x)")
    print("=" * 80)

    # Call Vision API on zoomed full image
    print("\nCalling Vision API on zoomed full image...")
    texts, error = call_vision_api(zoomed, api_key)

    if error:
        print(f"[ERROR] {error}")
        return

    if not texts:
        print("[ERROR] No text detected")
        return

    print(f"\nFound {len(texts) - 1} text items")
    print("-" * 80)

    # Track special detections
    target_detections = []
    single_digit_detections = []

    # Process detections (skip first full text)
    for i, text_ann in enumerate(texts[1:], 1):
        description = text_ann['description']

        # Get bounding box in zoomed image
        vertices = text_ann['boundingPoly']['vertices']
        x_coords = [v.get('x', 0) for v in vertices]
        y_coords = [v.get('y', 0) for v in vertices]

        # Convert back to original image coordinates (divide by 2)
        orig_x = (sum(x_coords) / len(x_coords)) / 2.0
        orig_y = (sum(y_coords) / len(y_coords)) / 2.0

        width = (max(x_coords) - min(x_coords)) / 2.0
        height = (max(y_coords) - min(y_coords)) / 2.0

        # Check if single digit
        is_single_digit = len(description) == 1 and description.isdigit()

        # Highlight if it's '6' or '9'
        highlight = ""
        if description in ['6', '9']:
            highlight = " *** TARGET FOUND ***"
            target_detections.append({
                'text': description,
                'x': orig_x,
                'y': orig_y,
                'width': width,
                'height': height
            })

        # Track all single digits
        if is_single_digit:
            single_digit_detections.append({
                'text': description,
                'x': orig_x,
                'y': orig_y,
                'width': width,
                'height': height
            })

        print(f"  {i:3d}. '{description}' at ({orig_x:.0f}, {orig_y:.0f}) "
              f"size: {width:.0f}x{height:.0f}px{highlight}")

    # Summary
    print("\n" + "=" * 80)
    print("\n[SUMMARY - FULL IMAGE 2X ZOOM DETECTION]")
    print("-" * 80)
    print(f"Total text items: {len(texts) - 1}")
    print(f"Single digits found: {len(single_digit_detections)}")
    print(f"Target '6' or '9' found: {len(target_detections)}")

    if single_digit_detections:
        print("\n[ALL SINGLE DIGITS]:")
        for det in single_digit_detections:
            marker = " <-- TARGET" if det['text'] in ['6', '9'] else ""
            print(f"  '{det['text']}' at ({det['x']:.0f}, {det['y']:.0f}) "
                  f"size: {det['width']:.0f}x{det['height']:.0f}px{marker}")

    if target_detections:
        print("\n[TARGET '6' or '9' MEASUREMENTS]:")
        for det in target_detections:
            print(f"  '{det['text']}' at ({det['x']:.0f}, {det['y']:.0f}) "
                  f"size: {det['width']:.0f}x{det['height']:.0f}px")
    else:
        print("\n[WARNING] No single '6' or '9' detected by Vision API even with 2x zoom!")

    print("=" * 80)


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv('GOOGLE_VISION_API_KEY')
    if not api_key:
        print("[ERROR] GOOGLE_VISION_API_KEY not found in environment")
        print("Please set the API key: export GOOGLE_VISION_API_KEY=your_key_here")
        sys.exit(1)

    # Default to page 5 preprocessed image
    image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_phase1_preprocessed.png"

    # Allow override from command line
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    test_vision_api_full_zoom(image_path, api_key)
