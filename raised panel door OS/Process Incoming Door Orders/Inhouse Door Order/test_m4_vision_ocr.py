#!/usr/bin/env python3
"""
Test script to check what Google Vision API detects on M4 DEBUG crop image.
Shows ALL text detected with confidence scores and bounding box coordinates.
"""

import sys
import os
import base64
import requests
import cv2
from dotenv import load_dotenv

# Load environment variables from .env file
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


def test_m4_vision_ocr(image_path, api_key, output_file):
    """Test Vision API detection on M4 DEBUG crop image."""

    print(f"Testing Vision API on: {image_path}")
    print("=" * 80)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}px")

    # Call Vision API
    print("\nCalling Google Vision API...")
    texts, error = call_vision_api(image, api_key)

    if error:
        print(f"[ERROR] {error}")
        return

    if not texts:
        print("[ERROR] No text detected")
        return

    print(f"\nFound {len(texts) - 1} text items")
    print("=" * 80)

    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Google Vision API Test - M4 DEBUG Crop Image\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Image size: {w}x{h}px\n")
        f.write(f"Total text items: {len(texts) - 1}\n")
        f.write("=" * 80 + "\n\n")

        # First item is full text - save separately
        if texts:
            f.write("[FULL TEXT DETECTED]\n")
            f.write(texts[0]['description'] + "\n")
            f.write("-" * 80 + "\n\n")

        # Track all detections
        all_detections = []

        # Process individual text detections (skip first full text)
        f.write("[ALL TEXT ITEMS DETECTED]\n")
        f.write("-" * 80 + "\n\n")

        for i, text_ann in enumerate(texts[1:], 1):
            description = text_ann['description']

            # Get bounding box
            vertices = text_ann['boundingPoly']['vertices']
            x_coords = [v.get('x', 0) for v in vertices]
            y_coords = [v.get('y', 0) for v in vertices]

            # Calculate center and size
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)

            # Get bounding box corners
            left = min(x_coords)
            right = max(x_coords)
            top = min(y_coords)
            bottom = max(y_coords)

            # Build detection info
            detection = {
                'index': i,
                'text': description,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
                'bbox': vertices
            }

            all_detections.append(detection)

            # Format output with detailed info
            output_line = (
                f"  {i:3d}. TEXT: '{description}'\n"
                f"       Center: ({center_x:.1f}, {center_y:.1f})\n"
                f"       Size: {width:.1f}x{height:.1f}px\n"
                f"       Bounds: left={left}, right={right}, top={top}, bottom={bottom}\n"
                f"       BBox vertices: {vertices}\n"
                f"\n"
            )
            f.write(output_line)
            print(output_line.rstrip())

        # Summary section
        f.write("=" * 80 + "\n")
        f.write("\n[SUMMARY]\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total text items detected: {len(all_detections)}\n")

        print("=" * 80)
        print("\n[SUMMARY]")
        print("-" * 80)
        print(f"Total text items detected: {len(all_detections)}")

        # List all detected text
        if all_detections:
            f.write("\n[ALL DETECTED TEXT]\n")
            f.write("-" * 80 + "\n")
            print("\n[ALL DETECTED TEXT]")
            print("-" * 80)
            for det in all_detections:
                line = f"  '{det['text']}' at ({det['center_x']:.0f}, {det['center_y']:.0f})\n"
                f.write(line)
                print(line.rstrip())

        f.write("\n" + "=" * 80 + "\n")
        print("=" * 80)

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv('GOOGLE_VISION_API_KEY')
    if not api_key:
        print("[ERROR] GOOGLE_VISION_API_KEY not found in environment")
        print("Please set the API key in .env file or environment")
        sys.exit(1)

    # M4 DEBUG crop image path
    image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/DEBUG_page7_M4_pos363x1203_–9_zoom1x.png"

    # Output file path
    output_file = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/test_m4_vision_ocr_result.txt"

    # Allow override from command line
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = os.path.join(image_dir, f"{base_name}_vision_ocr_result.txt")

    test_m4_vision_ocr(image_path, api_key, output_file)
