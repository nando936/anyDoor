#!/usr/bin/env python3
"""
Test script to check what Google Vision API detects on page 6.
Specifically looks for "F" notation detection near the "6 3/4" measurement (M1).
Shows ALL text detected and highlights items near M1 position.
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


def test_vision_f_detection(image_path, api_key, output_file):
    """Test Vision API detection and specifically look for F notation near M1."""

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
        f.write(f"Google Vision API Test - Page 6 F Notation Detection\n")
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
        f_detections = []
        measurement_related = []

        # Known M1 position (approximate) - "6 3/4" measurement
        # We'll look for text near x=500-800, y=500-700 range
        m1_x_range = (500, 800)
        m1_y_range = (500, 700)

        # Process individual text detections (skip first full text)
        f.write("[ALL TEXT ITEMS DETECTED]\n")
        f.write("-" * 80 + "\n")

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

            # Check if near M1 position
            near_m1 = (m1_x_range[0] <= center_x <= m1_x_range[1] and
                      m1_y_range[0] <= center_y <= m1_y_range[1])

            # Check if it's "F" notation
            is_f_notation = description.upper() == 'F'

            # Build detection info
            detection = {
                'index': i,
                'text': description,
                'x': center_x,
                'y': center_y,
                'width': width,
                'height': height,
                'near_m1': near_m1,
                'is_f': is_f_notation,
                'bbox': vertices
            }

            all_detections.append(detection)

            # Track special cases
            if is_f_notation:
                f_detections.append(detection)

            if near_m1 or '6' in description or '3/4' in description or is_f_notation:
                measurement_related.append(detection)

            # Format output
            marker = ""
            if near_m1 and is_f_notation:
                marker = " *** NEAR M1 + F NOTATION ***"
            elif near_m1:
                marker = " *** NEAR M1 ***"
            elif is_f_notation:
                marker = " *** F NOTATION ***"

            output_line = f"  {i:3d}. '{description}' at ({center_x:.0f}, {center_y:.0f}) size: {width:.0f}x{height:.0f}px{marker}\n"
            f.write(output_line)
            print(output_line.rstrip())

        # Summary section
        f.write("\n" + "=" * 80 + "\n")
        f.write("\n[SUMMARY]\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total text items: {len(all_detections)}\n")
        f.write(f"'F' notations found: {len(f_detections)}\n")
        f.write(f"Text near M1 region: {len([d for d in all_detections if d['near_m1']])}\n")
        f.write(f"Measurement-related text: {len(measurement_related)}\n\n")

        print("\n" + "=" * 80)
        print("\n[SUMMARY]")
        print("-" * 80)
        print(f"Total text items: {len(all_detections)}")
        print(f"'F' notations found: {len(f_detections)}")
        print(f"Text near M1 region: {len([d for d in all_detections if d['near_m1']])}")
        print(f"Measurement-related text: {len(measurement_related)}")

        # F Notation details
        if f_detections:
            f.write("\n[ALL 'F' NOTATIONS DETECTED]\n")
            f.write("-" * 80 + "\n")
            print("\n[ALL 'F' NOTATIONS DETECTED]")
            print("-" * 80)
            for det in f_detections:
                near = " <-- NEAR M1" if det['near_m1'] else ""
                line = f"  '{det['text']}' at ({det['x']:.0f}, {det['y']:.0f}) size: {det['width']:.0f}x{det['height']:.0f}px{near}\n"
                f.write(line)
                print(line.rstrip())
        else:
            msg = "\n[WARNING] No 'F' notation detected by Vision API!\n"
            f.write(msg)
            print(msg.rstrip())

        # M1 area details
        if measurement_related:
            f.write("\n[TEXT NEAR M1 OR MEASUREMENT-RELATED]\n")
            f.write("-" * 80 + "\n")
            f.write(f"Looking for text in region: x={m1_x_range}, y={m1_y_range}\n\n")
            print("\n[TEXT NEAR M1 OR MEASUREMENT-RELATED]")
            print("-" * 80)
            print(f"Looking for text in region: x={m1_x_range}, y={m1_y_range}\n")
            for det in measurement_related:
                markers = []
                if det['near_m1']:
                    markers.append("NEAR M1")
                if det['is_f']:
                    markers.append("F NOTATION")
                if '6' in det['text']:
                    markers.append("CONTAINS '6'")
                if '3/4' in det['text']:
                    markers.append("CONTAINS '3/4'")

                marker_str = " [" + ", ".join(markers) + "]" if markers else ""
                line = f"  '{det['text']}' at ({det['x']:.0f}, {det['y']:.0f}) size: {det['width']:.0f}x{det['height']:.0f}px{marker_str}\n"
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

    # Default to page 6 image
    image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page_6.png"

    # Generate output filename in same directory
    image_dir = os.path.dirname(image_path)
    output_file = os.path.join(image_dir, "page_6_vision_f_detection_test.txt")

    # Allow override from command line
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = os.path.join(image_dir, f"{base_name}_vision_f_detection_test.txt")

    test_vision_f_detection(image_path, api_key, output_file)
