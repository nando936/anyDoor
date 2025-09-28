#!/usr/bin/env python3
"""
Debug script to see exactly what Vision API returns with and without HSV preprocessing.
"""

import cv2
import numpy as np
import os
from google.cloud import vision
from google.api_core import client_options
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

def apply_hsv_preprocessing(image):
    """Apply HSV preprocessing to isolate green text"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    return mask_inv

def test_vision_api(image, label):
    """Test Vision API on an image"""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print('='*60)

    # Initialize Vision API
    api_key = os.environ.get('GOOGLE_VISION_API_KEY') or os.environ.get('GOOGLE_CLOUD_API_KEY')
    if not api_key:
        print("Error: API key not set")
        return

    options = client_options.ClientOptions(api_key=api_key)
    client = vision.ImageAnnotatorClient(client_options=options)

    # Encode image
    _, encoded = cv2.imencode('.png', image)
    content = encoded.tobytes()

    image_vision = vision.Image(content=content)

    # Detect text
    response = client.text_detection(image=image_vision)
    texts = response.text_annotations

    if not texts:
        print("No text detected!")
        return

    # Show full text
    full_text = texts[0].description if texts else ""
    print(f"FULL TEXT:\n{full_text}\n")

    # Count occurrences of "23"
    items = texts[1:] if len(texts) > 1 else []
    twenty_three_items = []
    two_items = []
    three_items = []

    print(f"Individual items ({len(items)}):")
    for text in items:
        desc = text.description.strip()
        vertices = text.bounding_poly.vertices
        x = sum(v.x for v in vertices) // 4
        y = sum(v.y for v in vertices) // 4

        # Only print number-related items
        if any(c.isdigit() for c in desc):
            print(f"  '{desc}' at ({x}, {y})")

        if desc == "23":
            twenty_three_items.append((x, y))
        elif desc == "2":
            two_items.append((x, y))
        elif desc == "3":
            three_items.append((x, y))

    print(f"\nSUMMARY:")
    print(f"  Found {len(twenty_three_items)} instances of '23': {twenty_three_items}")
    print(f"  Found {len(two_items)} instances of '2': {two_items}")
    print(f"  Found {len(three_items)} instances of '3': {three_items}")

def main():
    image_path = "page_3.png"

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    # Test original image
    test_vision_api(image, "ORIGINAL IMAGE")

    # Test with HSV preprocessing
    hsv_processed = apply_hsv_preprocessing(image)
    test_vision_api(hsv_processed, "HSV PREPROCESSED IMAGE")

    # Save HSV processed image for inspection
    cv2.imwrite("page_3_hsv_processed.png", hsv_processed)
    print(f"\nSaved HSV processed image as page_3_hsv_processed.png")

if __name__ == "__main__":
    main()