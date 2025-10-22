#!/usr/bin/env python3
"""
Test Google Vision API OCR on a specific debug image
"""
import cv2
import base64
import requests
import sys
import os

def test_vision_ocr(image_path, api_key):
    """Test Vision API on an image"""
    print(f"Testing Vision API OCR on: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Image type: {image.dtype}, channels: {image.shape[2] if len(image.shape) == 3 else 1}")

    # Encode image
    _, buffer = cv2.imencode('.png', image)
    content = base64.b64encode(buffer).decode('utf-8')

    # Call Vision API
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    print("\nCalling Vision API...")
    response = requests.post(url, json=request)

    if response.status_code != 200:
        print(f"ERROR: Vision API failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return

    result = response.json()

    if 'responses' not in result or not result['responses']:
        print("ERROR: No responses from Vision API")
        return

    annotations = result['responses'][0].get('textAnnotations', [])

    if not annotations:
        print("\n*** VISION API RESULT: NO TEXT DETECTED ***")
    else:
        print(f"\n*** VISION API RESULT: Found {len(annotations)} text annotations ***")
        print("\nFull text annotation:")
        print(f"  '{annotations[0]['description']}'")

        print("\nIndividual text items:")
        for i, ann in enumerate(annotations[1:], 1):
            print(f"  {i}. '{ann['description']}'")


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv('GOOGLE_VISION_API_KEY')

    if not api_key:
        print("ERROR: GOOGLE_VISION_API_KEY environment variable not set")
        sys.exit(1)

    # Default image path
    image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/DEBUG_page5_M15_pos316x791_9_hsv_zoom1x.png"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    test_vision_ocr(image_path, api_key)
