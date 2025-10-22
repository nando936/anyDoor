#!/usr/bin/env python3
"""
Test Google Vision API on M7 down arrow detection image
"""

import os
import cv2
import base64
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def analyze_arrow_image(image_path):
    """Analyze arrow image with Google Vision API"""

    # Get API key
    api_key = os.getenv('GOOGLE_VISION_API_KEY')
    if not api_key:
        print("[ERROR] GOOGLE_VISION_API_KEY not found in environment")
        return

    # Read and encode the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(image_path)}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"{'='*80}\n")

    # Build Vision API request
    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    request_body = {
        "requests": [{
            "image": {"content": image_base64},
            "features": [
                {"type": "TEXT_DETECTION"},
                {"type": "LABEL_DETECTION", "maxResults": 10},
                {"type": "OBJECT_LOCALIZATION"},
                {"type": "LOGO_DETECTION"}
            ]
        }]
    }

    # Make API request
    response = requests.post(vision_url, json=request_body)

    if response.status_code != 200:
        print(f"[ERROR] API request failed: {response.status_code}")
        print(response.text)
        return

    result = response.json()
    annotations = result['responses'][0]

    # Text detection
    print("TEXT DETECTION:")
    print("-" * 80)
    if 'textAnnotations' in annotations:
        texts = annotations['textAnnotations']
        print(f"Found {len(texts)} text items")
        for i, text in enumerate(texts[:5]):
            print(f"\n  {i+1}. Text: '{text['description']}'")
            if 'boundingPoly' in text:
                vertices = [(v['x'], v['y']) for v in text['boundingPoly']['vertices']]
                print(f"     Bounds: {vertices}")
    else:
        print("  No text detected")

    # Label detection
    print("\n\nLABEL DETECTION:")
    print("-" * 80)
    if 'labelAnnotations' in annotations:
        labels = annotations['labelAnnotations']
        print(f"Found {len(labels)} labels")
        for label in labels:
            print(f"  - {label['description']} (confidence: {label['score']:.2%})")
    else:
        print("  No labels detected")

    # Object localization
    print("\n\nOBJECT LOCALIZATION:")
    print("-" * 80)
    if 'localizedObjectAnnotations' in annotations:
        objects = annotations['localizedObjectAnnotations']
        print(f"Found {len(objects)} objects")
        for obj in objects:
            print(f"\n  - {obj['name']} (confidence: {obj['score']:.2%})")
            if 'boundingPoly' in obj:
                vertices = [(v['x'], v['y']) for v in obj['boundingPoly']['normalizedVertices']]
                print(f"    Bounds: {vertices}")
    else:
        print("  No objects detected")

    # Logo detection
    print("\n\nLOGO DETECTION:")
    print("-" * 80)
    if 'logoAnnotations' in annotations:
        logos = annotations['logoAnnotations']
        print(f"Found {len(logos)} logos")
        for logo in logos:
            print(f"  - {logo['description']} (confidence: {logo['score']:.2%})")
    else:
        print("  No logos detected")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Test on M7 arrow detection image
    image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page4_M7_22-7-8_down_arrow_detection.png"

    if os.path.exists(image_path):
        analyze_arrow_image(image_path)
    else:
        print(f"Image not found: {image_path}")

    # Also test the claude verification image
    claude_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page4_M7_227-8_claude_verification.png"
    if os.path.exists(claude_path):
        analyze_arrow_image(claude_path)
