#!/usr/bin/env python3
"""
Test Vision API on grayscale version of page 3 to improve detection of both 23 dimensions.
"""

import cv2
import numpy as np
import os
import sys
import json
from google.cloud import vision
from google.oauth2 import service_account
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

def create_grayscale_image(input_path, output_path):
    """Convert image to grayscale and save it."""
    print(f"Reading image: {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image from {input_path}")
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save grayscale image
    cv2.imwrite(output_path, gray)
    print(f"Saved grayscale image to: {output_path}")

    # Show image dimensions
    height, width = gray.shape
    print(f"Image dimensions: {width}x{height}")

    return True

def test_vision_api(image_path):
    """Run Vision API on the image and look for '23' measurements."""
    print(f"\nTesting Vision API on: {image_path}")

    # Initialize Vision API client
    # Try both possible environment variable names
    api_key = os.environ.get('GOOGLE_VISION_API_KEY') or os.environ.get('GOOGLE_CLOUD_API_KEY')
    if not api_key:
        print("Error: GOOGLE_VISION_API_KEY environment variable not set")
        return None

    # Use API key directly without service account credentials
    from google.api_core import client_options
    options = client_options.ClientOptions(api_key=api_key)
    client = vision.ImageAnnotatorClient(client_options=options)

    # Read image
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Detect text
    print("Running text detection...")
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("No text detected!")
        return []

    # Skip the first annotation which contains all text
    text_annotations = texts[1:] if len(texts) > 1 else []

    print(f"\nTotal text blocks detected: {len(text_annotations)}")

    # Look for "23" measurements
    twenty_three_found = []
    all_measurements = []

    for text in text_annotations:
        description = text.description.strip()

        # Track all potential measurements
        if any(char.isdigit() for char in description):
            vertices = text.bounding_poly.vertices
            x_coords = [v.x for v in vertices]
            y_coords = [v.y for v in vertices]
            center_x = sum(x_coords) // 4
            center_y = sum(y_coords) // 4

            measurement_info = {
                'text': description,
                'center': (center_x, center_y),
                'bounds': {
                    'x_min': min(x_coords),
                    'x_max': max(x_coords),
                    'y_min': min(y_coords),
                    'y_max': max(y_coords)
                }
            }

            all_measurements.append(measurement_info)

            # Special tracking for "23"
            if description == "23":
                twenty_three_found.append(measurement_info)

    # Display findings
    print(f"\n[OK] Found {len(twenty_three_found)} instances of '23':")
    for i, item in enumerate(twenty_three_found, 1):
        print(f"  {i}. '23' at position ({item['center'][0]}, {item['center'][1]})")

    # Show all numeric text for context
    print(f"\nAll numeric text detected ({len(all_measurements)} items):")
    # Group by approximate Y position (within 50 pixels)
    sorted_measurements = sorted(all_measurements, key=lambda x: (x['center'][1], x['center'][0]))

    current_y = -100
    for m in sorted_measurements:
        if abs(m['center'][1] - current_y) > 50:
            print()  # New line for different Y level
            current_y = m['center'][1]
        print(f"  '{m['text']}' at ({m['center'][0]}, {m['center'][1]})", end="  ")
    print()

    return twenty_three_found, all_measurements

def main():
    """Main function."""
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    else:
        input_image = "page_3.png"

    # Create grayscale version
    grayscale_image = input_image.replace('.png', '_grayscale.png')

    print("="*60)
    print("GRAYSCALE VISION API TEST FOR PAGE 3")
    print("="*60)

    # Convert to grayscale
    if not create_grayscale_image(input_image, grayscale_image):
        print("Failed to create grayscale image")
        return

    # Test both original and grayscale
    print("\n" + "="*60)
    print("TESTING ORIGINAL COLOR IMAGE")
    print("="*60)
    color_23s, color_all = test_vision_api(input_image)

    print("\n" + "="*60)
    print("TESTING GRAYSCALE IMAGE")
    print("="*60)
    grayscale_23s, grayscale_all = test_vision_api(grayscale_image)

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Color image: Found {len(color_23s) if color_23s else 0} instances of '23'")
    print(f"Grayscale image: Found {len(grayscale_23s) if grayscale_23s else 0} instances of '23'")

    if grayscale_23s and len(grayscale_23s) > (len(color_23s) if color_23s else 0):
        print("\n[OK] SUCCESS! Grayscale improved detection!")
    elif grayscale_23s and len(grayscale_23s) == (len(color_23s) if color_23s else 0):
        print("\n[!] No change in detection between color and grayscale")
    else:
        print("\n[X] Grayscale did not improve detection")

    # Save results
    results = {
        'original': {
            'twenty_three_count': len(color_23s) if color_23s else 0,
            'twenty_three_positions': [{'text': '23', 'center': t['center']} for t in (color_23s or [])]
        },
        'grayscale': {
            'twenty_three_count': len(grayscale_23s) if grayscale_23s else 0,
            'twenty_three_positions': [{'text': '23', 'center': t['center']} for t in (grayscale_23s or [])]
        }
    }

    output_file = 'grayscale_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()