#!/usr/bin/env python3
"""
Test Vision API specifically on the left side where we expect to find the other "23".
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

def enhance_green_text(image):
    """Enhance green text in the image using HSV."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green color range in HSV
    lower_green = np.array([35, 20, 20])
    upper_green = np.array([85, 255, 255])

    # Create mask for green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Create white background
    result = np.ones_like(image) * 255

    # Apply green pixels as black text on white background
    result[mask > 0] = [0, 0, 0]

    return result

def test_vision_on_crop(image, crop_bounds, label=""):
    """Test Vision API on a cropped region."""
    x1, y1, x2, y2 = crop_bounds
    cropped = image[y1:y2, x1:x2]

    # Save original crop
    crop_file = f"DEBUG_left_23_{label}_original.png"
    cv2.imwrite(crop_file, cropped)
    print(f"\nSaved original crop: {crop_file}")

    # Try HSV enhancement
    enhanced = enhance_green_text(cropped)
    enhanced_file = f"DEBUG_left_23_{label}_enhanced.png"
    cv2.imwrite(enhanced_file, enhanced)
    print(f"Saved enhanced crop: {enhanced_file}")

    # Initialize Vision API
    api_key = os.environ.get('GOOGLE_VISION_API_KEY') or os.environ.get('GOOGLE_CLOUD_API_KEY')
    if not api_key:
        print("Error: API key not set")
        return

    options = client_options.ClientOptions(api_key=api_key)
    client = vision.ImageAnnotatorClient(client_options=options)

    # Test both versions
    for img, img_type in [(cropped, "Original"), (enhanced, "Enhanced")]:
        print(f"\n{img_type} image:")
        print("-" * 40)

        # Encode image
        _, encoded = cv2.imencode('.png', img)
        content = encoded.tobytes()

        image_vision = vision.Image(content=content)

        # Detect text
        response = client.text_detection(image=image_vision)
        texts = response.text_annotations

        if not texts:
            print("  No text detected!")
            continue

        # Show full text
        full_text = texts[0].description if texts else ""
        print(f"  Full text: {full_text}")

        # Show individual items
        items = texts[1:] if len(texts) > 1 else []
        print(f"  Individual items ({len(items)}):")
        for text in items:
            desc = text.description.strip()
            vertices = text.bounding_poly.vertices
            x = sum(v.x for v in vertices) // 4
            y = sum(v.y for v in vertices) // 4
            print(f"    '{desc}' at ({x}, {y})")

            # Check for "23" or related
            if "23" in desc or desc in ["2", "3"]:
                print(f"      [!] Found relevant: '{desc}'")

def main():
    image_path = "page_3.png" if len(sys.argv) < 2 else sys.argv[1]

    print("TESTING LEFT SIDE FOR MISSING '23'")
    print("=" * 60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Image dimensions: {w}x{h}")

    # Test different crop regions on the left side
    # Based on visual inspection, the left "23" should be around x=200-400

    # Crop 1: Upper left area (above the drawers)
    print("\n\nTEST 1: Upper left quadrant")
    test_vision_on_crop(image, (0, 350, 500, 550), "upper_left")

    # Crop 2: Middle left area (near the drawers)
    print("\n\nTEST 2: Middle left area")
    test_vision_on_crop(image, (100, 450, 400, 650), "middle_left")

    # Crop 3: Lower middle left (below drawers)
    print("\n\nTEST 3: Lower left area")
    test_vision_on_crop(image, (150, 550, 450, 750), "lower_left")

    # Crop 4: Very specific area based on looking at image
    # The left 23 appears to be near the left drawer width
    print("\n\nTEST 4: Targeted area for left 23")
    test_vision_on_crop(image, (200, 475, 350, 525), "targeted")

if __name__ == "__main__":
    main()