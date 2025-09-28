#!/usr/bin/env python3
"""
Test Vision API on the exact area where left 23 should be - same X as 20 1/16 but higher.
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

def test_specific_area(image_path):
    """Test the specific area where left 23 should be."""

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Image dimensions: {w}x{h}")

    # Based on the measurements data, 20 1/16 is at X=336, Y=874
    # So 23 should be at similar X (around 300-350) but higher Y (maybe 750-850)

    # Test multiple crops in that area
    test_areas = [
        (250, 700, 450, 850, "area1_wide"),  # Wide area
        (280, 730, 380, 830, "area2_focused"),  # More focused
        (300, 750, 370, 820, "area3_narrow"),  # Narrow focus
        (250, 780, 400, 860, "area4_lower"),  # A bit lower
    ]

    # Initialize Vision API
    api_key = os.environ.get('GOOGLE_VISION_API_KEY') or os.environ.get('GOOGLE_CLOUD_API_KEY')
    if not api_key:
        print("Error: API key not set")
        return

    options = client_options.ClientOptions(api_key=api_key)
    client = vision.ImageAnnotatorClient(client_options=options)

    for x1, y1, x2, y2, label in test_areas:
        print(f"\n{'='*60}")
        print(f"Testing {label}: [{x1},{y1}] to [{x2},{y2}]")
        print(f"{'='*60}")

        # Crop image
        cropped = image[y1:y2, x1:x2]

        # Save crop
        crop_file = f"DEBUG_exact_23_{label}.png"
        cv2.imwrite(crop_file, cropped)
        print(f"Saved crop: {crop_file}")

        # Also try HSV enhanced version
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 20, 20])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        enhanced = np.ones_like(cropped) * 255
        enhanced[mask > 0] = [0, 0, 0]

        enhanced_file = f"DEBUG_exact_23_{label}_hsv.png"
        cv2.imwrite(enhanced_file, enhanced)

        # Test both versions
        for img, img_type in [(cropped, "Original"), (enhanced, "HSV Enhanced")]:
            print(f"\n{img_type}:")

            # Encode image
            _, encoded = cv2.imencode('.png', img)
            content = encoded.tobytes()

            image_vision = vision.Image(content=content)

            # Detect text
            response = client.text_detection(image=image_vision)
            texts = response.text_annotations

            if not texts:
                print("  No text detected")
                continue

            # Get full text (safely, avoiding Unicode errors)
            try:
                full_text = texts[0].description if texts else ""
                # Clean up any problematic characters
                full_text = full_text.encode('ascii', 'ignore').decode('ascii')
                print(f"  Full text: {full_text}")
            except:
                print("  Full text: [Unicode error - skipping]")

            # Show individual items
            items = texts[1:] if len(texts) > 1 else []
            print(f"  Found {len(items)} text items:")

            for text in items:
                desc = text.description.strip()
                vertices = text.bounding_poly.vertices
                x = sum(v.x for v in vertices) // 4 + x1  # Adjust for crop
                y = sum(v.y for v in vertices) // 4 + y1
                print(f"    '{desc}' at image pos ({x}, {y})")

                # Check for 23 or components
                if "23" in desc:
                    print(f"      [OK] FOUND '23'!")
                elif desc == "2":
                    print(f"      [!] Found '2' - might be part of 23")
                elif desc == "3":
                    print(f"      [!] Found '3' - might be part of 23")

def main():
    image_path = "page_3.png" if len(sys.argv) < 2 else sys.argv[1]

    print("SEARCHING FOR LEFT '23' AT SAME X AS '20 1/16'")
    print("Based on data: 20 1/16 is at X=336, so 23 should be around X=300-350")
    print("="*60)

    test_specific_area(image_path)

if __name__ == "__main__":
    main()