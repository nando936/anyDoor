"""
Test Google Vision API on the HSV mask preprocessed image
"""
import cv2
import base64
import requests
import numpy as np
import os
from dotenv import load_dotenv
import sys

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

def create_hsv_mask_and_test(image_path, x, y, api_key):
    """Create HSV mask and test with Google Vision API"""

    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print("Cannot load image")
        return

    h, w = image.shape[:2]

    # Extract region around the "914" position
    padding = 40
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(w, int(x + padding))
    y2 = min(h, int(y + padding))

    cropped = image[y1:y2, x1:x2]

    # Convert to HSV
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Green color range in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create mask for green
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert so text is black on white
    mask_inv = cv2.bitwise_not(mask)

    # Save the mask
    cv2.imwrite("hsv_mask_for_vision.png", mask_inv)
    print(f"Created HSV mask image: hsv_mask_for_vision.png")
    print(f"Image shape: {mask_inv.shape}")

    # Encode for Vision API
    _, buffer = cv2.imencode('.png', mask_inv)
    content = base64.b64encode(buffer).decode('utf-8')

    # Call Vision API
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    print("\nCalling Google Vision API on HSV mask...")
    response = requests.post(url, json=request)

    if response.status_code == 200:
        result = response.json()
        if 'responses' in result and result['responses']:
            annotations = result['responses'][0].get('textAnnotations', [])
            if annotations:
                # Full text
                full_text = annotations[0]['description']
                print(f"\nVision API Result:")
                print(f"Full text: '{full_text.strip()}'")

                # Individual items
                print("\nIndividual text items detected:")
                for ann in annotations[1:]:
                    text = ann['description']
                    print(f"  - '{text}'")

                return full_text.strip()
    else:
        print(f"API Error: {response.status_code}")

    return None

def test_with_different_masks(image_path, x, y, api_key):
    """Try different HSV ranges to find optimal mask"""

    image = cv2.imread(image_path)
    if image is None:
        return

    h, w = image.shape[:2]

    # Extract region
    padding = 40
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(w, int(x + padding))
    y2 = min(h, int(y + padding))

    cropped = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Try different HSV ranges
    ranges = [
        ("Narrow green", [45, 50, 50], [75, 255, 255]),
        ("Wide green", [35, 30, 30], [85, 255, 255]),
        ("Very wide", [30, 20, 20], [90, 255, 255]),
    ]

    print("\nTrying different HSV ranges:")
    print("="*60)

    for name, lower, upper in ranges:
        print(f"\n{name}: {lower} to {upper}")

        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask_inv = cv2.bitwise_not(mask)

        # Save
        filename = f"hsv_{name.replace(' ', '_')}.png"
        cv2.imwrite(filename, mask_inv)

        # Test with Vision API
        _, buffer = cv2.imencode('.png', mask_inv)
        content = base64.b64encode(buffer).decode('utf-8')

        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        request = {
            "requests": [{
                "image": {"content": content},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }

        response = requests.post(url, json=request)

        if response.status_code == 200:
            result = response.json()
            if 'responses' in result and result['responses']:
                annotations = result['responses'][0].get('textAnnotations', [])
                if annotations:
                    full_text = annotations[0]['description']
                    print(f"  Result: '{full_text.strip()}'")

def main():
    API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')
    if not API_KEY:
        print("Set GOOGLE_VISION_API_KEY")
        return

    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3.png"

    # Position of "914"
    x, y = 701, 1264

    print("TESTING GOOGLE VISION API WITH HSV MASK")
    print("="*60)

    print(f"\nTesting position ({x}, {y}) where '914' is detected:")
    result = create_hsv_mask_and_test(image_path, x, y, API_KEY)

    if result:
        print(f"\n[SUCCESS] Vision API read: '{result}'")
        if "9 1/4" in result or "9 1 / 4" in result:
            print("  ✓ Correctly detected as '9 1/4'!")

    # Try different ranges
    print("\n\nTrying variations...")
    test_with_different_masks(image_path, x, y, API_KEY)

if __name__ == "__main__":
    main()