"""
Test the actual zoomed image from 2nd pass with HSV masking
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

def test_zoomed_image_with_hsv(api_key):
    """Test the exact zoomed image from 2nd pass with HSV preprocessing"""

    # Load the exact zoomed image that the 2nd pass creates
    zoomed_image_path = "debug_914_EXACT_2nd_pass_image.png"

    image = cv2.imread(zoomed_image_path)
    if image is None:
        print(f"Cannot load {zoomed_image_path}")
        return

    print(f"Testing zoomed image: {zoomed_image_path}")
    print(f"Image shape: {image.shape}")
    print("="*60)

    # Test 1: Direct OCR on zoomed image (baseline)
    print("\n[1] Direct OCR on zoomed image (current behavior):")
    _, buffer = cv2.imencode('.png', image)
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
                print(f"Result: '{full_text.strip()}'")

    # Test 2: HSV mask preprocessing
    print("\n[2] With HSV mask preprocessing:")

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green color range
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # Save for inspection
    cv2.imwrite("zoomed_hsv_mask.png", mask_inv)
    print(f"Saved HSV mask: zoomed_hsv_mask.png")

    # OCR on HSV mask
    _, buffer = cv2.imencode('.png', mask_inv)
    content = base64.b64encode(buffer).decode('utf-8')

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
                print(f"Result: '{full_text.strip()}'")

                # Show individual items
                print("\nIndividual items:")
                for ann in annotations[1:]:
                    print(f"  - '{ann['description']}'")

    # Test 3: Try different preprocessing
    print("\n[3] Alternative preprocessing (broader green range):")

    # Broader green range
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    cv2.imwrite("zoomed_hsv_broad.png", mask_inv)

    _, buffer = cv2.imencode('.png', mask_inv)
    content = base64.b64encode(buffer).decode('utf-8')

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
                print(f"Result: '{full_text.strip()}'")

def main():
    API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')
    if not API_KEY:
        print("Set GOOGLE_VISION_API_KEY")
        return

    print("TESTING ZOOMED 2ND PASS IMAGE WITH HSV PREPROCESSING")
    print("="*60)

    test_zoomed_image_with_hsv(API_KEY)

    print("\n" + "="*60)
    print("SUMMARY:")
    print("If HSV preprocessing fixes the zoomed image,")
    print("we should add it to verify_measurement_with_zoom function")

if __name__ == "__main__":
    main()