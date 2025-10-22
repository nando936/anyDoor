"""
Extract handwritten door order by cropping known table regions
Then OCR each table separately for better accuracy
"""
import os
import sys
import json
import base64
import requests
import cv2
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv('GOOGLE_VISION_API_KEY')

if not google_api_key:
    print("[ERROR] GOOGLE_VISION_API_KEY not found in .env file")
    sys.exit(1)

def ocr_with_google_vision(image_array):
    """OCR a cropped image array using Google Vision API"""

    # Encode image
    _, buffer = cv2.imencode('.jpg', image_array)
    image_data = base64.standard_b64encode(buffer).decode('utf-8')

    url = f"https://vision.googleapis.com/v1/images:annotate?key={google_api_key}"
    request_body = {
        "requests": [{
            "image": {"content": image_data},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, json=request_body)
    if response.status_code != 200:
        print(f"[ERROR] Vision API failed: {response.status_code}")
        return None

    result = response.json()
    if 'responses' in result and len(result['responses']) > 0:
        text_annotations = result['responses'][0].get('textAnnotations', [])
        if text_annotations:
            return text_annotations[0].get('description', '')

    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_with_manual_crops.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Convert path if needed
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # Read the full image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    image_height, image_width = image.shape[:2]
    print(f"[INFO] Image size: {image_width}x{image_height}")

    # Get output directory
    image_dir = os.path.dirname(image_path)
    image_name = Path(image_path).stem

    # Define table regions (based on visual inspection)
    # Format: (room, type, top%, bottom%, left%, right%)
    tables = [
        ("Living_Room", "doors", 15, 20, 0, 50),
        ("Pantry", "doors", 23, 62, 0, 50),
        ("Pantry", "drawers", 23, 62, 50, 100),
        ("White_Oak", "doors", 65, 72, 0, 50),
        ("White_Oak", "drawers", 65, 75, 50, 100),
    ]

    # Process each table
    for room, table_type, top_pct, bottom_pct, left_pct, right_pct in tables:
        print(f"\n[INFO] Processing {room} {table_type}...")

        # Calculate crop boundaries
        top = int(image_height * top_pct / 100)
        bottom = int(image_height * bottom_pct / 100)
        left = int(image_width * left_pct / 100)
        right = int(image_width * right_pct / 100)

        # Crop
        cropped = image[top:bottom, left:right]

        # Save cropped image for debugging
        crop_name = f"{room}_{table_type}"
        crop_path = os.path.join(image_dir, f"{image_name}_crop_{crop_name}.jpg")
        cv2.imwrite(crop_path, cropped)
        print(f"[OK] Saved crop: {crop_path}")

        # OCR the cropped table
        ocr_text = ocr_with_google_vision(cropped)

        if ocr_text:
            # Save OCR text
            ocr_path = os.path.join(image_dir, f"{image_name}_ocr_{crop_name}.txt")
            with open(ocr_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            print(f"[OK] Saved OCR: {ocr_path}")
        else:
            print(f"[WARNING] No OCR text for {crop_name}")

    print(f"\n[OK] All crops and OCR saved to {image_dir}")

if __name__ == '__main__':
    main()
