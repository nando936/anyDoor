"""
Debug script to investigate why the second "20 1/16" dimension on page 3 isn't being detected.
Shows OCR results from both initial pass and zoomed verification.
"""
import cv2
import numpy as np
import requests
import base64
import json
import re
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

def get_ocr_from_region(image_content_base64, api_key):
    """Run OCR on an image region"""
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request = {
        "requests": [{
            "image": {"content": image_content_base64},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, json=request)

    if response.status_code == 200:
        result = response.json()
        if 'responses' in result and result['responses']:
            annotations = result['responses'][0].get('textAnnotations', [])
            return annotations
    return []

def debug_ocr_at_position(image_path, x, y, api_key, label=""):
    """Debug OCR detection at a specific position with zoom"""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    print(f"\n{'='*60}")
    print(f"DEBUG: {label} at position ({x:.0f}, {y:.0f})")
    print('='*60)

    # Create a zoomed crop area
    padding_x = 200
    padding_y = 50
    x1 = max(0, int(x - padding_x))
    y1 = max(0, int(y - padding_y))
    x2 = min(w, int(x + padding_x))
    y2 = min(h, int(y + padding_y))

    # Crop and zoom
    cropped = image[y1:y2, x1:x2]
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Save the zoomed region for visualization
    debug_filename = f"debug_zoom_{label.replace(' ', '_').replace('/', '-')}.png"
    cv2.imwrite(debug_filename, zoomed)
    print(f"Saved zoomed region to: {debug_filename}")

    # Run OCR on zoomed region
    _, buffer = cv2.imencode('.png', zoomed)
    zoomed_content = base64.b64encode(buffer).decode('utf-8')

    annotations = get_ocr_from_region(zoomed_content, api_key)

    if annotations:
        # First annotation is full text
        full_text = annotations[0].get('description', '')
        print(f"\nFull text in zoomed region:")
        print(f"  '{full_text.replace(chr(10), ' | ')}'")

        # Calculate center of zoomed image
        zoomed_height, zoomed_width = zoomed.shape[:2]
        center_x = zoomed_width / 2
        center_y = zoomed_height / 2

        print(f"\nIndividual text items near center:")
        items_near_center = []

        for ann in annotations[1:]:  # Skip full text
            text = ann['description']
            vertices = ann.get('boundingPoly', {}).get('vertices', [])

            if vertices:
                x_coords = [v.get('x', 0) for v in vertices]
                y_coords = [v.get('y', 0) for v in vertices]
                item_x = sum(x_coords) / len(x_coords)
                item_y = sum(y_coords) / len(y_coords)

                # Check if near center
                dist_from_center = ((item_x - center_x)**2 + (item_y - center_y)**2)**0.5

                if dist_from_center < 100:  # Within 100 pixels of center
                    items_near_center.append({
                        'text': text,
                        'x': item_x,
                        'y': item_y,
                        'dist': dist_from_center
                    })
                    print(f"  '{text}' at ({item_x:.0f}, {item_y:.0f}) - dist from center: {dist_from_center:.1f}")

        # Check for specific patterns
        print(f"\nPattern matching:")
        if '20' in full_text:
            print(f"  [OK] Found '20' in full text")

            # Check how it's detected
            for item in items_near_center:
                if '20' in item['text']:
                    print(f"    - '20' appears in item: '{item['text']}'")

        if '1/16' in full_text:
            print(f"  [OK] Found '1/16' in full text")
            for item in items_near_center:
                if '1/16' in item['text'] or '16' in item['text']:
                    print(f"    - '1/16' or '16' appears in item: '{item['text']}'")

        # Check if it's detected as one unit or separate
        for item in items_near_center:
            if item['text'] == '20 1/16':
                print(f"  [SINGLE UNIT] Detected as complete measurement: '20 1/16'")
                return '20 1/16'
            elif item['text'] == '20':
                # Look for nearby fraction
                for other in items_near_center:
                    if '1/16' in other['text'] and abs(other['x'] - item['x']) < 150:
                        print(f"  [SEPARATE] Found as two parts: '20' and '{other['text']}'")
                        return f"20 {other['text']}"

    return None

def main():
    print("="*80)
    print("PAGE 3 '20 1/16' DIMENSION DEBUG")
    print("="*80)

    # Get API key
    API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')
    if not API_KEY:
        print("[ERROR] Please set GOOGLE_VISION_API_KEY")
        return

    # Path to page 3 image
    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3.png"

    # Load full image for initial OCR
    print("\n[STEP 1] Running initial OCR on full image...")
    with open(image_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode('utf-8')

    annotations = get_ocr_from_region(content, API_KEY)

    if annotations:
        full_text = annotations[0].get('description', '')

        # Count occurrences of "20" in full text
        count_20 = len(re.findall(r'\b20\b', full_text))
        count_20_1_16 = len(re.findall(r'20\s+1/16', full_text))

        print(f"\nFull text analysis:")
        print(f"  - Occurrences of '20': {count_20}")
        print(f"  - Occurrences of '20 1/16': {count_20_1_16}")

        # Find all individual "20" items
        twenty_positions = []
        for ann in annotations[1:]:
            if ann['description'] == '20':
                vertices = ann.get('boundingPoly', {}).get('vertices', [])
                if vertices:
                    x = sum(v.get('x', 0) for v in vertices) / 4
                    y = sum(v.get('y', 0) for v in vertices) / 4
                    twenty_positions.append((x, y))
                    print(f"  - Found '20' at position ({x:.0f}, {y:.0f})")

        # Find all items containing "1/16"
        fraction_positions = []
        for ann in annotations[1:]:
            if '1/16' in ann['description'] or ann['description'] == '1' or ann['description'] == '16':
                vertices = ann.get('boundingPoly', {}).get('vertices', [])
                if vertices:
                    x = sum(v.get('x', 0) for v in vertices) / 4
                    y = sum(v.get('y', 0) for v in vertices) / 4
                    fraction_positions.append({
                        'text': ann['description'],
                        'x': x,
                        'y': y
                    })
                    print(f"  - Found fraction part '{ann['description']}' at ({x:.0f}, {y:.0f})")

    # Known position of the detected "20 1/16" from the JSON data
    detected_20_1_16 = (502, 1309)

    print(f"\n[STEP 2] Checking known detected position...")
    result1 = debug_ocr_at_position(image_path, detected_20_1_16[0], detected_20_1_16[1],
                                    API_KEY, "First 20 1/16 (detected)")

    # Now check other potential positions where "20" appears
    # Based on typical cabinet layouts, the second one might be:
    # - To the right of the first one (for double doors)
    # - Below the first one (for stacked cabinets)

    print(f"\n[STEP 3] Searching for potential second '20 1/16' location...")

    # Check to the right (around x=1100-1300 range based on cabinet layouts)
    potential_x = 1186  # This is where we see another "20" in the HEIGHT position
    potential_y = 1409
    result2 = debug_ocr_at_position(image_path, potential_x, potential_y,
                                    API_KEY, "Potential second 20 1/16")

    # Also check if there's text around the middle-right area
    print(f"\n[STEP 4] Scanning right side of image for any '20' measurements...")

    # Load image to check dimensions
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Scan the right half of the image
    scan_regions = [
        (w * 0.6, h * 0.4, "Upper Right"),
        (w * 0.6, h * 0.5, "Middle Right"),
        (w * 0.6, h * 0.6, "Lower Middle Right"),
        (w * 0.7, h * 0.5, "Far Right Middle"),
    ]

    for x, y, label in scan_regions:
        print(f"\n[Scanning {label} region...]")
        result = debug_ocr_at_position(image_path, x, y, API_KEY, label)
        if result and '20' in result:
            print(f"  [FOUND] Potential '20' measurement in {label}: {result}")

    print("\n" + "="*80)
    print("SUMMARY:")
    print("-"*80)
    print("The debug shows whether '20 1/16' appears once or twice in the image")
    print("and how OCR detects it (as single unit vs separate parts)")
    print("Check the saved debug images to visually verify the text locations.")

if __name__ == "__main__":
    main()