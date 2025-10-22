"""
Extract handwritten door order by detecting table regions using Google Vision API
Then crop and OCR each table separately for better accuracy
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

def call_vision_api(image_path):
    """Call Google Vision API for text detection with positions"""

    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

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
        return result['responses'][0].get('textAnnotations', [])
    return None

def find_table_headers(text_annotations):
    """Find positions of table headers (room names)"""

    headers = []

    for annotation in text_annotations[1:]:  # Skip first (full text)
        text = annotation.get('description', '').lower()
        vertices = annotation['boundingPoly']['vertices']

        # Calculate bounding box
        x_coords = [v.get('x', 0) for v in vertices]
        y_coords = [v.get('y', 0) for v in vertices]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Look for table headers
        if 'pantry' in text and ('door' in text or 'drawer' in text):
            table_type = 'doors' if 'door' in text else 'drawers'
            headers.append({
                'room': 'Pantry',
                'type': table_type,
                'text': annotation['description'],
                'y_min': y_min,
                'y_max': y_max,
                'x_min': x_min,
                'x_max': x_max
            })
        elif 'living' in text or 'room' in text:
            # Check if next annotations say "Room Door"
            headers.append({
                'room': 'Living Room',
                'type': 'doors',
                'text': annotation['description'],
                'y_min': y_min,
                'y_max': y_max,
                'x_min': x_min,
                'x_max': x_max
            })
        elif 'white' in text or 'oak' in text:
            headers.append({
                'room': 'White Oak',
                'type': 'doors',  # Will determine type by position
                'text': annotation['description'],
                'y_min': y_min,
                'y_max': y_max,
                'x_min': x_min,
                'x_max': x_max
            })

    return headers

def find_table_boundaries(text_annotations, image_width, image_height):
    """Find table boundaries by detecting headers and measurement lines"""

    print("[INFO] Detecting table boundaries from OCR...")

    # Find all text elements with positions
    all_elements = []
    for annotation in text_annotations[1:]:
        text = annotation.get('description', '').strip()
        vertices = annotation['boundingPoly']['vertices']

        x_coords = [v.get('x', 0) for v in vertices]
        y_coords = [v.get('y', 0) for v in vertices]

        all_elements.append({
            'text': text,
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords),
            'x_center': sum(x_coords) / len(x_coords),
            'y_center': sum(y_coords) / len(y_coords)
        })

    # Group by detecting section headers
    tables = []

    # Find headers by looking for specific keywords and grouping nearby elements
    # Headers appear as sequences: "Living", "Room", "Door" or "Pantry", "Doors", etc.
    headers_found = []
    used_indices = set()

    for i, elem in enumerate(all_elements):
        if i in used_indices:
            continue

        text = elem['text']
        text_lower = text.lower()

        # Living Room Door (3 elements around y=304-312)
        if text_lower == 'living':
            # Look for "Room" and "Door" nearby
            nearby = [e for e in all_elements if abs(e['y_center'] - elem['y_center']) < 20]
            if any('room' in e['text'].lower() for e in nearby):
                headers_found.append({
                    'room': 'Living Room',
                    'type': 'doors',
                    'y': elem['y_max'],
                    'x': elem['x_min']
                })

        # Pantry Doors (2 elements around y=453-456)
        elif text_lower == 'pantry':
            # Look for "Doors" nearby
            nearby = [e for e in all_elements if abs(e['y_center'] - elem['y_center']) < 20]
            if any('door' in e['text'].lower() for e in nearby):
                headers_found.append({
                    'room': 'Pantry',
                    'type': 'doors',
                    'y': elem['y_max'],
                    'x': elem['x_min']
                })
            # Or "Dravers" (drawers)
            elif any('drawer' in e['text'].lower() or 'draver' in e['text'].lower() for e in nearby):
                headers_found.append({
                    'room': 'Pantry',
                    'type': 'drawers',
                    'y': elem['y_max'],
                    'x': elem['x_min']
                })

        # White oak (around y=1231)
        elif text_lower == 'white':
            # Look for "oak" nearby
            nearby = [e for e in all_elements if abs(e['y_center'] - elem['y_center']) < 20]
            if any('oak' in e['text'].lower() for e in nearby):
                # White oak - check x position to determine if doors or drawers
                # Left side = doors, right side = drawers
                if elem['x_center'] < image_width * 0.3:
                    headers_found.append({
                        'room': 'White Oak',
                        'type': 'doors',
                        'y': elem['y_max'],
                        'x': elem['x_min']
                    })

        # Drawers 3/8 ply (right column under white oak)
        elif text_lower == 'drawers':
            # This is the right column header for white oak drawers
            if elem['x_center'] > image_width * 0.5:  # Right side
                headers_found.append({
                    'room': 'White Oak',
                    'type': 'drawers',
                    'y': elem['y_max'],
                    'x': elem['x_min']
                })

    # Sort headers by y position
    headers_found.sort(key=lambda h: h['y'])

    print(f"[DEBUG] Found {len(headers_found)} headers:")
    for h in headers_found:
        print(f"  - {h['room']} {h['type']} at y={h['y']}")

    # Create table regions based on headers
    for i, header in enumerate(headers_found):
        y_start = header['y'] + 10  # Start below header

        # Find y_end (next header or end of image)
        if i + 1 < len(headers_found):
            # Next header exists - end just before it
            y_end = headers_found[i + 1]['y'] - 10
        else:
            # Last header - go to bottom
            y_end = image_height

        # Determine x boundaries based on position
        # Check if header is on left or right side
        if header['x'] < image_width / 2:
            # Left column
            x_min = 0
            x_max = int(image_width * 0.5)
        else:
            # Right column
            x_min = int(image_width * 0.5)
            x_max = image_width

        tables.append({
            'room': header['room'],
            'type': header['type'],
            'y_min': y_start,
            'y_max': y_end,
            'x_min': x_min,
            'x_max': x_max
        })

    return tables

def ocr_image(image_array):
    """OCR an image array using Google Vision API"""

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
        return None

    result = response.json()
    if 'responses' in result and len(result['responses']) > 0:
        text_annotations = result['responses'][0].get('textAnnotations', [])
        if text_annotations:
            return text_annotations[0].get('description', '')
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_with_auto_crops.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

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

    # Get OCR with positions
    print("[INFO] Running Google Vision OCR on full image...")
    text_annotations = call_vision_api(image_path)

    if not text_annotations:
        print("[ERROR] OCR failed")
        sys.exit(1)

    # Find table boundaries
    tables = find_table_boundaries(text_annotations, image_width, image_height)

    print(f"[INFO] Detected {len(tables)} tables:")
    for t in tables:
        print(f"  - {t['room']} {t['type']}: y={t['y_min']}-{t['y_max']}, x={t['x_min']}-{t['x_max']}")

    # Get output directory
    image_dir = os.path.dirname(image_path)
    image_name = Path(image_path).stem

    # Process each table
    for table in tables:
        room = table['room'].replace(' ', '_')
        table_type = table['type']

        print(f"\n[INFO] Cropping {room} {table_type}...")

        # Add padding
        padding = 20
        y_min = max(0, table['y_min'] - padding)
        y_max = min(image_height, table['y_max'] + padding)
        x_min = max(0, table['x_min'] - padding)
        x_max = min(image_width, table['x_max'] + padding)

        # Crop
        cropped = image[y_min:y_max, x_min:x_max]

        # Save cropped image
        crop_name = f"{room}_{table_type}"
        crop_path = os.path.join(image_dir, f"{image_name}_autocrop_{crop_name}.jpg")
        cv2.imwrite(crop_path, cropped)
        print(f"[OK] Saved crop: {crop_path}")

        # OCR the cropped table
        ocr_text = ocr_image(cropped)

        if ocr_text:
            ocr_path = os.path.join(image_dir, f"{image_name}_autocrop_ocr_{crop_name}.txt")
            with open(ocr_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            print(f"[OK] Saved OCR: {ocr_path}")

    print(f"\n[OK] All crops saved to {image_dir}")

if __name__ == '__main__':
    main()
