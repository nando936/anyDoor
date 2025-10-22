"""
Create marked up image showing row numbers on handwritten order
Uses Google Vision API to find the actual position of each measurement line
Handles multiple tables per room with corrections
"""
import os
import sys
import json
import cv2
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_VISION_API_KEY')
if not api_key:
    print("[ERROR] GOOGLE_VISION_API_KEY not found in .env file")
    sys.exit(1)

def encode_image(image_path):
    """Encode image to base64"""
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')

def call_vision_api(image_path):
    """Call Google Vision API for text detection"""
    image_data = encode_image(image_path)

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
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

def create_markup(image_path, extraction_json_path):
    """Create markup image with row numbers"""

    # Convert Windows UNC path if needed
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return False

    # Read the extraction JSON
    with open(extraction_json_path, 'r') as f:
        data = json.load(f)

    # Get OCR annotations to find positions
    print("[INFO] Getting text positions from Google Vision API...")
    text_annotations = call_vision_api(image_path)

    if not text_annotations:
        print("[ERROR] Could not get text annotations")
        return False

    # Create a copy for markup
    markup = image.copy()
    height, width = image.shape[:2]

    # Identify the main qty column position
    # Look for the first valid measurement line to establish the qty column x position
    qty_column_x = None

    # Find measurement lines - looking for "qty-" pattern (e.g., "2-171", "4-1813")
    measurement_candidates = []

    for i, annotation in enumerate(text_annotations[1:], 1):
        text = annotation.get('description', '').strip()

        # Skip header/room names
        if text.lower() in ['pantry', 'doors', 'drawers', 'living', 'room', 'door', 'white', 'oak', 'dour', 'ply', 'paint', 'dravers']:
            continue

        # Look for pattern: digit(s) followed by dash followed by digits
        if '-' in text and text and text[0].isdigit():
            parts = text.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1] and parts[1][0].isdigit():
                vertices = annotation['boundingPoly']['vertices']
                x_min = min(v.get('x', 0) for v in vertices)
                x_max = max(v.get('x', 0) for v in vertices)
                x_center = (x_min + x_max) / 2
                y = sum(v.get('y', 0) for v in vertices) / len(vertices)

                measurement_candidates.append({
                    'x_min': int(x_min),
                    'x_max': int(x_max),
                    'x_center': int(x_center),
                    'y': int(y),
                    'text': text
                })

    if not measurement_candidates:
        print("[ERROR] No measurement lines found")
        return False

    # Determine the main qty column position (most common x position)
    # Group by x_center with tolerance
    x_positions = [m['x_center'] for m in measurement_candidates]
    x_positions.sort()

    # Find clusters of x positions
    clusters = []
    current_cluster = [x_positions[0]]

    for x in x_positions[1:]:
        if x - current_cluster[-1] < 100:  # Within 100 pixels
            current_cluster.append(x)
        else:
            clusters.append(current_cluster)
            current_cluster = [x]
    clusters.append(current_cluster)

    # Use the largest cluster as the main qty column
    largest_cluster = max(clusters, key=len)
    qty_column_x = sum(largest_cluster) / len(largest_cluster)

    print(f"[INFO] Identified qty column at x={int(qty_column_x)}")

    # Filter to only measurements in the main column (within 80 pixels)
    measurement_lines = [
        m for m in measurement_candidates
        if abs(m['x_center'] - qty_column_x) < 80
    ]

    # Sort by Y position (top to bottom)
    measurement_lines.sort(key=lambda p: p['y'])

    print(f"[INFO] Found {len(measurement_lines)} measurement lines in main column")

    # Match with extraction entries
    all_entries = data.get('all_entries', [])

    # Check for corrections on the left side
    # Look for text elements to the left of the qty column that might be corrections
    left_margin = qty_column_x - 150  # 150 pixels to the left of qty column

    corrections_y_positions = []
    for annotation in text_annotations[1:]:
        text = annotation.get('description', '').strip()
        vertices = annotation['boundingPoly']['vertices']
        x = sum(v.get('x', 0) for v in vertices) / len(vertices)
        y = sum(v.get('y', 0) for v in vertices) / len(vertices)

        # Look for measurement-like text on the left side (corrections)
        if x < qty_column_x - 50:  # To the left of qty column
            # Check if it looks like a correction (numbers, fractions, measurements)
            if any(c.isdigit() for c in text) or '/' in text:
                corrections_y_positions.append(int(y))

    # Draw markers on each measurement line
    for idx, position in enumerate(measurement_lines):
        if idx < len(all_entries):
            entry = all_entries[idx]
            marker = entry.get('marker', '')
            marker_num = marker.replace('#', '')

            y_pos = position['y']

            # Check if there's a correction near this line (within 25 pixels vertically)
            has_left_correction = any(abs(y_pos - cy) < 25 for cy in corrections_y_positions)

            if has_left_correction:
                # Place marker on the right side (after height/desc column)
                # Find the rightmost text element on this line
                rightmost_x = position['x_max']
                for annotation in text_annotations[1:]:
                    ann_vertices = annotation['boundingPoly']['vertices']
                    ann_y = sum(v.get('y', 0) for v in ann_vertices) / len(ann_vertices)
                    if abs(ann_y - y_pos) < 15:  # Same line
                        ann_x_max = max(v.get('x', 0) for v in ann_vertices)
                        if ann_x_max > rightmost_x:
                            rightmost_x = ann_x_max

                x_pos = rightmost_x + 30  # To the right of rightmost text
            else:
                # Place marker on the left (default position)
                x_pos = position['x_min'] - 45  # To the left of qty

            # Draw circle background
            cv2.circle(markup, (x_pos, y_pos), 20, (0, 255, 0), -1)

            # Draw marker text
            text_x = x_pos - 12 if int(marker_num) < 10 else x_pos - 18
            cv2.putText(markup, marker_num, (text_x, y_pos + 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Save marked up image
    image_dir = os.path.dirname(image_path)
    image_name = Path(image_path).stem
    markup_path = os.path.join(image_dir, f"{image_name}_markup.jpg")

    cv2.imwrite(markup_path, markup)
    print(f"[OK] Saved markup image: {markup_path}")

    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_markup.py <image_path>")
        print("Example: python create_markup.py '//vmware-host/Shared Folders/D/OneDrive/customers/raised panel/handwritten orders/page1.jpg'")
        sys.exit(1)

    image_path = sys.argv[1]

    # Convert path if needed
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    # Find corresponding extraction JSON
    image_dir = os.path.dirname(image_path)
    image_name = Path(image_path).stem
    extraction_json = os.path.join(image_dir, f"{image_name}_raw_extraction.json")

    if not os.path.exists(extraction_json):
        print(f"[ERROR] Extraction JSON not found: {extraction_json}")
        print("Please run extract_handwritten.py first")
        sys.exit(1)

    print(f"[INFO] Creating markup for: {image_path}")

    if create_markup(image_path, extraction_json):
        print("[OK] Markup complete")
    else:
        print("[ERROR] Markup failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
