"""
Extract TruCustom door order form using position-based parsing
Uses Google Vision API to get text with coordinates, then groups by position
"""
import os
import sys
import json
import base64
import requests
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_VISION_API_KEY')
if not api_key:
    print("[ERROR] GOOGLE_VISION_API_KEY not found in .env file")
    sys.exit(1)

def extract_order_form(image_path):
    """Extract all text with positions from order form"""
    # Convert Windows path to forward slashes for OpenCV
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    print("=" * 60)
    print(f"Processing: {os.path.basename(image_path)}")
    print("=" * 60)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None

    _, buffer = cv2.imencode('.png', image)
    content = base64.b64encode(buffer).decode('utf-8')

    # Call Vision API
    print("\n=== Performing OCR with Google Vision API ===")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, json=request)

    if response.status_code != 200:
        print(f"[ERROR] Vision API failed: {response.status_code}")
        print(response.text)
        return None

    result = response.json()
    if 'responses' not in result or not result['responses']:
        print("[WARNING] No response from Vision API")
        return None

    annotations = result['responses'][0].get('textAnnotations', [])
    if not annotations:
        print("[WARNING] No text detected in image")
        return None

    # Get full text
    full_text = annotations[0].get('description', '')

    # Get individual text elements with positions
    text_elements = []
    for text in annotations[1:]:  # Skip first (full text)
        vertices = text['boundingPoly']['vertices']
        x_coords = [v.get('x', 0) for v in vertices]
        y_coords = [v.get('y', 0) for v in vertices]

        text_elements.append({
            'text': text['description'],
            'x': sum(x_coords) / len(x_coords),
            'y': sum(y_coords) / len(y_coords),
            'left': min(x_coords),
            'top': min(y_coords),
            'right': max(x_coords),
            'bottom': max(y_coords)
        })

    print(f"[OK] Detected {len(text_elements)} text elements")

    # Parse using positions
    result = parse_order_form_positions(full_text, text_elements, image.shape[0])

    # Save to JSON
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{base_name}_extracted.json")

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[OK] Saved to: {json_path}")

    # Print summary
    print("\n=== EXTRACTED DATA ===")
    print(f"Company: {result['company_name']}")
    print(f"Form Type: {result['form_type']}")
    print(f"\nFields:")
    for key, value in result['fields'].items():
        print(f"  {key}: {value}")
    print(f"\nDoors: {len(result['doors_table']['rows'])} rows")
    print(f"Drawer Fronts: {len(result['drawer_fronts_table']['rows'])} rows")
    print("=" * 60)

    return result


def parse_order_form_positions(full_text, text_elements, image_height):
    """Parse form using position-based grouping - DOORS and DRAWER FRONTS are side-by-side"""

    result = {
        'company_name': 'TRUCUSTOM CABINETS',
        'form_type': 'DOOR ORDER',
        'fields': {},
        'doors_table': {
            'headers': ['QTY', 'WIDTH', 'HEIGHT', 'NOTE'],
            'rows': []
        },
        'drawer_fronts_table': {
            'headers': ['QTY', 'WIDTH', 'HEIGHT', 'NOTE'],
            'rows': []
        }
    }

    # Extract form fields from full text
    lines = full_text.split('\n')

    # Simple field extraction
    field_map = {
        'submitted_by': None,
        'date': None,
        'jobsite': None,
        'wood_type': None,
        'door_style': None,
        'edge_profile': None,
        'panel_cut': None,
        'sticking_cut': None
    }

    for i, line in enumerate(lines):
        if 'SUBMITTED BY:' in line:
            # Next line usually has the value
            if i + 1 < len(lines):
                field_map['submitted_by'] = lines[i + 1].strip()
        elif 'DATE:' in line:
            parts = line.split('DATE:')
            if len(parts) > 1:
                field_map['date'] = parts[1].strip()
        elif 'JOBSITE' in line and i + 1 < len(lines):
            field_map['jobsite'] = lines[i + 1].strip()
        elif 'WOOD TYPE' in line and i + 1 < len(lines):
            field_map['wood_type'] = lines[i + 1].strip()
        elif 'DOOR STYLE' in line and i + 1 < len(lines):
            field_map['door_style'] = lines[i + 1].strip()
        elif 'EDGE PROFILE' in line and i + 1 < len(lines):
            field_map['edge_profile'] = lines[i + 1].strip()
        elif 'PANEL CUT' in line and i + 1 < len(lines):
            field_map['panel_cut'] = lines[i + 1].strip()
        elif 'STICKING CUT' in line and i + 1 < len(lines):
            field_map['sticking_cut'] = lines[i + 1].strip()

    result['fields'] = {k: v for k, v in field_map.items() if v}

    # Find table headers to determine column split
    doors_header = None
    drawer_header = None

    for elem in text_elements:
        if elem['text'] == 'DOORS':
            doors_header = elem
        elif elem['text'] == 'DRAWER':
            # Check if FRONTS is nearby
            for e2 in text_elements:
                if e2['text'] == 'FRONTS' and abs(e2['y'] - elem['y']) < 20 and e2['x'] > elem['x']:
                    drawer_header = elem
                    break

    # Find middle x-coordinate to split DOORS and DRAWER FRONTS columns
    # Use the position between DOORS and DRAWER FRONTS headers
    split_x = None
    if doors_header and drawer_header:
        split_x = (doors_header['x'] + drawer_header['x']) / 2

    # Get table data region (below headers)
    table_start_y = doors_header['y'] + 30 if doors_header else 400

    # Get all table elements
    table_elements = [e for e in text_elements if e['y'] > table_start_y]

    # Group by rows
    rows = group_by_rows(table_elements, y_threshold=10)

    for row in rows:
        # Skip if contains signature
        if any('SIGNATURE' in e['text'] or 'HEARBY' in e['text'] for e in row):
            break

        # Sort by x position
        row.sort(key=lambda e: e['x'])

        # Split row into left (DOORS) and right (DRAWER FRONTS) based on x position
        if split_x:
            left_elements = [e for e in row if e['x'] < split_x]
            right_elements = [e for e in row if e['x'] >= split_x]

            # Process DOORS (left side)
            if left_elements and len(left_elements) >= 3:
                if left_elements[0]['text'].isdigit():
                    door_row = {
                        'qty': left_elements[0]['text'],
                        'width': left_elements[1]['text'] if len(left_elements) > 1 else '',
                        'height': left_elements[2]['text'] if len(left_elements) > 2 else '',
                        'note': ' '.join(e['text'] for e in left_elements[3:]) if len(left_elements) > 3 else ''
                    }
                    result['doors_table']['rows'].append(door_row)

            # Process DRAWER FRONTS (right side)
            if right_elements and len(right_elements) >= 3:
                # Skip if just "DRAWER" placeholders
                row_text = ' '.join(e['text'] for e in right_elements)
                if 'DRAWER DRAWER DRAWER' not in row_text and right_elements[0]['text'].isdigit():
                    drawer_row = {
                        'qty': right_elements[0]['text'],
                        'width': right_elements[1]['text'] if len(right_elements) > 1 else '',
                        'height': right_elements[2]['text'] if len(right_elements) > 2 else '',
                        'note': ' '.join(e['text'] for e in right_elements[3:]) if len(right_elements) > 3 else ''
                    }
                    result['drawer_fronts_table']['rows'].append(drawer_row)

    return result


def group_by_rows(elements, y_threshold=10):
    """Group text elements by row based on y position"""
    if not elements:
        return []

    # Sort by y position
    sorted_elements = sorted(elements, key=lambda e: e['y'])

    rows = []
    current_row = [sorted_elements[0]]

    for elem in sorted_elements[1:]:
        # If y position is close to current row, add to it
        if abs(elem['y'] - current_row[0]['y']) < y_threshold:
            current_row.append(elem)
        else:
            # Start new row
            rows.append(current_row)
            current_row = [elem]

    # Add last row
    if current_row:
        rows.append(current_row)

    return rows


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify image path")
        print("Usage: python extract_order_position_based.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    extract_order_form(image_path)
