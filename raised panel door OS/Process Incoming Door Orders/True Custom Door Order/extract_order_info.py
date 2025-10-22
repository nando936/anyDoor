"""
Extract all information from TruCustom Cabinets door order form
Uses Google Vision API to extract:
- Company name
- Form fields and values
- Doors table with rows
- Drawer Fronts table with rows
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
    """Extract all text and structure from order form"""
    # Convert Windows path to forward slashes for OpenCV compatibility
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    print("=" * 60)
    print(f"Processing: {os.path.basename(image_path)}")
    print("=" * 60)

    # Read image and encode for API
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

    # Full text is in first annotation
    full_text = annotations[0].get('description', '')

    # Get individual text elements with positions
    text_elements = []
    for text in annotations[1:]:  # Skip first (full text)
        vertices = text['boundingPoly']['vertices']
        text_elements.append({
            'text': text['description'],
            'bounds': [(v.get('x', 0), v.get('y', 0)) for v in vertices],
            'center': (
                sum(v.get('x', 0) for v in vertices) / 4,
                sum(v.get('y', 0) for v in vertices) / 4
            )
        })

    print(f"[OK] Detected {len(text_elements)} text elements")

    # Parse the structure
    result = parse_order_form(full_text, text_elements)

    # Save to JSON
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{base_name}_order_info.json")

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[OK] Saved to: {json_path}")
    print("=" * 60)

    return result


def parse_order_form(full_text, text_elements):
    """Parse the order form structure"""
    lines = full_text.split('\n')

    result = {
        'company_name': None,
        'form_type': None,
        'fields': {},
        'doors_table': {
            'headers': [],
            'rows': []
        },
        'drawer_fronts_table': {
            'headers': [],
            'rows': []
        }
    }

    # Extract company name and form type
    if 'TRUCUSTOM' in full_text or 'CABINETS' in full_text:
        result['company_name'] = 'TRUCUSTOM CABINETS'
    if 'DOOR ORDER' in full_text:
        result['form_type'] = 'DOOR ORDER'

    # Extract form fields
    field_mapping = {
        'SUBMITTED BY:': 'submitted_by',
        'DATE:': 'date',
        'JOBSITE': 'jobsite',
        'WOOD TYPE': 'wood_type',
        'DOOR STYLE': 'door_style',
        'EDGE PROFILE': 'edge_profile',
        'PANEL CUT': 'panel_cut',
        'STICKING CUT': 'sticking_cut'
    }

    for i, line in enumerate(lines):
        for field_label, field_key in field_mapping.items():
            if field_label in line:
                # Get value from same line or next line
                value = line.replace(field_label, '').strip()
                if not value and i + 1 < len(lines):
                    value = lines[i + 1].strip()
                if value:
                    result['fields'][field_key] = value

    # Parse tables - find "DOORS" and "DRAWER FRONTS" sections
    doors_start = None
    drawer_fronts_start = None

    for i, line in enumerate(lines):
        if line.strip() == 'DOORS':
            doors_start = i
        elif 'DRAWER FRONTS' in line or 'DRAWER' in line and 'FRONT' in lines[i+1] if i+1 < len(lines) else False:
            drawer_fronts_start = i

    # Extract door table headers and rows
    if doors_start is not None:
        # Headers typically follow "DOORS"
        header_idx = doors_start + 1
        while header_idx < len(lines):
            if 'QTY' in lines[header_idx] and 'WIDTH' in lines[header_idx]:
                result['doors_table']['headers'] = ['QTY', 'WIDTH', 'HEIGHT', 'NOTE']
                break
            header_idx += 1

        # Extract rows until we hit drawer fronts section
        row_start = header_idx + 1
        end_idx = drawer_fronts_start if drawer_fronts_start else len(lines)

        current_row = {}
        for i in range(row_start, end_idx):
            line = lines[i].strip()
            if not line or line in ['DOORS', 'DRAWER FRONTS', 'DRAWER', 'FRONTS']:
                continue

            # Try to parse as table row (QTY, WIDTH, HEIGHT, NOTE)
            parts = line.split()
            if len(parts) >= 3:
                # Check if first part looks like quantity
                if parts[0].isdigit():
                    row = {
                        'qty': parts[0],
                        'width': parts[1] if len(parts) > 1 else '',
                        'height': parts[2] if len(parts) > 2 else '',
                        'note': ' '.join(parts[3:]) if len(parts) > 3 else ''
                    }
                    result['doors_table']['rows'].append(row)

    # Extract drawer fronts table
    if drawer_fronts_start is not None:
        # Headers after "DRAWER FRONTS"
        header_idx = drawer_fronts_start + 1
        while header_idx < len(lines):
            if 'QTY' in lines[header_idx] and 'WIDTH' in lines[header_idx]:
                result['drawer_fronts_table']['headers'] = ['QTY', 'WIDTH', 'HEIGHT', 'NOTE']
                break
            header_idx += 1

        # Extract rows
        row_start = header_idx + 1
        for i in range(row_start, len(lines)):
            line = lines[i].strip()
            if not line or 'SIGNATURE' in line or 'DATE' in line:
                continue

            # Parse drawer front rows
            parts = line.split()
            if len(parts) >= 2:
                # Check for "DRAWER" entries
                if 'DRAWER' in line:
                    # These appear to be placeholders
                    continue

                # Normal entries
                if parts[0].isdigit():
                    row = {
                        'qty': parts[0],
                        'width': parts[1] if len(parts) > 1 else '',
                        'height': parts[2] if len(parts) > 2 else '',
                        'note': ' '.join(parts[3:]) if len(parts) > 3 else ''
                    }
                    result['drawer_fronts_table']['rows'].append(row)

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify image path")
        print("Usage: python extract_order_info.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    extract_order_form(image_path)
