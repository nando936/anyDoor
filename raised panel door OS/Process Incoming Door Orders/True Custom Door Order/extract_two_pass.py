"""
Two-pass OCR extraction:
1. First pass: Find table boundaries
2. Second pass: Crop and OCR each table separately for better accuracy
"""
import os
import sys
import json
import base64
import requests
import cv2
import numpy as np
from dotenv import load_dotenv

# Add parent directory to path for shared utilities
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from shared_utils import (
    fraction_to_decimal,
    calculate_sqft,
    calculate_summary,
    save_unified_json,
    get_current_date
)

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_VISION_API_KEY')
if not api_key:
    print("[ERROR] GOOGLE_VISION_API_KEY not found in .env file")
    sys.exit(1)

def preprocess_image(image):
    """Preprocess image for better OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better text contrast
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

    # Convert back to BGR for Vision API
    preprocessed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    return preprocessed

def call_vision_api(image, preprocess=False):
    """Call Vision API on an image (numpy array)"""
    if preprocess:
        image = preprocess_image(image)

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
    if response.status_code != 200:
        print(f"[ERROR] Vision API failed: {response.status_code}")
        return None

    result = response.json()
    return result['responses'][0].get('textAnnotations', [])

def find_table_boundaries(text_elements):
    """Find approximate boundaries for DOORS and DRAWER FRONTS tables"""
    doors_header = None
    drawer_header = None

    for elem in text_elements:
        vertices = elem['boundingPoly']['vertices']
        x = sum(v.get('x', 0) for v in vertices) / len(vertices)
        y = sum(v.get('y', 0) for v in vertices) / len(vertices)

        if elem['description'] == 'DOORS':
            doors_header = {'x': x, 'y': y}
        elif elem['description'] == 'DRAWER':
            drawer_header = {'x': x, 'y': y}

    return doors_header, drawer_header

def extract_two_pass(image_path):
    """Two-pass extraction with cropping"""
    # Convert Windows path
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

    img_height, img_width = image.shape[:2]

    # PASS 1: Find table locations
    print("\n=== PASS 1: Finding table boundaries ===")
    annotations = call_vision_api(image)
    if not annotations:
        return None

    doors_header, drawer_header = find_table_boundaries(annotations[1:])

    if not doors_header or not drawer_header:
        print("[ERROR] Could not find table headers")
        return None

    print(f"DOORS table at x={doors_header['x']:.0f}, y={doors_header['y']:.0f}")
    print(f"DRAWER FRONTS table at x={drawer_header['x']:.0f}, y={drawer_header['y']:.0f}")

    # Calculate split position (middle between headers)
    split_x = int((doors_header['x'] + drawer_header['x']) / 2)
    table_top = int(doors_header['y'] + 20)  # Start below headers
    table_bottom = img_height - 100  # Leave room for signature

    # PASS 2a: Crop and OCR DOORS table with 4X ZOOM
    print("\n=== PASS 2a: OCR DOORS table (cropped + zoomed 4x) ===")
    doors_crop = image[table_top:table_bottom, 0:split_x]

    # Save debug crop BEFORE zoom
    output_dir = os.path.dirname(image_path)
    cv2.imwrite(os.path.join(output_dir, "DEBUG_doors_crop_original.png"), doors_crop)

    # Scale up 4x for better OCR
    doors_crop_height, doors_crop_width = doors_crop.shape[:2]
    doors_crop = cv2.resize(doors_crop, (doors_crop_width * 4, doors_crop_height * 4), interpolation=cv2.INTER_CUBIC)

    # Save debug crop AFTER zoom
    cv2.imwrite(os.path.join(output_dir, "DEBUG_doors_crop_4x.png"), doors_crop)

    # Preprocess and save debug
    doors_preprocessed = preprocess_image(doors_crop)
    cv2.imwrite(os.path.join(output_dir, "DEBUG_doors_preprocessed.png"), doors_preprocessed)

    doors_annotations = call_vision_api(doors_crop, preprocess=False)
    # Pass offset info: table starts at table_top, left edge at 0, zoom factor 4
    doors_rows = parse_table_crop(
        doors_annotations[1:] if doors_annotations else [],
        start_number=1,
        offset_x=0,
        offset_y=table_top,
        zoom_factor=4
    )
    print(f"Found {len(doors_rows)} door rows")

    # PASS 2b: Crop and OCR DRAWER FRONTS table with 4X ZOOM
    print("\n=== PASS 2b: OCR DRAWER FRONTS table (cropped + zoomed 4x) ===")
    drawer_crop = image[table_top:table_bottom, split_x:img_width]

    # Save debug crop BEFORE zoom
    cv2.imwrite(os.path.join(output_dir, "DEBUG_drawer_crop_original.png"), drawer_crop)

    # Scale up 4x for better OCR
    drawer_crop_height, drawer_crop_width = drawer_crop.shape[:2]
    drawer_crop = cv2.resize(drawer_crop, (drawer_crop_width * 4, drawer_crop_height * 4), interpolation=cv2.INTER_CUBIC)

    # Save debug crop AFTER zoom
    cv2.imwrite(os.path.join(output_dir, "DEBUG_drawer_crop_4x.png"), drawer_crop)

    # Preprocess and save debug
    drawer_preprocessed = preprocess_image(drawer_crop)
    cv2.imwrite(os.path.join(output_dir, "DEBUG_drawer_preprocessed.png"), drawer_preprocessed)

    drawer_annotations = call_vision_api(drawer_crop, preprocess=False)
    # Continue numbering from where doors left off
    drawer_start_number = len(doors_rows) + 1
    # Pass offset info: table starts at table_top, left edge at split_x, zoom factor 4
    drawer_rows = parse_table_crop(
        drawer_annotations[1:] if drawer_annotations else [],
        start_number=drawer_start_number,
        offset_x=split_x,
        offset_y=table_top,
        zoom_factor=4
    )
    print(f"Found {len(drawer_rows)} drawer front rows")

    # Get form fields from full image
    full_text = annotations[0].get('description', '')
    fields = extract_fields(full_text)

    # Extract CHECK 1 fields
    check1_fields = extract_check1_fields(full_text)

    result = {
        'company_name': 'TRUCUSTOM CABINETS',
        'form_type': 'DOOR ORDER',
        'fields': fields,
        'check1': check1_fields,
        'doors_table': {
            'headers': ['QTY', 'WIDTH', 'HEIGHT', 'NOTE'],
            'rows': doors_rows
        },
        'drawer_fronts_table': {
            'headers': ['QTY', 'WIDTH', 'HEIGHT', 'NOTE'],
            'rows': drawer_rows
        }
    }

    # Save output
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{base_name}_two_pass.json")

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[OK] Saved to: {json_path}")

    # Also save in unified format
    unified_data = convert_to_unified_format(result, image_path)
    unified_json_path = os.path.join(output_dir, f"{base_name}_unified_door_order.json")
    save_unified_json(unified_data, unified_json_path)

    print("=" * 60)

    return result

def merge_measurements(row):
    """Merge elements that belong together (e.g., '23' and '15/16' become '23 15/16')"""
    merged = []
    i = 0

    while i < len(row):
        current = row[i]['text']

        # Check if next element is a fraction (e.g., '15/16', '1/2', '3/4')
        if i + 1 < len(row):
            next_elem = row[i + 1]['text']
            # If next element is a fraction AND close in x-position (within 200 pixels)
            if '/' in next_elem and (row[i + 1]['x'] - row[i]['x']) < 200:
                # Merge them
                merged.append(f"{current} {next_elem}")
                i += 2  # Skip next element since we merged it
                continue

        # No merge, just add current
        merged.append(current)
        i += 1

    return merged

def parse_table_crop(text_elements, start_number=1, offset_x=0, offset_y=0, zoom_factor=1):
    """Parse table from cropped image and save position data

    Args:
        text_elements: OCR text elements
        start_number: Starting number for row markers (default 1)
        offset_x: X offset to add to positions (crop's left edge in full image)
        offset_y: Y offset to add to positions (crop's top edge in full image)
        zoom_factor: Zoom factor applied to crop (divide positions by this)
    """
    if not text_elements:
        return []

    # Convert to simpler format
    elements = []
    for elem in text_elements:
        vertices = elem['boundingPoly']['vertices']
        x = sum(v.get('x', 0) for v in vertices) / len(vertices)
        y = sum(v.get('y', 0) for v in vertices) / len(vertices)
        elements.append({
            'text': elem['description'],
            'x': x,
            'y': y
        })

    # Group by rows
    rows = group_by_rows(elements, y_threshold=15)

    table_rows = []
    row_number = start_number
    for row in rows:
        # Skip header row
        if any('QTY' in e['text'] or 'WIDTH' in e['text'] for e in row):
            continue

        # Skip DRAWER placeholders
        row_text = ' '.join(e['text'] for e in row)
        if 'DRAWER DRAWER' in row_text or row_text.strip() == 'DRAWER':
            continue

        # Sort by x position
        row.sort(key=lambda e: e['x'])

        # Must have at least qty, width, height
        if len(row) >= 3 and row[0]['text'].isdigit():
            # Merge measurements with fractions based on x-position clustering
            merged = merge_measurements(row)

            if len(merged) >= 3:
                # Calculate position in FULL IMAGE coordinates
                # 1. Divide by zoom factor to get original crop coordinates
                # 2. Add offset to get full image coordinates
                row_x = (row[0]['x'] / zoom_factor) + offset_x
                row_y = (row[0]['y'] / zoom_factor) + offset_y

                table_row = {
                    'marker': f'#{row_number}',
                    'position': {
                        'x': int(row_x),
                        'y': int(row_y)
                    },
                    'qty': merged[0],
                    'width': merged[1],
                    'height': merged[2],
                    'note': ' '.join(merged[3:]) if len(merged) > 3 else ''
                }
                table_rows.append(table_row)
                row_number += 1

    return table_rows

def group_by_rows(elements, y_threshold=15):
    """Group text elements by row based on y position"""
    if not elements:
        return []

    sorted_elements = sorted(elements, key=lambda e: e['y'])
    rows = []
    current_row = [sorted_elements[0]]

    for elem in sorted_elements[1:]:
        if abs(elem['y'] - current_row[0]['y']) < y_threshold:
            current_row.append(elem)
        else:
            rows.append(current_row)
            current_row = [elem]

    if current_row:
        rows.append(current_row)

    return rows

def extract_fields(full_text):
    """Extract form fields from full text"""
    lines = full_text.split('\n')
    fields = {}

    for i, line in enumerate(lines):
        # SUBMITTED BY and JOBSITE appear as separate header lines
        # Pattern: "SUBMITTED BY:" -> "JOBSITE" -> "Keith" -> "Morningstar - Leatherman"
        if 'SUBMITTED BY:' in line:
            # Look ahead to confirm pattern
            # Pattern: "SUBMITTED BY:" -> "JOBSITE" -> "Keith" -> "DATE: ..." -> "Morningstar - Leatherman"
            if i + 1 < len(lines) and 'JOBSITE' in lines[i + 1]:
                # Value for submitted_by is 2 lines down (i+2)
                if i + 2 < len(lines):
                    val = lines[i + 2].strip()
                    # Skip if it looks like a date or another field label
                    if val and not val.startswith('DATE:') and not val.isupper():
                        fields['submitted_by'] = val
                # Value for jobsite: scan forward to find it (skip DATE line)
                for j in range(i + 3, min(i + 6, len(lines))):
                    val = lines[j].strip()
                    # Skip DATE and CHECK lines, take first valid value
                    if val and not val.startswith('DATE:') and not val.startswith('CHECK') and not val.isupper():
                        fields['jobsite'] = val
                        break
        elif 'DATE:' in line:
            parts = line.split('DATE:')
            if len(parts) > 1:
                fields['date'] = parts[1].strip()
        elif 'WOOD TYPE' in line and i + 1 < len(lines):
            fields['wood_type'] = lines[i + 1].strip()
        elif 'DOOR STYLE' in line and i + 1 < len(lines):
            fields['door_style'] = lines[i + 1].strip()
        elif 'EDGE PROFILE' in line and i + 1 < len(lines):
            fields['edge_profile'] = lines[i + 1].strip()
        elif 'PANEL CUT' in line and i + 1 < len(lines):
            fields['panel_cut'] = lines[i + 1].strip()
        elif 'STICKING CUT' in line and i + 1 < len(lines):
            fields['sticking_cut'] = lines[i + 1].strip()

    return fields

def extract_check1_fields(full_text):
    """Extract CHECK 1 section fields with checkbox states and values

    Since OCR doesn't reliably detect checkbox symbols, we use a simpler approach:
    - FINISHED DOOR SIZE: Default to CHECKED (most common case)
    - OPENING SIZE: Extract value after "ADD:" if present
    - HINGE TYPE: Extract CUT and SUPPLY values if present
    - PLATE SIZE: Extract value if present

    User can manually verify and correct if needed.
    """
    lines = full_text.split('\n')
    check1 = {}

    # FINISHED DOOR SIZE - typically checked by default
    # Look for the line in text
    finished_found = any('FINISHED DOOR SIZE' in line for line in lines)
    check1['finished_door_size'] = {
        'checked': True,  # Assuming checked based on the image we saw
        'note': '(ADD NOTHING)'
    }

    # OPENING SIZE - extract ADD value
    opening_add = 'EMPTY'
    for line in lines:
        if 'OPENING SIZE' in line and 'ADD:' in line:
            # Try to extract value between "ADD:" and "TO EACH SIDE"
            parts = line.split('ADD:')
            if len(parts) > 1:
                value_part = parts[1].split('TO EACH SIDE')[0].strip()
                if value_part:
                    opening_add = value_part

    check1['opening_size'] = {
        'checked': False,  # Default unchecked unless value found
        'add_value': opening_add
    }

    # HINGE TYPE - extract CUT and SUPPLY
    hinge_cut = 'EMPTY'
    hinge_supply = 'EMPTY'
    for i, line in enumerate(lines):
        if 'CUT:' in line:
            parts = line.split('CUT:')
            if len(parts) > 1:
                value = parts[1].split('SUPPLY:')[0].strip() if 'SUPPLY:' in parts[1] else parts[1].strip()
                if value:  # Only set if there's an actual value
                    hinge_cut = value
        if 'SUPPLY:' in line:
            parts = line.split('SUPPLY:')
            if len(parts) > 1:
                value = parts[1].strip()
                if value:  # Only set if there's an actual value
                    hinge_supply = value

    check1['hinge_type'] = {
        'checked': False,  # Default unchecked
        'cut': hinge_cut,
        'supply': hinge_supply
    }

    # PLATE SIZE - extract value
    plate_value = 'EMPTY'
    for line in lines:
        if 'PLATE SIZE:' in line:
            parts = line.split('PLATE SIZE:')
            if len(parts) > 1:
                value = parts[1].strip()
                if value:  # Only set if there's an actual value
                    plate_value = value

    check1['plate_size'] = {
        'checked': False,  # Default unchecked
        'value': plate_value
    }

    return check1

def convert_to_unified_format(true_custom_data, original_file_path):
    """
    Convert True Custom format to Unified Door Order format

    Args:
        true_custom_data: Dictionary in True Custom format
        original_file_path: Path to original image/PDF file

    Returns:
        dict: Data in unified format
    """
    fields = true_custom_data.get('fields', {})
    check1 = true_custom_data.get('check1', {})

    # Determine if sizes are finished or opening
    all_sizes_are_finished = check1.get('finished_door_size', {}).get('checked', True)

    # Convert door rows to unified format
    doors = []
    for row in true_custom_data.get('doors_table', {}).get('rows', []):
        width_str = row.get('width', '')
        height_str = row.get('height', '')
        width_dec = fraction_to_decimal(width_str)
        height_dec = fraction_to_decimal(height_str)

        doors.append({
            "marker": row.get('marker', ''),
            "qty": int(row.get('qty', 1)),
            "width": width_str,
            "height": height_str,
            "width_decimal": width_dec,
            "height_decimal": height_dec,
            "sqft": calculate_sqft(width_dec, height_dec),
            "location": row.get('note', '')
        })

    # Convert drawer rows to unified format
    drawers = []
    for row in true_custom_data.get('drawer_fronts_table', {}).get('rows', []):
        width_str = row.get('width', '')
        height_str = row.get('height', '')
        width_dec = fraction_to_decimal(width_str)
        height_dec = fraction_to_decimal(height_str)

        drawers.append({
            "marker": row.get('marker', ''),
            "qty": int(row.get('qty', 1)),
            "width": width_str,
            "height": height_str,
            "width_decimal": width_dec,
            "height_decimal": height_dec,
            "location": row.get('note', '')
        })

    # Build unified format
    unified_data = {
        "schema_version": "1.0",
        "source": {
            "type": "true_custom",
            "original_file": os.path.basename(original_file_path),
            "extraction_date": get_current_date(),
            "extractor_version": "1.0"
        },
        "order_info": {
            "customer_company": true_custom_data.get('company_name', ''),
            "jobsite": fields.get('jobsite', ''),
            "room": "",
            "submitted_by": fields.get('submitted_by', ''),
            "order_date": fields.get('date', get_current_date())
        },
        "specifications": {
            "wood_type": fields.get('wood_type', ''),
            "door_style": fields.get('door_style', ''),
            "edge_profile": fields.get('edge_profile', ''),
            "panel_cut": fields.get('panel_cut', ''),
            "sticking_cut": fields.get('sticking_cut', '')
        },
        "size_info": {
            "all_sizes_are_finished": all_sizes_are_finished,
            "conversion_notes": "Sizes as provided in True Custom order form"
        },
        "doors": doors,
        "drawers": drawers,
        "summary": calculate_summary(doors, drawers)
    }

    return unified_data

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify image path")
        print("Usage: python extract_two_pass.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    extract_two_pass(image_path)
