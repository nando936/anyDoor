"""
Extract handwritten door order information using Google Vision API
Outputs in unified door order format
"""
import os
import sys
import json
import base64
import requests
from pathlib import Path
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

def encode_image(image_path):
    """Encode image to base64"""
    # Convert Windows UNC path if needed
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')

def call_vision_api(image_path):
    """Call Google Vision API for text detection"""

    # Encode image
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
        print(response.text)
        return None

    result = response.json()

    if 'responses' in result and len(result['responses']) > 0:
        text_annotations = result['responses'][0].get('textAnnotations', [])
        if text_annotations:
            # First annotation contains full text
            full_text = text_annotations[0].get('description', '')
            return full_text

    return None

def parse_handwritten_text(text):
    """Parse the OCR text to extract door order information"""

    lines = text.split('\n')

    extracted = {
        'door_profile': '',
        'wood_type': '',
        'jobsite': '',
        'notes': '',
        'all_entries': []  # All doors and drawers in order
    }

    # Track current section
    current_section = None
    row_number = 1

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for door profile
        if 'door profile' in line.lower():
            # Extract number
            parts = line.split()
            for part in parts:
                if part.isdigit():
                    extracted['door_profile'] = part
                    break

        # Look for section headers
        if 'pantry doors' in line.lower():
            current_section = 'pantry_doors'
            continue
        elif 'pantry drawers' in line.lower():
            current_section = 'pantry_drawers'
            continue
        elif 'living room doors' in line.lower() or 'living room door' in line.lower():
            current_section = 'living_room_doors'
            continue
        elif 'white oak' in line.lower() and 'door' in line.lower():
            current_section = 'white_oak_doors'
            continue
        elif 'white oak' in line.lower() or ('drawer' in line.lower() and current_section == 'white_oak_doors'):
            current_section = 'white_oak_drawers'
            continue
        elif 'drawer' in line.lower() and 'ply' in line.lower():
            current_section = 'white_oak_drawers'
            continue

        # Try to parse door/drawer entries
        if current_section and '-' in line:
            # Look for pattern: qty - width x height
            parts = line.split('-')
            if len(parts) >= 2:
                qty_str = parts[0].strip()
                measurement_str = parts[1].strip()

                # Extract quantity
                qty = 1
                for char in qty_str:
                    if char.isdigit():
                        qty = int(char)
                        break

                # Extract width x height
                if 'x' in measurement_str.lower():
                    meas_parts = measurement_str.lower().split('x')
                    if len(meas_parts) >= 2:
                        width = meas_parts[0].strip()
                        height = meas_parts[1].strip()

                        # Clean up measurements - remove notes
                        width = width.split()[0] if width.split() else width
                        height_parts = height.split()
                        height = height_parts[0] if height_parts else height

                        # Look for notes
                        notes = ''
                        if 'mdf' in measurement_str.lower():
                            notes = 'MDF'
                        elif 'hdf' in measurement_str.lower():
                            notes = 'HDF'

                        # Determine type (door or drawer)
                        item_type = 'drawer' if 'drawer' in current_section else 'door'

                        # Determine location
                        location = ''
                        if 'pantry' in current_section:
                            location = 'Pantry'
                        elif 'living_room' in current_section:
                            location = 'Living Room'
                        elif 'white_oak' in current_section:
                            location = 'White Oak'

                        item = {
                            'marker': f'#{row_number}',
                            'type': item_type,
                            'location': location,
                            'qty': qty,
                            'width': width,
                            'height': height
                        }
                        if notes:
                            item['notes'] = notes

                        extracted['all_entries'].append(item)
                        row_number += 1

    return extracted

def convert_to_unified_format(extracted_data, image_filename):
    """Convert extracted data to unified door order format"""

    # Collect all doors and drawers from all_entries
    doors = []
    drawers = []

    for entry in extracted_data.get('all_entries', []):
        qty = entry.get('qty', 1)
        width = entry.get('width', '')
        height = entry.get('height', '')
        notes = entry.get('notes', '')
        marker = entry.get('marker', '#1')
        location = entry.get('location', '')
        item_type = entry.get('type', 'door')

        # Calculate decimals
        width_decimal = fraction_to_decimal(width)
        height_decimal = fraction_to_decimal(height)

        # Build item
        item_data = {
            'marker': marker,
            'qty': qty,
            'width': width,
            'height': height,
            'width_decimal': width_decimal,
            'height_decimal': height_decimal,
            'location': location
        }

        # Add sqft for doors
        if item_type == 'door':
            item_data['sqft'] = calculate_sqft(width_decimal, height_decimal)
            doors.append(item_data)
        else:
            drawers.append(item_data)

    # Build unified format
    unified = {
        'schema_version': '1.0',
        'source': {
            'type': 'other',
            'original_file': image_filename,
            'extraction_date': get_current_date(),
            'extractor_version': 'handwritten_v1.0'
        },
        'order_info': {
            'jobsite': extracted_data.get('jobsite', 'Unknown'),
            'room': 'Multiple'
        },
        'specifications': {
            'wood_type': extracted_data.get('wood_type', 'Unknown'),
            'door_style': extracted_data.get('door_profile', 'Unknown'),
            'edge_profile': 'Unknown',
            'panel_cut': extracted_data.get('notes', '')
        },
        'size_info': {
            'all_sizes_are_finished': True,
            'conversion_notes': 'Handwritten order - sizes as written'
        },
        'doors': doors,
        'drawers': drawers,
        'summary': calculate_summary(doors, drawers)
    }

    return unified

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_handwritten.py <image_path>")
        print("Example: python extract_handwritten.py '//vmware-host/Shared Folders/D/OneDrive/customers/raised panel/handwritten orders/8814453144994501118.jpg'")
        sys.exit(1)

    image_path = sys.argv[1]

    # Convert path if needed
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    print(f"[INFO] Processing handwritten order: {image_path}")

    # Extract with Google Vision
    print("[INFO] Extracting text with Google Vision API...")
    ocr_text = call_vision_api(image_path)

    if not ocr_text:
        print("[ERROR] OCR extraction failed")
        sys.exit(1)

    # Get output directory (same as input image)
    image_dir = os.path.dirname(image_path)
    image_name = Path(image_path).stem

    # Save raw OCR text
    ocr_output = os.path.join(image_dir, f"{image_name}_ocr.txt")
    with open(ocr_output, 'w', encoding='utf-8') as f:
        f.write(ocr_text)
    print(f"[OK] Saved OCR text: {ocr_output}")

    # Parse the OCR text
    print("[INFO] Parsing extracted text...")
    extracted = parse_handwritten_text(ocr_text)

    # Save raw extraction
    raw_output = os.path.join(image_dir, f"{image_name}_raw_extraction.json")
    with open(raw_output, 'w') as f:
        json.dump(extracted, f, indent=2)
    print(f"[OK] Saved raw extraction: {raw_output}")

    # Convert to unified format
    print("[INFO] Converting to unified format...")
    unified = convert_to_unified_format(extracted, Path(image_path).name)

    # Save unified format
    unified_output = os.path.join(image_dir, f"{image_name}_unified.json")
    if save_unified_json(unified, unified_output):
        print(f"[OK] Saved unified format: {unified_output}")
        print(f"\n[INFO] Summary:")
        print(f"  Total doors: {unified['summary']['total_door_units']} units ({unified['summary']['total_doors']} items)")
        print(f"  Total drawers: {unified['summary']['total_drawer_units']} units ({unified['summary']['total_drawers']} items)")
        print(f"  Total units: {unified['summary']['total_units']}")
    else:
        print("[ERROR] Failed to save unified format")
        sys.exit(1)

if __name__ == '__main__':
    main()
