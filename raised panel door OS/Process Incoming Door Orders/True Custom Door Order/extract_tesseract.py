"""
Extract TruCustom door order form using Tesseract OCR
"""
import os
import sys
import json
import cv2
import pytesseract
from dotenv import load_dotenv

def extract_with_tesseract(image_path):
    """Two-pass extraction using Tesseract"""
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

    # PASS 1: Get full text to find table boundaries
    print("\n=== PASS 1: Finding table boundaries ===")
    full_text = pytesseract.image_to_string(image)

    # Simple boundary detection - look for DOORS and DRAWER FRONTS in text
    # For this form, we know approximate positions
    # DOORS table is on left half, DRAWER FRONTS on right half
    split_x = img_width // 2
    table_top = 400  # Below headers
    table_bottom = img_height - 200  # Above signature

    print(f"Table region: top={table_top}, bottom={table_bottom}, split={split_x}")

    # PASS 2a: OCR DOORS table (left half)
    print("\n=== PASS 2a: OCR DOORS table (left side) ===")
    doors_crop = image[table_top:table_bottom, 0:split_x]

    # Save debug crop
    output_dir = os.path.dirname(image_path)
    cv2.imwrite(os.path.join(output_dir, "DEBUG_doors_tesseract.png"), doors_crop)

    # Use Tesseract with table detection
    doors_data = pytesseract.image_to_data(doors_crop, output_type=pytesseract.Output.DICT)
    doors_rows = parse_table_from_tesseract(doors_data)
    print(f"Found {len(doors_rows)} door rows")

    # PASS 2b: OCR DRAWER FRONTS table (right half)
    print("\n=== PASS 2b: OCR DRAWER FRONTS table (right side) ===")
    drawer_crop = image[table_top:table_bottom, split_x:img_width]

    # Save debug crop
    cv2.imwrite(os.path.join(output_dir, "DEBUG_drawer_tesseract.png"), drawer_crop)

    drawer_data = pytesseract.image_to_data(drawer_crop, output_type=pytesseract.Output.DICT)
    drawer_rows = parse_table_from_tesseract(drawer_data)
    print(f"Found {len(drawer_rows)} drawer front rows")

    # Extract form fields from full text
    fields = extract_fields(full_text)

    result = {
        'company_name': 'TRUCUSTOM CABINETS',
        'form_type': 'DOOR ORDER',
        'fields': fields,
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
    json_path = os.path.join(output_dir, f"{base_name}_tesseract.json")

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[OK] Saved to: {json_path}")
    print("=" * 60)

    return result

def parse_table_from_tesseract(data):
    """Parse table from Tesseract data output"""
    n_boxes = len(data['text'])

    # Group text elements by row (similar y positions)
    elements = []
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if not text or data['conf'][i] < 0:  # Skip empty or low confidence
            continue

        elements.append({
            'text': text,
            'x': data['left'][i] + data['width'][i] / 2,
            'y': data['top'][i] + data['height'][i] / 2,
            'conf': data['conf'][i]
        })

    # Group by rows
    rows = group_by_rows(elements, y_threshold=15)

    table_rows = []
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
            table_row = {
                'qty': row[0]['text'],
                'width': row[1]['text'] if len(row) > 1 else '',
                'height': row[2]['text'] if len(row) > 2 else '',
                'note': ' '.join(e['text'] for e in row[3:]) if len(row) > 3 else ''
            }
            table_rows.append(table_row)

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
        if 'SUBMITTED BY:' in line and i + 1 < len(lines):
            fields['submitted_by'] = lines[i + 1].strip()
        elif 'DATE:' in line:
            parts = line.split('DATE:')
            if len(parts) > 1:
                fields['date'] = parts[1].strip()
        elif 'JOBSITE' in line and i + 1 < len(lines):
            fields['jobsite'] = lines[i + 1].strip()
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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify image path")
        print("Usage: python extract_tesseract.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    extract_with_tesseract(image_path)
