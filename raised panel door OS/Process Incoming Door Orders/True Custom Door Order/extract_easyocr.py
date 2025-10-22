"""
Extract TruCustom door order form using EasyOCR
"""
import os
import sys
import json
import cv2
import easyocr

def extract_with_easyocr(image_path):
    """Two-pass extraction using EasyOCR"""
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

    # Initialize EasyOCR reader
    print("\n=== Initializing EasyOCR ===")
    reader = easyocr.Reader(['en'], gpu=False)

    # PASS 1: Find table boundaries
    print("\n=== PASS 1: Finding table boundaries ===")
    results = reader.readtext(image)

    # Find DOORS and DRAWER FRONTS headers
    doors_x = None
    drawer_x = None
    table_top = 400

    for (bbox, text, conf) in results:
        if 'DOORS' in text.upper():
            doors_x = bbox[0][0]  # Top-left x coordinate
        elif 'DRAWER' in text.upper() and 'FRONT' in text.upper():
            drawer_x = bbox[0][0]

    # Calculate split position
    if doors_x and drawer_x:
        split_x = int((doors_x + drawer_x) / 2)
    else:
        split_x = img_width // 2

    table_bottom = img_height - 200

    print(f"Table region: top={table_top}, bottom={table_bottom}, split={split_x}")

    # PASS 2a: OCR DOORS table (left half)
    print("\n=== PASS 2a: OCR DOORS table (left side) ===")
    doors_crop = image[table_top:table_bottom, 0:split_x]

    # Save debug crop
    output_dir = os.path.dirname(image_path)
    cv2.imwrite(os.path.join(output_dir, "DEBUG_doors_easyocr.png"), doors_crop)

    doors_results = reader.readtext(doors_crop)
    doors_rows = parse_table_from_easyocr(doors_results)
    print(f"Found {len(doors_rows)} door rows")

    # PASS 2b: OCR DRAWER FRONTS table (right half)
    print("\n=== PASS 2b: OCR DRAWER FRONTS table (right side) ===")
    drawer_crop = image[table_top:table_bottom, split_x:img_width]

    # Save debug crop
    cv2.imwrite(os.path.join(output_dir, "DEBUG_drawer_easyocr.png"), drawer_crop)

    drawer_results = reader.readtext(drawer_crop)
    drawer_rows = parse_table_from_easyocr(drawer_results)
    print(f"Found {len(drawer_rows)} drawer front rows")

    # Extract form fields
    full_text = '\n'.join([text for (bbox, text, conf) in results])
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
    json_path = os.path.join(output_dir, f"{base_name}_easyocr.json")

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[OK] Saved to: {json_path}")
    print("=" * 60)

    return result

def parse_table_from_easyocr(results):
    """Parse table from EasyOCR results"""
    # Convert to simpler format
    elements = []
    for (bbox, text, conf) in results:
        # Get center position
        x = (bbox[0][0] + bbox[2][0]) / 2
        y = (bbox[0][1] + bbox[2][1]) / 2

        elements.append({
            'text': text.strip(),
            'x': x,
            'y': y,
            'conf': conf
        })

    # Group by rows
    rows = group_by_rows(elements, y_threshold=15)

    table_rows = []
    for row in rows:
        # Skip header row
        if any('QTY' in e['text'].upper() or 'WIDTH' in e['text'].upper() for e in row):
            continue

        # Skip DRAWER placeholders
        row_text = ' '.join(e['text'] for e in row)
        if 'DRAWER DRAWER' in row_text.upper() or row_text.strip().upper() == 'DRAWER':
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
        if 'SUBMITTED BY:' in line.upper() and i + 1 < len(lines):
            fields['submitted_by'] = lines[i + 1].strip()
        elif 'DATE:' in line.upper():
            parts = line.split(':')
            if len(parts) > 1:
                fields['date'] = parts[1].strip()
        elif 'JOBSITE' in line.upper() and i + 1 < len(lines):
            fields['jobsite'] = lines[i + 1].strip()
        elif 'WOOD TYPE' in line.upper() and i + 1 < len(lines):
            fields['wood_type'] = lines[i + 1].strip()
        elif 'DOOR STYLE' in line.upper() and i + 1 < len(lines):
            fields['door_style'] = lines[i + 1].strip()
        elif 'EDGE PROFILE' in line.upper() and i + 1 < len(lines):
            fields['edge_profile'] = lines[i + 1].strip()
        elif 'PANEL CUT' in line.upper() and i + 1 < len(lines):
            fields['panel_cut'] = lines[i + 1].strip()
        elif 'STICKING CUT' in line.upper() and i + 1 < len(lines):
            fields['sticking_cut'] = lines[i + 1].strip()

    return fields

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify image path")
        print("Usage: python extract_easyocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    extract_with_easyocr(image_path)
