"""
Extract handwritten door order using Claude CLI to identify tables
Then crop and OCR each table separately for better accuracy
"""
import os
import sys
import json
import base64
import requests
import cv2
import subprocess
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
google_api_key = os.getenv('GOOGLE_VISION_API_KEY')

if not google_api_key:
    print("[ERROR] GOOGLE_VISION_API_KEY not found in .env file")
    sys.exit(1)

def encode_image(image_path):
    """Encode image to base64"""
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')

def analyze_with_claude(image_path):
    """Use Claude CLI (Max plan) to analyze the image and identify table boundaries"""

    print("[INFO] Analyzing image with Claude CLI to identify tables...")

    # Get image directory for --add-dir permission
    image_dir = os.path.dirname(image_path)

    prompt = f"""Analyze this handwritten door order image and identify all the tables: {image_path}

I need you to identify:
1. The image dimensions (approximate width and height in pixels)
2. Each table's approximate position as percentages of the image (top%, bottom%, left%, right%)
3. The room name for each table
4. Whether the table is for doors or drawers

Please provide the information in JSON format:

{{
  "image_width": estimated_width,
  "image_height": estimated_height,
  "tables": [
    {{
      "room": "room name",
      "type": "doors" or "drawers",
      "top_percent": percentage from top (0-100),
      "bottom_percent": percentage from top (0-100),
      "left_percent": percentage from left (0-100),
      "right_percent": percentage from right (0-100)
    }}
  ]
}}

The image shows multiple tables - there are "Living Room Doors", "Pantry Doors", "Pantry Drawers", "White Oak Doors", and "White Oak Drawers" tables.
Each table has columns: QTY - WIDTH x HEIGHT and sometimes a DESC/NOTES column.

IMPORTANT: Respond with ONLY the JSON, no other text."""

    # Call Claude via CLI
    claude_exe = r'C:\Users\nando\AppData\Roaming\npm\claude.cmd'

    try:
        print(f"Calling Claude CLI to analyze tables...")
        result = subprocess.run(
            f'{claude_exe} --print --add-dir "{image_dir}"',
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
            shell=True
        )

        if result.returncode != 0:
            print(f"[ERROR] Claude call failed with code {result.returncode}")
            if result.stderr:
                print(f"  STDERR: {result.stderr}")
            if result.stdout:
                print(f"  STDOUT: {result.stdout}")
            return None

        # Parse Claude's response
        response_text = result.stdout.strip()
        print(f"\n[DEBUG] Claude response:\n{response_text}\n")

        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)

        return None

    except subprocess.TimeoutExpired:
        print("[ERROR] Claude call timed out")
        return None
    except Exception as e:
        print(f"[ERROR] Claude call failed: {e}")
        return None

def crop_table(image, table_info, image_width, image_height):
    """Crop a table region from the image"""

    top = int(image_height * table_info['top_percent'] / 100)
    bottom = int(image_height * table_info['bottom_percent'] / 100)
    left = int(image_width * table_info['left_percent'] / 100)
    right = int(image_width * table_info['right_percent'] / 100)

    # Add some padding
    padding = 10
    top = max(0, top - padding)
    bottom = min(image_height, bottom + padding)
    left = max(0, left - padding)
    right = min(image_width, right + padding)

    cropped = image[top:bottom, left:right]
    return cropped

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

def parse_table_text(text, room, table_type):
    """Parse OCR text from a single table"""

    lines = text.split('\n')
    entries = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip headers
        if any(header in line.lower() for header in ['qty', 'width', 'height', 'door', 'drawer', room.lower()]):
            continue

        # Look for measurement pattern: qty - width x height
        if '-' in line and 'x' in line.lower():
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

                        # Clean up
                        width = width.split()[0] if width.split() else width
                        height_parts = height.split()
                        height = height_parts[0] if height_parts else height

                        entry = {
                            'room': room,
                            'type': table_type,
                            'qty': qty,
                            'width': width,
                            'height': height
                        }
                        entries.append(entry)

    return entries

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_with_table_detection.py <image_path>")
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

    # Analyze with Claude to find tables
    table_analysis = analyze_with_claude(image_path)

    if not table_analysis:
        print("[ERROR] Could not analyze tables with Claude")
        sys.exit(1)

    tables = table_analysis.get('tables', [])
    print(f"[INFO] Found {len(tables)} tables")

    # Get output directory
    image_dir = os.path.dirname(image_path)
    image_name = Path(image_path).stem

    # Process each table
    all_entries = []
    row_number = 1

    for idx, table_info in enumerate(tables):
        room = table_info.get('room', f'Table{idx+1}')
        table_type = table_info.get('type', 'door')

        print(f"\n[INFO] Processing {room} {table_type}...")

        # Crop table
        cropped = crop_table(image, table_info, image_width, image_height)

        # Save cropped image for debugging
        crop_path = os.path.join(image_dir, f"{image_name}_crop_{room.replace(' ', '_')}.jpg")
        cv2.imwrite(crop_path, cropped)
        print(f"[OK] Saved crop: {crop_path}")

        # OCR the cropped table
        ocr_text = ocr_with_google_vision(cropped)

        if ocr_text:
            # Save OCR text
            ocr_path = os.path.join(image_dir, f"{image_name}_ocr_{room.replace(' ', '_')}.txt")
            with open(ocr_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            print(f"[OK] Saved OCR: {ocr_path}")

            # Parse the table
            entries = parse_table_text(ocr_text, room, table_type)

            # Add markers
            for entry in entries:
                entry['marker'] = f'#{row_number}'
                all_entries.append(entry)
                row_number += 1

            print(f"[OK] Extracted {len(entries)} entries from {room}")

    # Save combined extraction
    extraction_data = {
        'door_profile': '235',
        'all_entries': all_entries
    }

    raw_output = os.path.join(image_dir, f"{image_name}_table_extraction.json")
    with open(raw_output, 'w') as f:
        json.dump(extraction_data, f, indent=2)
    print(f"\n[OK] Saved extraction: {raw_output}")

    print(f"\n[INFO] Total entries extracted: {len(all_entries)}")

if __name__ == '__main__':
    main()
