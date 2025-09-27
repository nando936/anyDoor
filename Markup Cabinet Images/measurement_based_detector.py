"""
Measurement-Based Dimension Detector
First finds measurements with Vision API, then looks for lines near those measurements
"""
import cv2
import numpy as np
import requests
import base64
import json
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import sys

sys.stdout.reconfigure(encoding='utf-8')

def fix_ocr_errors(text, is_full_text=False):
    """Apply generic OCR error fixes to any text

    Args:
        text: The text to fix
        is_full_text: If True, applies fixes for full text strings.
                      If False, applies fixes for individual text items.
    """
    if is_full_text:
        # Fixes for full text (multiple words/measurements in one string)
        # Remove negative signs before numbers anywhere in the text
        text = re.sub(r'-(\d+)', r'\1', text)

        # Fix decimal points that should be spaces in fractions
        text = re.sub(r'(\d+)\.(\d+/\d+)', r'\1 \2', text)  # "16.5/8" → "16 5/8"
        text = re.sub(r'(\d+)\.(\d+)\s+(\d+)', r'\1 \2/\3', text)  # "16.5 8" → "16 5/8"
    else:
        # Fixes for individual text items (single words/numbers)
        # Remove leading negative signs from numbers
        text = re.sub(r'^-(\d+)$', r'\1', text)  # "-32" → "32"

        # Fix decimal points that should be spaces in fractions
        text = re.sub(r'^(\d+)\.(\d+)$', r'\1 \2', text)  # "16.5" → "16 5"

    # Add more patterns here as we encounter them

    return text

def verify_measurement_with_zoom(image_path, x, y, text, api_key):
    """Verify a measurement by zooming in on its location"""
    import cv2

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return text  # Return original if can't load

    h, w = image.shape[:2]

    # Create a padded crop area - wider to ensure complete text capture
    # Increased from 100 to 200 to avoid cutting off measurements
    padding_x = 200  # Wider horizontal padding to capture complete measurements
    padding_y = 50   # Vertical padding
    x1 = max(0, int(x - padding_x))
    y1 = max(0, int(y - padding_y))
    x2 = min(w, int(x + padding_x))
    y2 = min(h, int(y + padding_y))

    # Crop the region
    cropped = image[y1:y2, x1:x2]

    # Zoom in 3x
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Encode the zoomed image for Vision API
    _, buffer = cv2.imencode('.png', zoomed)
    zoomed_content = base64.b64encode(buffer).decode('utf-8')

    # Run OCR on the zoomed region
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request = {
        "requests": [{
            "image": {"content": zoomed_content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, json=request)

    if response.status_code == 200:
        result = response.json()
        if 'responses' in result and result['responses']:
            annotations = result['responses'][0].get('textAnnotations', [])
            if annotations:
                # Get the full text from the zoomed region
                verified_text = annotations[0]['description'].strip()

                # Look for measurement patterns in the verified text
                import re
                # Check for measurements in the verified text
                lines = verified_text.split('\n')
                for line in lines:
                    line = line.strip()
                    # First check if line exactly matches measurement patterns
                    if re.match(r'^\d+$', line) or \
                       re.match(r'^\d+\s+\d+/\d+$', line) or \
                       re.match(r'^\d+-\d+/\d+$', line) or \
                       re.match(r'^\d+/\d+/\d+$', line):
                        print(f"    Verification: '{text}' -> '{line}'")
                        return line

                    # If line contains extra text, try to extract the measurement
                    # This handles cases like "20.34 20 3/16" where we want "20 3/16"
                    measurement_match = re.search(r'\b(\d+\s+\d+/\d+)\b', line)
                    if measurement_match:
                        extracted = measurement_match.group(1)
                        print(f"    Verification: '{text}' -> '{extracted}' (extracted from '{line}')")
                        return extracted

    return text  # Return original if verification doesn't find anything clear

def correct_image_skew(image_path):
    """Detect and correct skew in an image using line detection"""
    import cv2
    import numpy as np

    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        print(f"  [SKEW] No lines detected for skew correction")
        return image_path, 0

    # Calculate angles of horizontal/vertical lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize to small angles (close to 0 or 90 degrees)
        if -45 < angle < 45:
            angles.append(angle)  # Horizontal lines
        elif angle > 45:
            angles.append(angle - 90)  # Vertical lines
        elif angle < -45:
            angles.append(angle + 90)  # Vertical lines

    if not angles:
        print(f"  [SKEW] No suitable lines for skew detection")
        return image_path, 0

    # Filter to only small angles (likely skew, not diagonal lines)
    small_angles = [a for a in angles if abs(a) < 5]

    if small_angles:
        # Use mean of small angles for better skew detection
        mean_angle = np.mean(small_angles)
        print(f"  [SKEW] Detected angles: all={len(angles)}, small={len(small_angles)}, mean_small={mean_angle:.2f}")

        # Only correct if skew is consistent and significant
        # INCREASED THRESHOLD to avoid losing measurements
        if 2.0 < abs(mean_angle) < 5:
            skew_angle = mean_angle
        else:
            skew_angle = 0
    else:
        print(f"  [SKEW] No small angles detected for skew correction")
        skew_angle = 0

    # Only correct if skew is significant
    if abs(skew_angle) > 0.5:
        print(f"  [SKEW] Correcting skew of {skew_angle:.2f} degrees...")

        # Get image dimensions
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        # Rotate the image to correct skew
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        corrected = cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

        # Save the corrected image temporarily and also save for viewing
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "skew_corrected_temp.png")
        cv2.imwrite(temp_path, corrected)

        # Also save a copy in the same folder as the original for viewing
        base_name = os.path.basename(image_path).replace('.png', '_skew_corrected.png')
        dir_name = os.path.dirname(image_path)
        view_path = os.path.join(dir_name, base_name)
        cv2.imwrite(view_path, corrected)
        print(f"  [SKEW] Corrected image saved to: {view_path}")

        return temp_path, skew_angle
    else:
        print(f"  [SKEW] Minimal skew detected ({skew_angle:.2f} degrees), no correction needed")
        return image_path, 0

def get_text_item_info(all_text_items, text_key):
    """Safely get text item info, handling duplicates"""
    if text_key in all_text_items:
        info = all_text_items[text_key]
        if isinstance(info, list):
            return info[0]  # Return first occurrence
        return info
    return None

def get_measurements_from_vision_api(image_path, api_key, debug_text=None):
    """Get all measurements with positions from Vision API using two-pass approach"""
    import re

    # Skip skew correction - it can lose measurements
    # corrected_path, skew_angle = correct_image_skew(image_path)
    corrected_path = image_path  # Use original image directly

    # PASS 1: Initial OCR to get all text positions
    print("\n  PASS 1: Initial OCR scan...")
    with open(corrected_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode('utf-8')

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, json=request)

    measurements = []
    all_text_items = {}
    full_text = ""
    all_text_items_list = []

    if response.status_code == 200:
        result = response.json()
        if 'responses' in result and result['responses']:
            annotations = result['responses'][0].get('textAnnotations', [])

            if annotations:
                # First annotation is full text
                full_text = annotations[0]['description']

                # Apply OCR error fixes to the full text
                full_text = fix_ocr_errors(full_text, is_full_text=True)

                print(f"  Full text found: {full_text.replace(chr(10), ' ')}")

                # Find measurements in full text
                lines = full_text.split('\n')
                measurement_patterns = []
                for line in lines:
                    line = line.strip()
                    # Match whole numbers (like "14" or "18") - common for even measurements
                    if re.match(r'^\d+$', line):
                        # Check if it's a reasonable measurement (between 2 and 100)
                        num = int(line)
                        if 2 <= num <= 100:
                            measurement_patterns.append(line)
                    # Match measurements with optional F suffix (finish size)
                    # Match "31 1/4 F" or "31 1/4" formats
                    elif re.match(r'^\d+\s+\d+/\d+(\s+F)?$', line):
                        measurement_patterns.append(line)
                    # Match "31-1/4 F" or "31-1/4" formats (with dash)
                    elif re.match(r'^\d+-\d+/\d+(\s+F)?$', line):
                        # Convert "31-1/4 F" to "31 1/4 F" (replace dash with space)
                        converted = line.replace('-', ' ')
                        measurement_patterns.append(converted)
                        print(f"  Converted '{line}' to '{converted}'")
                    elif re.match(r'^\d+/\d+/\d+$', line):
                        # Convert "5/1/2" to "5 1/2"
                        parts = line.split('/')
                        if len(parts) == 3:
                            converted = f"{parts[0]} {parts[1]}/{parts[2]}"
                            measurement_patterns.append(converted)
                            print(f"  Converted '{line}' to '{converted}'")

                print(f"  Measurements in text: {measurement_patterns}")

                # Get all individual text items with positions
                for ann in annotations[1:]:
                    text = ann['description']
                    vertices = ann['boundingPoly']['vertices']

                    x = sum(v.get('x', 0) for v in vertices) / 4
                    y = sum(v.get('y', 0) for v in vertices) / 4

                    # Store original text for now (will verify with zoom)
                    all_text_items_list.append({
                        'text': text,
                        'x': x,
                        'y': y,
                        'vertices': vertices,
                        'original_text': text  # Keep original for comparison
                    })

                    # Also keep the dictionary for backward compatibility
                    # But if there are duplicates, keep all positions in a list
                    if text not in all_text_items:
                        all_text_items[text] = {'x': x, 'y': y, 'vertices': vertices}
                    else:
                        # Convert to list if duplicate found
                        if not isinstance(all_text_items[text], list):
                            all_text_items[text] = [all_text_items[text]]
                        all_text_items[text].append({'x': x, 'y': y, 'vertices': vertices})

                # Debug: Show all text items found with numbers
                print(f"\n  [DEBUG] All text items found (including duplicates):")
                for item in all_text_items_list:
                    if any(char.isdigit() for char in item['text']):
                        print(f"    '{item['text']}' at ({item['x']:.0f}, {item['y']:.0f})")

                # PASS 2: Group nearby text items and zoom verify groups
                print(f"\n  PASS 2: Grouping nearby text and zoom verification...")

                # Group nearby text items that might be parts of the same measurement
                groups = []
                used_indices = set()

                for i, item in enumerate(all_text_items_list):
                    if i in used_indices:
                        continue

                    # Only process items with numbers
                    if not any(char.isdigit() for char in item['text']):
                        continue

                    # Start a new group with this item
                    group = [item]
                    used_indices.add(i)

                    # Find nearby items (within 120 pixels horizontally, 30 pixels vertically)
                    for j, other in enumerate(all_text_items_list):
                        if j in used_indices:
                            continue

                        # Check proximity
                        x_dist = abs(other['x'] - item['x'])
                        y_dist = abs(other['y'] - item['y'])

                        if x_dist < 120 and y_dist < 30:  # Same line, close together (increased from 100 to 120 for "35 5/16")
                            group.append(other)
                            used_indices.add(j)

                    groups.append(group)

                print(f"  Formed {len(groups)} groups for verification")

                # Now zoom verify each group
                for group in groups:
                    if len(group) == 1:
                        # Single item - verify normally
                        item = group[0]
                        print(f"    Zoom verifying single: '{item['text']}' at ({item['x']:.0f}, {item['y']:.0f})...")
                        verified_text = verify_measurement_with_zoom(image_path, item['x'], item['y'], item['text'], api_key)

                        # Sanity check - cabinet dimensions shouldn't be > 100 inches
                        if verified_text:
                            import re
                            if re.match(r'^\d+', verified_text):
                                first_num = int(re.match(r'^(\d+)', verified_text).group(1))
                                if first_num > 100:
                                    print(f"      [REJECTED] '{verified_text}' - unrealistic dimension, keeping original")
                                    verified_text = item['text']

                        if verified_text != item['text']:
                            print(f"      [CORRECTED] '{item['text']}' -> '{verified_text}'")
                            item['text'] = verified_text
                            item['zoom_corrected'] = True
                        else:
                            # Check if it's a format like "5/1/2" that needs conversion to "5 1/2"
                            if re.match(r'^\d+/\d+/\d+$', item['text']):
                                parts = item['text'].split('/')
                                if len(parts) == 3:
                                    converted = f"{parts[0]} {parts[1]}/{parts[2]}"
                                    print(f"      [CONFIRMED] '{item['text']}' -> converted to '{converted}'")
                                    item['text'] = converted
                                    item['zoom_corrected'] = False
                            else:
                                print(f"      [CONFIRMED] '{item['text']}'")
                                item['zoom_corrected'] = False
                    else:
                        # Multiple items - zoom the center of the group
                        group_text = ' '.join([g['text'] for g in group])
                        avg_x = sum(g['x'] for g in group) / len(group)
                        avg_y = sum(g['y'] for g in group) / len(group)

                        print(f"    Zoom verifying group: '{group_text}' at ({avg_x:.0f}, {avg_y:.0f})...")
                        verified_text = verify_measurement_with_zoom(image_path, avg_x, avg_y, group_text, api_key)

                        # Sanity check - cabinet dimensions shouldn't be > 100 inches
                        if verified_text:
                            # Check if it's a measurement with unrealistic values
                            import re
                            if re.match(r'^\d+', verified_text):
                                first_num = int(re.match(r'^(\d+)', verified_text).group(1))
                                if first_num > 100:
                                    print(f"      [REJECTED] '{verified_text}' - unrealistic dimension, keeping original")
                                    verified_text = group_text

                        # If verification returns a single measurement, update the first item and clear the rest
                        if verified_text and verified_text != group_text:
                            print(f"      [GROUP CORRECTED] '{group_text}' -> '{verified_text}'")
                            # Keep the verified text in the first item of the group
                            group[0]['text'] = verified_text
                            group[0]['x'] = avg_x
                            group[0]['y'] = avg_y
                            group[0]['zoom_corrected'] = True

                            # Mark other items as merged (will be skipped later)
                            for g in group[1:]:
                                g['merged'] = True
                        else:
                            print(f"      [GROUP CONFIRMED] '{group_text}'")
                            # Even if not corrected, we need to update the first item with the grouped text
                            if len(group) > 1:
                                group[0]['text'] = group_text
                                group[0]['x'] = avg_x
                                group[0]['y'] = avg_y
                                # Mark other items as merged
                                for g in group[1:]:
                                    g['merged'] = True
                            for g in group:
                                g['zoom_corrected'] = False

                # PASS 3: Pattern matching with verified text
                print(f"\n  PASS 3: Identifying measurements from verified text...")

                # Update full text with verified measurements
                verified_full_text_parts = []
                for item in all_text_items_list:
                    verified_full_text_parts.append(item['text'])

                # Rebuild measurement patterns from verified text (skip merged items)
                measurement_patterns = []
                for item in all_text_items_list:
                    if item.get('merged', False):
                        continue  # Skip items that were merged into groups

                    text = item['text'].strip()
                    # Match whole numbers
                    if re.match(r'^\d+$', text):
                        num = int(text) if text.isdigit() else 0
                        if 2 <= num <= 100:
                            measurement_patterns.append(text)
                            print(f"    Found whole number: '{text}'")
                    # Match fractions with whole numbers (like "35 5/16")
                    elif re.match(r'^\d+\s+\d+/\d+$', text):
                        measurement_patterns.append(text)
                        print(f"    Found measurement with fraction: '{text}'")
                    # Match dash format with optional F suffix (like "21-1/4 F" or "31-1/4")
                    elif re.match(r'^\d+-\d+/\d+(\s+F)?$', text):
                        # Convert to space format and remove F suffix for measurement
                        converted = text.replace('-', ' ')
                        if ' F' in converted:
                            converted = converted.replace(' F', '')  # Remove F suffix
                        measurement_patterns.append(converted)
                        print(f"    Found measurement with dash format: '{text}' -> '{converted}'")
                    elif re.match(r'^\d+/\d+$', text):
                        print(f"    Found partial fraction: '{text}'")

                print(f"  Measurement patterns found: {measurement_patterns}")

                # Simple reconstruction for whole number + fraction pairs
                print(f"\n  PASS 4: Reconstructing split measurements...")

                # Look for whole numbers followed by fractions
                reconstructed = []
                used_indices = set()

                for i, item in enumerate(all_text_items_list):
                    if i in used_indices or item.get('merged', False):
                        continue

                    # Check if this is a whole number
                    if re.match(r'^\d+$', item['text']):
                        # Look for a fraction nearby (within 100 pixels horizontally)
                        for j, other in enumerate(all_text_items_list):
                            if j != i and j not in used_indices:
                                # Check if other is a fraction
                                if re.match(r'^\d+/\d+$', other['text']):
                                    # Check proximity
                                    x_dist = abs(other['x'] - item['x'])
                                    y_dist = abs(other['y'] - item['y'])

                                    if x_dist < 100 and y_dist < 30:  # Close horizontally, same line
                                        # Combine them
                                        combined = f"{item['text']} {other['text']}"
                                        avg_x = (item['x'] + other['x']) / 2
                                        avg_y = (item['y'] + other['y']) / 2

                                        reconstructed.append({
                                            'text': combined,
                                            'x': avg_x,
                                            'y': avg_y,
                                            'width': x_dist + 30,
                                            'left_bound': min(item['x'], other['x']) - 20,
                                            'right_bound': max(item['x'], other['x']) + 30
                                        })

                                        used_indices.add(i)
                                        used_indices.add(j)
                                        measurement_patterns.append(combined)
                                        print(f"    Reconstructed: '{item['text']}' + '{other['text']}' = '{combined}'")
                                        break

                # Add reconstructed measurements
                measurements.extend(reconstructed)

                # Add verified measurements directly from text items (skip if already used in reconstruction or merged)
                for idx, item in enumerate(all_text_items_list):
                    if idx in used_indices or item.get('merged', False):
                        continue  # Skip if already used in reconstruction or merged

                    text = item['text'].strip()
                    # Check if this is a measurement pattern
                    # Handle direct match or converted dash format with F suffix
                    converted_text = None
                    if text in measurement_patterns:
                        converted_text = text
                    elif re.match(r'^\d+-\d+/\d+(\s+F)?$', text):
                        # Convert dash format and remove F suffix to check against patterns
                        temp = text.replace('-', ' ')
                        if ' F' in temp:
                            temp = temp.replace(' F', '')
                        if temp in measurement_patterns:
                            converted_text = temp

                    if converted_text and not any(m['x'] == item['x'] and m['y'] == item['y'] for m in measurements):
                        # Check if original text had F suffix (finished size - no overlay)
                        is_finished = ' F' in text
                        measurements.append({
                            'text': converted_text,
                            'x': item['x'],
                            'y': item['y'],
                            'width': len(converted_text) * 15,  # Estimate
                            'left_bound': item['x'] - len(converted_text) * 7,
                            'right_bound': item['x'] + len(converted_text) * 7,
                            'finished_size': is_finished  # Track if this is a finished size
                        })
                        if is_finished:
                            print(f"    Added measurement: '{converted_text}' at ({item['x']:.0f}, {item['y']:.0f}) [FINISHED SIZE - no overlay]")
                        else:
                            print(f"    Added measurement: '{converted_text}' at ({item['x']:.0f}, {item['y']:.0f})")

                # Skip complex reconstruction for now - zoom verification should handle most cases
                '''
                # Find all items within proximity
                        for j, other in enumerate(all_text_items_list):
                            if j != i and j not in used_indices:
                                # Check if on same horizontal line (within 20 pixels vertically)
                                if abs(other['y'] - base_y) < 20:
                                    # Check if horizontally close (within 150 pixels)
                                    min_x = min(g['item']['x'] for g in group)
                                    max_x = max(g['item']['x'] for g in group)
                                    if abs(other['x'] - min_x) < 150 or abs(other['x'] - max_x) < 150:
                                        group.append({'idx': j, 'item': other})

                        # Debug groups formed
                        if len(group) > 1 and any(g['item']['text'] == '16 5' for g in group):
                            print(f"    [DEBUG GROUP] Formed group: {[g['item']['text'] for g in group]}")

                        # Sort group by x position
                        group.sort(key=lambda g: g['item']['x'])

                        # Try different combinations to see if they form our measurement
                        # Try joining all items in the group
                        combined_text = ' '.join(g['item']['text'] for g in group)
                        # Also try without spaces for cases like "16 5" + "8" -> "16 5/8"
                        combined_no_space = ''.join(g['item']['text'] for g in group)

                        # Check various formats
                        if combined_text == measurement:
                            # Direct match
                            avg_x = sum(g['item']['x'] for g in group) / len(group)
                            avg_y = sum(g['item']['y'] for g in group) / len(group)
                            measurements.append({
                                'text': measurement,
                                'x': avg_x,
                                'y': avg_y,
                                'width': max(g['item']['x'] for g in group) - min(g['item']['x'] for g in group) + 40,
                                'left_bound': min(g['item']['x'] for g in group) - 20,
                                'right_bound': max(g['item']['x'] for g in group) + 40
                            })
                            print(f"  Matched '{measurement}' at ({avg_x:.0f}, {avg_y:.0f})")
                            for g in group:
                                used_indices.add(g['idx'])
                            break

                        # Try reconstructing measurements from group
                        # Handle 3-item groups first: "16 5" + "/" + "8" -> "16 5/8"
                        if len(group) == 3:
                            text1 = group[0]['item']['text']
                            text2 = group[1]['item']['text']
                            text3 = group[2]['item']['text']

                            reconstructed = None
                            # Pattern: "16 5" + "/" + "8" -> "16 5/8"
                            if re.match(r'^\d+\s+\d+$', text1) and text2 == '/' and re.match(r'^\d+$', text3):
                                reconstructed = text1 + text2 + text3
                            # Pattern: "16" + "5" + "/8" or similar combinations
                            elif '/' in (text1 + text2 + text3):
                                reconstructed = text1 + ' ' + text2 + text3 if text2.startswith('/') or text3.startswith('/') else text1 + text2 + text3

                            if reconstructed and reconstructed == measurement:
                                avg_x = sum(g['item']['x'] for g in group) / len(group)
                                avg_y = sum(g['item']['y'] for g in group) / len(group)
                                measurements.append({
                                    'text': measurement,
                                    'x': avg_x,
                                    'y': avg_y,
                                    'width': max(g['item']['x'] for g in group) - min(g['item']['x'] for g in group) + 40,
                                    'left_bound': min(g['item']['x'] for g in group) - 20,
                                    'right_bound': max(g['item']['x'] for g in group) + 40
                                })
                                print(f"  Reconstructed '{measurement}' from '{text1}' + '{text2}' + '{text3}'")
                                for g in group:
                                    used_indices.add(g['idx'])
                                break

                        # Handle 2-item groups: "16" + "5/8" -> "16 5/8"
                        elif len(group) == 2:
                            text1 = group[0]['item']['text']
                            text2 = group[1]['item']['text']

                            # Try multiple reconstruction patterns
                            reconstructed = None

                            # Pattern 1: "16 5" + "8" -> "16 5/8"
                            # This handles cases where the "/" was lost in OCR
                            if re.match(r'^\d+\s+\d+$', text1) and re.match(r'^\d+$', text2):
                                reconstructed = text1 + '/' + text2
                                print(f"    [DEBUG] Trying to reconstruct '{text1}' + '{text2}' = '{reconstructed}' (looking for '{measurement}')")
                            # Pattern 2: "16" + "5/8" -> "16 5/8"
                            elif re.match(r'^\d+$', text1) and re.match(r'^\d+/\d+$', text2):
                                reconstructed = text1 + ' ' + text2
                            # Pattern 3: "32" + "15/16" -> "32 15/16"
                            elif re.match(r'^\d+$', text1) and text2.count('/') == 1:
                                reconstructed = text1 + ' ' + text2

                            if reconstructed and reconstructed == measurement:
                                avg_x = (group[0]['item']['x'] + group[1]['item']['x']) / 2
                                avg_y = (group[0]['item']['y'] + group[1]['item']['y']) / 2
                                measurements.append({
                                    'text': measurement,
                                    'x': avg_x,
                                    'y': avg_y,
                                    'width': abs(group[1]['item']['x'] - group[0]['item']['x']) + 40,
                                    'left_bound': min(group[0]['item']['x'], group[1]['item']['x']) - 20,
                                    'right_bound': max(group[0]['item']['x'], group[1]['item']['x']) + 40
                                })
                                print(f"  Reconstructed '{measurement}' from '{text1}' + '{text2}'")
                                used_indices.add(group[0]['idx'])
                                used_indices.add(group[1]['idx'])
                                break

                # Now match measurements to positions (for those not already matched)
                for measurement in measurement_patterns:
                    # Skip if already matched in the split measurements section
                    already_matched = any(m['text'] == measurement for m in measurements)
                    if already_matched:
                        continue

                    # Check if it's a simple whole number
                    if re.match(r'^\d+$', measurement):
                        # Whole number - should exist as-is
                        if measurement in all_text_items:
                            info = get_text_item_info(all_text_items, measurement)
                            if info:
                                measurements.append({
                                    'text': measurement,
                                    'x': info['x'],
                                    'y': info['y'],
                                    'width': len(measurement) * 15,  # Estimate
                                    'left_bound': info['x'] - len(measurement) * 7,
                                    'right_bound': info['x'] + len(measurement) * 7
                                })
                                print(f"  Matched whole number '{measurement}' at ({info['x']:.0f}, {info['y']:.0f})")
                        continue

                # Also check for standalone numbers in the individual text items that might be measurements
                # This catches cases where OCR detects "18" as part of other text
                for item in all_text_items_list:
                    text = item['text']
                    # Check if it's a whole number that could be a measurement
                    if re.match(r'^\d+$', text):
                        num = int(text)
                        # Check if it's in a reasonable range and not already added
                        if 10 <= num <= 50 and not any(m['x'] == item['x'] and m['y'] == item['y'] for m in measurements):
                            measurements.append({
                                'text': text,
                                'x': item['x'],
                                'y': item['y'],
                                'width': len(text) * 15,  # Estimate
                                'left_bound': item['x'] - len(text) * 7,
                                'right_bound': item['x'] + len(text) * 7
                            })
                            print(f"  Found potential measurement '{text}' at ({item['x']:.0f}, {item['y']:.0f})")
                            measurement_patterns.append(text)  # Add to patterns for classification

                    # Check if this measurement exists as single item with dash format (like "31-1/4 F")
                    dash_format = measurement.replace(" ", "-", 1)  # Convert "31 1/4 F" to "31-1/4 F"

                    # Check if this measurement exists as single item (like "5/1/2")
                    original_format = measurement.replace(" ", "/")  # Convert "5 1/2" to "5/1/2"

                    if dash_format in all_text_items:
                        # Found as dash format (like "31-1/4 F")
                        info = get_text_item_info(all_text_items, dash_format)
                        if info:
                            measurements.append({
                                'text': measurement,
                                'x': info['x'],
                                'y': info['y'],
                                'width': len(measurement) * 15,  # Estimate
                                'left_bound': info['x'] - len(measurement) * 7,
                                'right_bound': info['x'] + len(measurement) * 7
                            })
                    elif original_format in all_text_items:
                        # Found as single item (like "5/1/2")
                        info = get_text_item_info(all_text_items, original_format)
                        if info:
                            measurements.append({
                                'text': measurement,
                                'x': info['x'],
                                'y': info['y'],
                                'width': len(measurement) * 15,  # Estimate
                                'left_bound': info['x'] - len(measurement) * 7,
                                'right_bound': info['x'] + len(measurement) * 7
                            })
                    else:
                        # Try to find as separate parts
                        parts = measurement.split()  # e.g., ["45", "7/8"] or ["31", "1/4", "F"]

                        # For measurements with F suffix (finish size)
                        if len(parts) == 3 and parts[2] == 'F':
                            # Looking for "31 1/4 F" format converted from "31-1/4 F"
                            # The Vision API might detect "31-1/4 F" as "31-1", "/", "4", "F"
                            # So look for "31-1" as a single item (but it's "21-1" in page 17)
                            whole_part = parts[0]  # "21" or "31"
                            frac_parts = parts[1].split('/')  # ["1", "4"]
                            dash_number = f"{whole_part}-{frac_parts[0]}"  # "21-1"

                            if dash_number in all_text_items:
                                # Found the dash format number, get its position
                                info = get_text_item_info(all_text_items, dash_number)
                                if info:
                                    measurements.append({
                                        'text': measurement,
                                        'x': info['x'],
                                        'y': info['y'],
                                        'width': len(measurement) * 15,  # Estimate
                                        'left_bound': info['x'] - len(measurement) * 7,
                                        'right_bound': info['x'] + len(measurement) * 7
                                    })
                                    print(f"  Matched '{measurement}' via dash format at ({info['x']:.0f}, {info['y']:.0f})")
                        # For measurements with multiple parts, find the closest matching parts
                        elif len(parts) == 2:
                            # Debug for "20 3/16"
                            if measurement == "20 3/16":
                                print(f"  [DEBUG] Processing '20 3/16' as 2 parts:")
                                print(f"    Part 1: '{parts[0]}'")
                                print(f"    Part 2: '{parts[1]}'")

                            # Find all instances of first part
                            first_part_positions = []
                            first_part_widths = []
                            for ann in annotations[1:]:
                                if ann['description'] == parts[0]:
                                    vertices = ann['boundingPoly']['vertices']
                                    x = sum(v.get('x', 0) for v in vertices) / 4
                                    y = sum(v.get('y', 0) for v in vertices) / 4
                                    width = max(v.get('x', 0) for v in vertices) - min(v.get('x', 0) for v in vertices)
                                    first_part_positions.append((x, y))
                                    first_part_widths.append(width)

                                    if measurement == "20 3/16":
                                        print(f"    Found '{parts[0]}' at ({x:.0f}, {y:.0f}), width={width}")

                            # Find position of second part
                            second_part_pos = None
                            second_part_width = 0
                            if parts[1] in all_text_items:
                                info = get_text_item_info(all_text_items, parts[1])
                                if info:
                                    second_part_pos = (info['x'], info['y'])
                                # Get width of second part
                                for ann in annotations[1:]:
                                    if ann['description'] == parts[1]:
                                        vertices = ann['boundingPoly']['vertices']
                                        second_part_width = max(v.get('x', 0) for v in vertices) - min(v.get('x', 0) for v in vertices)
                                        if measurement == "20 3/16":
                                            print(f"    Found '{parts[1]}' at ({second_part_pos[0]:.0f}, {second_part_pos[1]:.0f}), width={second_part_width}")
                                        break

                            if first_part_positions and second_part_pos:
                                # Find the first part instance closest to the second part
                                min_distance = float('inf')
                                best_first_pos = None

                                for first_pos in first_part_positions:
                                    dist = ((first_pos[0] - second_part_pos[0])**2 +
                                           (first_pos[1] - second_part_pos[1])**2) ** 0.5
                                    if dist < min_distance:
                                        min_distance = dist
                                        best_first_pos = first_pos

                                if best_first_pos and min_distance < 200:  # Parts should be close together
                                    avg_x = (best_first_pos[0] + second_part_pos[0]) / 2
                                    avg_y = (best_first_pos[1] + second_part_pos[1]) / 2

                                    # Calculate actual text width for multi-part measurements
                                    if measurement == "20 3/16":
                                        # Calculate span from leftmost to rightmost point
                                        left_x = min(best_first_pos[0], second_part_pos[0])
                                        right_x = max(best_first_pos[0], second_part_pos[0])

                                        # Add half-widths of each part to get full extent
                                        if first_part_widths:
                                            left_x -= first_part_widths[0] / 2
                                        right_x += second_part_width / 2

                                        actual_width = right_x - left_x
                                        print(f"    [DEBUG] Full text width: {actual_width:.0f}px (from {left_x:.0f} to {right_x:.0f})")
                                        print(f"    [DEBUG] Center position: ({avg_x:.0f}, {avg_y:.0f})")

                                    # Store the actual width for multi-part measurements
                                    actual_width = None
                                    if first_part_widths and second_part_width:
                                        left_x = min(best_first_pos[0], second_part_pos[0])
                                        right_x = max(best_first_pos[0], second_part_pos[0])
                                        left_x -= first_part_widths[0] / 2
                                        right_x += second_part_width / 2
                                        actual_width = right_x - left_x

                                    measurements.append({
                                        'text': measurement,
                                        'x': avg_x,
                                        'y': avg_y,
                                        'width': actual_width,  # Store actual width
                                        'left_bound': left_x if actual_width else avg_x - 50,
                                        'right_bound': right_x if actual_width else avg_x + 50
                                    })
                                    print(f"  Matched '{measurement}' at ({avg_x:.0f}, {avg_y:.0f})")
                '''  # End of commented out complex reconstruction

    # No need for additional zoom verification - already done in PASS 2
    print(f"\n  Found {len(measurements)} unique measurements after verification")
    return measurements, full_text, all_text_items_list

def get_dominant_color_at_position(image, x, y, radius=15):
    """Get the dominant color around a position (for text color detection)"""
    # Define region around the text
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(image.shape[1], x + radius)
    y2 = min(image.shape[0], y + radius)

    # Get the region
    region = image[y1:y2, x1:x2]

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Find the dominant non-white color (text color)
    # Exclude white/light colors (high value, low saturation)
    mask = (hsv[:,:,1] > 30) & (hsv[:,:,2] < 200)  # Saturation > 30, Value < 200

    if np.any(mask):
        # Get average color of non-white pixels
        avg_color = np.mean(region[mask], axis=0)
        return avg_color

    # If no colored pixels found, return None
    return None

def check_for_arrow_at_endpoint(edges, x, y, radius=20):
    """
    Check if there's an arrow shape at a line endpoint
    Arrows typically have 2-3 converging lines at 20-45 degree angles
    """
    # Get region around the endpoint
    y1 = max(0, y - radius)
    y2 = min(edges.shape[0], y + radius)
    x1 = max(0, x - radius)
    x2 = min(edges.shape[1], x + radius)

    roi = edges[y1:y2, x1:x2]

    # Find lines in this small region
    lines = cv2.HoughLinesP(roi, 1, np.pi/180,
                            threshold=10,  # Lower threshold for small region
                            minLineLength=5,
                            maxLineGap=3)

    if lines is not None and len(lines) >= 2:
        # Check for converging lines (arrow pattern)
        # Simplified check: if we have 2+ short lines in this region, likely an arrow
        return True
    return False

def find_lines_near_measurement(image, measurement, image_path=""):
    """Find lines near a specific measurement position that match the text color"""
    x, y = int(measurement['x']), int(measurement['y'])

    # Use actual text width from Vision API if available, otherwise estimate
    text_height = 30  # Approximate height of text in pixels
    if 'width' in measurement and measurement['width']:
        text_width = int(measurement['width'])
        print(f"    Using actual text width: {text_width}px")
    else:
        text_width = len(measurement.get('text', '')) * 15  # Approximate width based on character count
        print(f"    Using estimated text width: {text_width}px")

    # Add padding for image skew or misalignment
    padding = 10  # pixels of padding to allow for skew
    text_height_with_padding = text_height + padding
    text_width_with_padding = text_width + padding

    # Get the color of the measurement text
    text_color = get_dominant_color_at_position(image, x, y)

    if text_color is not None:
        print(f"    Text color (BGR): [{text_color[0]:.0f}, {text_color[1]:.0f}, {text_color[2]:.0f}]")
        print(f"    Text bounds: width={text_width}px, height={text_height}px")
        # Green-ish colors typically for measurements
        is_green = text_color[1] > text_color[0] and text_color[1] > text_color[2]
        if is_green:
            print(f"    Color appears GREEN (typical for dimension lines)")

    # Create TWO different ROIs - one for horizontal lines, one for vertical lines
    # These ROIs AVOID the text area to find dimension lines

    # For horizontal lines: Create TWO strips - one LEFT of text, one RIGHT of text
    # (WIDTH text sits ON the horizontal line, so we look to the sides)
    h_strip_width = text_width * 2  # Look 2x text width to each side
    h_strip_height = text_height + 10  # Height centered on text Y

    # Left horizontal ROI
    h_left_x1 = max(0, x - text_width//2 - h_strip_width)
    h_left_x2 = x - text_width//2
    h_left_y1 = max(0, y - h_strip_height//2)
    h_left_y2 = min(image.shape[0], y + h_strip_height//2)

    # Right horizontal ROI
    h_right_x1 = x + text_width//2
    h_right_x2 = min(image.shape[1], x + text_width//2 + h_strip_width)
    h_right_y1 = h_left_y1
    h_right_y2 = h_left_y2

    # For vertical lines: Create TWO strips - one ABOVE text, one BELOW text
    # (HEIGHT text sits ON the vertical line, so we look above/below)
    v_strip_height = text_height * 4  # Look 4x text height above/below to capture full lines
    v_strip_width = text_width + 20  # Width centered on text X with some padding

    # Top vertical ROI
    v_top_x1 = max(0, x - v_strip_width//2)
    v_top_x2 = min(image.shape[1], x + v_strip_width//2)
    v_top_y1 = max(0, y - text_height//2 - v_strip_height)
    v_top_y2 = y - text_height//2

    # Bottom vertical ROI
    v_bottom_x1 = v_top_x1
    v_bottom_x2 = v_top_x2
    v_bottom_y1 = y + text_height//2
    v_bottom_y2 = min(image.shape[0], y + text_height//2 + v_strip_height)

    # Debug: Save ROIs for "45 1/4"
    if "45" in str(measurement.get('text', '')):
        # Draw debug image showing ROIs
        debug_img = image.copy()

        # Draw horizontal ROIs in blue
        cv2.rectangle(debug_img, (h_left_x1, h_left_y1), (h_left_x2, h_left_y2), (255, 0, 0), 2)
        cv2.rectangle(debug_img, (h_right_x1, h_right_y1), (h_right_x2, h_right_y2), (255, 0, 0), 2)
        cv2.putText(debug_img, "H_LEFT", (h_left_x1, h_left_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_img, "H_RIGHT", (h_right_x1, h_right_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw vertical ROIs in green
        cv2.rectangle(debug_img, (v_top_x1, v_top_y1), (v_top_x2, v_top_y2), (0, 255, 0), 2)
        cv2.rectangle(debug_img, (v_bottom_x1, v_bottom_y1), (v_bottom_x2, v_bottom_y2), (0, 255, 0), 2)
        cv2.putText(debug_img, "V_TOP", (v_top_x1, v_top_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_img, "V_BOTTOM", (v_bottom_x1, v_bottom_y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw text position in red
        cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)
        cv2.rectangle(debug_img, (x - text_width//2, y - text_height//2),
                     (x + text_width//2, y + text_height//2), (0, 0, 255), 2)
        cv2.putText(debug_img, "TEXT", (x - 20, y - text_height//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Save debug image
        debug_path = image_path.replace('.png', '_45_1_4_roi_debug.png')
        cv2.imwrite(debug_path, debug_img)
        print(f"    [DEBUG] Saved ROI visualization to: {debug_path}")
        print(f"    [DEBUG] Text at ({x}, {y}), width={text_width}, height={text_height}")
        print(f"    [DEBUG] Horizontal ROI: Left({h_left_x1},{h_left_y1})-({h_left_x2},{h_left_y2}) + Right({h_right_x1},{h_right_y1})-({h_right_x2},{h_right_y2})")
        print(f"    [DEBUG] Vertical ROI: Top({v_top_x1},{v_top_y1})-({v_top_x2},{v_top_y2}) + Bottom({v_bottom_x1},{v_bottom_y1})-({v_bottom_x2},{v_bottom_y2})")

    # Process horizontal and vertical ROIs separately
    horizontal_lines = []
    vertical_lines = []

    # Helper function to detect lines in an ROI
    def detect_lines_in_roi(roi_image, roi_x_offset, roi_y_offset, debug_name=""):
        if ("20" in str(measurement.get('text', '')) or "45" in str(measurement.get('text', ''))) and debug_name:
            print(f"      [DEBUG] Processing {debug_name} ROI: shape={roi_image.shape}")

        # Try color filtering - look for greenish colors
        if text_color is not None and text_color[1] > text_color[0] and text_color[1] > text_color[2]:
            # Text appears green - use green color detection
            # Convert ROI to HSV
            roi_hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

            # Debug output for "45 1/4"
            if "45" in str(measurement.get('text', '')) and debug_name:
                print(f"      [DEBUG] Text color BGR: {text_color} - detected as GREEN")

            # Create mask for green colors (hue roughly 30-90 for green range in HSV)
            # Green in HSV: Hue 30-90, Saturation > 30, Value > 50
            lower = np.array([25, 30, 50], dtype=np.uint8)  # Lower bound for green
            upper = np.array([95, 255, 255], dtype=np.uint8)  # Upper bound for green
            color_mask = cv2.inRange(roi_hsv, lower, upper)

            # Apply morphology
            kernel = np.ones((3,3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            # Edge detection with lower thresholds for better line detection
            edges = cv2.Canny(color_mask, 30, 100)
        else:
            # Fallback to grayscale
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)

        # Find lines - reduced thresholds for better detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=15,  # Reduced from 30
                                minLineLength=20,  # Reduced from 30
                                maxLineGap=30)  # Increased from 20

        if ("20" in str(measurement.get('text', '')) or "45" in str(measurement.get('text', ''))) and debug_name:
            if lines is not None:
                print(f"      [DEBUG] {debug_name}: Found {len(lines)} lines")
                # Show details of each line for debugging
                for i, line in enumerate(lines):
                    lx1, ly1, lx2, ly2 = line[0]
                    length = ((lx2-lx1)**2 + (ly2-ly1)**2)**0.5
                    angle = np.abs(np.degrees(np.arctan2(ly2-ly1, lx2-lx1)))
                    print(f"        Line {i}: ({lx1},{ly1}) to ({lx2},{ly2}), length={length:.1f}, angle={angle:.1f}")
            else:
                print(f"      [DEBUG] {debug_name}: No lines found")

        # Convert line coordinates back to full image coordinates
        if lines is not None:
            adjusted_lines = []
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                adjusted_lines.append([lx1 + roi_x_offset, ly1 + roi_y_offset,
                                      lx2 + roi_x_offset, ly2 + roi_y_offset])
            return adjusted_lines
        return []

    # Search for horizontal lines in left and right ROIs
    h_left_roi = image[h_left_y1:h_left_y2, h_left_x1:h_left_x2]
    h_right_roi = image[h_right_y1:h_right_y2, h_right_x1:h_right_x2]

    left_h_lines = detect_lines_in_roi(h_left_roi, h_left_x1, h_left_y1, "Left-H")
    right_h_lines = detect_lines_in_roi(h_right_roi, h_right_x1, h_right_y1, "Right-H")

    # Filter for horizontal lines (angle < 20 or > 160 degrees)
    for line in left_h_lines + right_h_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
        if angle < 20 or angle > 160:
            horizontal_lines.append({
                'coords': (lx1, ly1, lx2, ly2),
                'distance': abs(y - (ly1 + ly2) / 2),
                'type': 'horizontal_line'
            })

    # Search for vertical lines in top and bottom ROIs
    v_top_roi = image[v_top_y1:v_top_y2, v_top_x1:v_top_x2]
    v_bottom_roi = image[v_bottom_y1:v_bottom_y2, v_bottom_x1:v_bottom_x2]

    top_v_lines = detect_lines_in_roi(v_top_roi, v_top_x1, v_top_y1, "Top-V")
    bottom_v_lines = detect_lines_in_roi(v_bottom_roi, v_bottom_x1, v_bottom_y1, "Bottom-V")

    # Filter for vertical lines (angle between 70 and 110 degrees)
    for line in top_v_lines + bottom_v_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
        if 70 < angle < 110:
            # For vertical lines found in V-ROIs (above/below text),
            # ignore lines that are too close to the text center horizontally
            # These are likely spurious detections
            x_distance = abs(x - (lx1 + lx2) / 2)
            if x_distance > 5:  # Require at least 5 pixels offset from text center
                vertical_lines.append({
                    'coords': (lx1, ly1, lx2, ly2),
                    'distance': x_distance,
                    'type': 'vertical_line'
                })

    # Debug output for measurements
    if "20" in str(measurement.get('text', '')) or "45" in str(measurement.get('text', '')) or "14" in str(measurement.get('text', '')):
        print(f"    [DEBUG] Found {len(horizontal_lines)} horizontal lines in H-ROIs")
        print(f"    [DEBUG] Found {len(vertical_lines)} vertical lines in V-ROIs")
        if horizontal_lines:
            for h in horizontal_lines:
                print(f"      H-line distance: {h['distance']:.1f}")
        if vertical_lines:
            for v in vertical_lines:
                print(f"      V-line distance: {v['distance']:.1f}")

    # If no lines found, return None
    if not horizontal_lines and not vertical_lines:
        return None

    # Choose the closest line of each type
    best_h = min(horizontal_lines, key=lambda l: l['distance']) if horizontal_lines else None
    best_v = min(vertical_lines, key=lambda l: l['distance']) if vertical_lines else None

    # Determine which orientation is more likely
    if best_h and best_v:
        # If we have both, pick the closer one
        if best_h['distance'] < best_v['distance']:
            return {
                'line': best_h['coords'],
                'orientation': 'horizontal_line',
                'distance': best_h['distance']
            }
        else:
            return {
                'line': best_v['coords'],
                'orientation': 'vertical_line',
                'distance': best_v['distance']
            }
    elif best_h:
        return {
            'line': best_h['coords'],
            'orientation': 'horizontal_line',
            'distance': best_h['distance']
        }
    elif best_v:
        return {
            'line': best_v['coords'],
            'orientation': 'vertical_line',
            'distance': best_v['distance']
        }

    return None

def classify_measurements_by_local_lines(image, measurements, all_text_items_list=None, image_path=""):
    classified = {
        'vertical': [],
        'horizontal': [],
        'unclassified': []
    }

    measurement_details = []

    for meas in measurements:
        print(f"\n  Analyzing {meas['text']} at ({meas['x']:.0f}, {meas['y']:.0f})")

        # Find lines near this measurement
        line_info = find_lines_near_measurement(image, meas, image_path)

        if line_info:
            print(f"    Found {line_info['orientation']} at distance {line_info['distance']:.0f}")

            # Classification based on line orientation:
            # Horizontal line = WIDTH (measuring horizontally)
            # Vertical line = HEIGHT (measuring vertically)
            if line_info['orientation'] == 'horizontal_line':
                classified['horizontal'].append(meas['text'])  # WIDTH measurement
            elif line_info['orientation'] == 'vertical_line':
                classified['vertical'].append(meas['text'])  # HEIGHT measurement

            measurement_details.append({
                'measurement': meas,
                'line': line_info
            })
        else:
            print(f"    No clear line found near measurement")
            classified['unclassified'].append(meas['text'])
            measurement_details.append({
                'measurement': meas,
                'line': None
            })

    return classified, measurement_details, all_text_items_list

def main():
    import os
    print("=" * 80)
    print("MEASUREMENT-BASED DIMENSION DETECTOR")
    print("Finds measurements first, then looks for lines near them")
    print("=" * 80)
    print()

    # Get API key from environment variable for security
    API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')
    if not API_KEY:
        print("[ERROR] Please set GOOGLE_VISION_API_KEY environment variable")
        print("Example: export GOOGLE_VISION_API_KEY='your-api-key-here'")
        return

    # Get image path from command line or default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "page_16.png"

    # Get base name and directory for output files
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(os.path.abspath(image_path))

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Cannot load image")
        return

    print(f"[OK] Image: {image.shape[1]}x{image.shape[0]} pixels")

    # Step 1: Get measurements with positions from Vision API
    print("\n[1/2] Getting measurements from Vision API...")
    measurements, full_text, all_text_items_list = get_measurements_from_vision_api(image_path, API_KEY)
    print(f"\n  Found {len(measurements)} measurements:")
    for m in measurements:
        print(f"    {m['text']} at ({m['x']:.0f}, {m['y']:.0f})")

    # Step 2: Find lines near each measurement and classify
    print("\n[2/2] Finding lines near each measurement...")
    classified, details, _ = classify_measurements_by_local_lines(image, measurements, all_text_items_list, image_path)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("-" * 60)

    print(f"VERTICAL (Heights): {len(classified['vertical'])}")
    for v in classified['vertical']:
        print(f"  - {v}")

    print(f"\nHORIZONTAL (Widths): {len(classified['horizontal'])}")
    for h in classified['horizontal']:
        print(f"  - {h}")

    if classified['unclassified']:
        print(f"\nUNCLASSIFIED measurements: {len(classified['unclassified'])}")
        for u in classified['unclassified']:
            print(f"  - {u}")

    # Don't create openings here - let proximity_pairing_detector.py handle that
    print(f"\nMEASUREMENTS FOUND: {len(measurements)}")
    print(f"  Vertical (Heights): {len(classified['vertical'])}")
    print(f"  Horizontal (Widths): {len(classified['horizontal'])}")
    print(f"  Unclassified: {len(classified.get('unclassified', []))}")

    # Create annotated image
    annotated = image.copy()

    # Draw measurement locations and their associated lines
    for detail in details:
        meas = detail['measurement']
        x, y = int(meas['x']), int(meas['y'])

        if detail['line']:
            line_info = detail['line']
            x1, y1, x2, y2 = line_info['line']

            # Color based on what the measurement represents
            # CORRECTED:
            # vertical_line with measurement = HEIGHT
            # horizontal_line with measurement = WIDTH
            if line_info['orientation'] == 'vertical_line':
                color = (255, 0, 0)  # Blue for HEIGHT
                label = "H"  # Height
            else:  # horizontal_line
                color = (0, 0, 255)  # Red for WIDTH
                label = "W"  # Width

            # Draw the line
            cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            # Draw measurement location
            cv2.circle(annotated, (x, y), 15, color, 2)
            cv2.putText(annotated, f"{label}:{meas['text']}", (x+20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Unclassified - draw in gray
            cv2.circle(annotated, (x, y), 15, (128, 128, 128), 2)
            cv2.putText(annotated, f"?:{meas['text']}", (x+20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

    # Add summary
    cv2.putText(annotated, f"Vertical: {classified['vertical']}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(annotated, f"Horizontal: {classified['horizontal']}",
                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    output_path = os.path.join(image_dir, f"{base_name}_measurements_detected.png")
    cv2.imwrite(output_path, annotated)
    print(f"\n[OK] Annotated image saved: {output_path}")

    # Look for overlay/hinge notations and room name in the full text
    overlay_info = ""
    room_name = ""
    customer_notations = []

    if full_text:
        # Look for common overlay patterns
        overlay_patterns = [
            r'(\d+/\d+)\s+OL',  # Like "5/8 OL" or "1/2 OL"
            r'(\d+/\d+)\s+OL\s+(\w+)',  # Like "5/8 OL KITCHEN" - captures room name
            r'FULL\s+OL',  # Full overlay
            r'PARTIAL\s+OL',  # Partial overlay
        ]
        for pattern in overlay_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                overlay_info = match.group(0)
                print(f"\n[OK] Found overlay info: {overlay_info}")
                # Extract room name if it's in the overlay info
                if match.groups() and len(match.groups()) > 1:
                    room_name = match.group(2)
                    print(f"[OK] Found room name: {room_name}")
                break

        # If no room name found in overlay, look for common room names
        if not room_name:
            room_patterns = ['KITCHEN', 'BATHROOM', 'BATH', 'BEDROOM', 'LAUNDRY', 'CLOSET', 'PANTRY', 'GARAGE', 'OFFICE']
            for room in room_patterns:
                if room in full_text.upper():
                    room_name = room
                    print(f"[OK] Found room name: {room_name}")
                    break

        # Look for customer notations (hinges, special instructions)
        notation_patterns = [
            r'\bNH\b',  # NH (no hinges)
            r'\bno\s+hinges?\b',  # no hinge, no hinges
            r'\bNO\s+HINGES?\b',  # NO HINGE, NO HINGES
            r'\bsoft\s+close\b',  # soft close
            r'\bSC\b',  # SC (soft close abbreviation)
            r'\bpush\s+to\s+open\b',  # push to open
            r'\bPTO\b',  # PTO (push to open abbreviation)
            r'\bflip\s+up\b',  # flip up door
            r'\bFU\b',  # FU (flip up abbreviation)
            r'\bglass\b',  # glass door
            r'\bmicrowave\b',  # microwave shelf
            r'\bMW\b',  # MW (microwave abbreviation)
            r'\blazy\s+susan\b',  # lazy susan
            r'\bLS\b',  # LS (lazy susan abbreviation)
            r'\btrash\s+pull\s*out\b',  # trash pull out
            r'\bTPO\b',  # TPO (trash pull out)
            r'\bspice\s+rack\b',  # spice rack
            r'\bdrawer\s+box\b',  # drawer box
            r'\bDB\b',  # DB (drawer box)
            r'\broll\s*out\b',  # roll out shelf
            r'\bRO\b',  # RO (roll out)
        ]

        # Find all notations in the text
        for pattern in notation_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                notation_text = match.group(0)
                # Get position of this notation from all_text_items_list
                for item in all_text_items_list:
                    if notation_text.upper() in item['text'].upper():
                        customer_notations.append({
                            'text': notation_text,
                            'x': item['x'],
                            'y': item['y']
                        })
                        print(f"[OK] Found customer notation: '{notation_text}' at ({item['x']:.0f}, {item['y']:.0f})")
                        break

    # Save results
    results = {
        'measurements': [{
            'text': m['text'],
            'position': [m['x'], m['y']],
            'bounds': [m.get('left_bound', m['x']-50), m.get('right_bound', m['x']+50)],
            'finished_size': m.get('finished_size', False)  # Include F suffix flag
        } for m in measurements],
        'vertical': classified['vertical'],
        'horizontal': classified['horizontal'],
        'unclassified': classified['unclassified'],
        'overlay_info': overlay_info,
        'room_name': room_name,
        'customer_notations': customer_notations
    }

    output_json = os.path.join(image_dir, f"{base_name}_measurements_data.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved: {output_json}")

if __name__ == "__main__":
    main()