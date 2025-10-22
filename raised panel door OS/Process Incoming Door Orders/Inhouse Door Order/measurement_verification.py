#!/usr/bin/env python3
"""
Measurement verification functions using zoom OCR.
Handles zooming into interest areas and extracting precise measurements.
"""

import cv2
import numpy as np
import base64
import requests
import re
import os
from measurement_config import ZOOM_CONFIG, HSV_CONFIG, VALIDATION_CONFIG, GROUPING_CONFIG, OVERLAY_PATTERN
from image_preprocessing import apply_hsv_preprocessing


def extract_measurements_from_text(text):
    """Extract measurement patterns from text"""
    # Pattern for measurements: "24 3/4", "11 1/4", "5/8", "20", "6", etc.
    # Matches: whole + fraction | fraction | decimal | any whole number
    pattern = r'\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\b\d+\b'
    matches = re.findall(pattern, text)
    return matches


def verify_measurement_at_center(image_path, center, bounds, texts, api_key, center_index=0):
    """Phase 2: Zoom and verify measurement at a specific center"""

    # Convert Windows network paths to Unix-style for OpenCV
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Use bounds with configured padding
    padding = ZOOM_CONFIG['padding']
    x1 = max(0, int(bounds['left'] - padding))
    y1 = max(0, int(bounds['top'] - padding))
    x2 = min(w, int(bounds['right'] + padding))
    y2 = min(h, int(bounds['bottom'] + padding))

    # Crop and zoom with configured factor
    cropped = image[y1:y2, x1:x2]
    zoom_factor = ZOOM_CONFIG['zoom_factor']
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Apply HSV
    zoomed_hsv = apply_hsv_preprocessing(zoomed)

    # Save debug images with better naming
    # Extract page number from image path if available
    page_num = ""
    if "page_" in image_path.lower():
        page_match = re.search(r'page_(\d+)', image_path.lower())
        if page_match:
            page_num = f"page{page_match.group(1)}_"

    # Create readable text string from texts list
    text_str = "_".join(texts[:3]).replace("/", "-").replace(" ", "")
    if len(text_str) > 20:
        text_str = text_str[:20]

    # Get the directory of the source image to save debug images in the same location
    image_dir = os.path.dirname(os.path.abspath(image_path))

    # Save with temporary names first in the same directory as source image
    temp_file_orig = os.path.join(image_dir, f"TEMP_{page_num}C{center_index:02d}_pos{int(center[0])}x{int(center[1])}_{text_str}_original.png")
    temp_file_hsv = os.path.join(image_dir, f"TEMP_{page_num}C{center_index:02d}_pos{int(center[0])}x{int(center[1])}_{text_str}_hsv.png")
    cv2.imwrite(temp_file_orig, zoomed)
    cv2.imwrite(temp_file_hsv, zoomed_hsv)

    # Run OCR
    _, buffer = cv2.imencode('.png', zoomed_hsv)
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
        return None

    result = response.json()
    if 'responses' not in result or not result['responses']:
        return None

    annotations = result['responses'][0].get('textAnnotations', [])
    if not annotations:
        return None

    full_text = annotations[0]['description']
    # Debug: show raw text first
    print(f"  OCR raw: {repr(full_text)}")
    print(f"  OCR: '{full_text.replace(chr(10), ' ')}'")

    # Skip if this is clearly not a measurement (like "H2" or other letter+number combos)
    if re.match(r'^[A-Z]+\d+$', full_text.strip()):
        print(f"    Skipping non-measurement text: {full_text.strip()}")
        return None

    # Find all measurements
    patterns = [
        r'\b(\d+\s+\d+/\d+)\b',      # "5 1/2"
        r'[-]?(\d+)\s*[,.]?\s*(\d+/\d+)', # "-22,3/8" or "22 3/8"
        r'\b(\d+)\s+(\d+/\d+)\b',    # "5" "1/2" on different lines
        r'(\d{1,2})\s+(\d+/\d+)',    # "20 1/16" even if OCR sees "0 1/16"
        r'[-]?(\d{1,2})[-]?(?!\s*[/:])', # Standalone "20" or "-23-" or "23-"
    ]

    measurements = []
    for pattern in patterns:
        for match in re.finditer(pattern, full_text):
            if match.lastindex == 2:
                meas = f"{match.group(1)} {match.group(2)}"
            else:
                meas = match.group(1)

            # Validate measurement range
            first_num = re.match(r'^(\d+)', meas)
            if first_num:
                num = int(first_num.group(1))
                if VALIDATION_CONFIG['min_value'] <= num <= VALIDATION_CONFIG['max_value']:
                    measurements.append(meas)

    # Remove duplicates
    measurements = list(dict.fromkeys(measurements))

    if len(measurements) == 0:
        return None
    elif len(measurements) == 1:
        return measurements[0]
    else:
        # Multiple measurements - find closest to center
        print(f"  Multiple measurements: {measurements}")

        zoomed_h, zoomed_w = zoomed.shape[:2]
        center_x = zoomed_w / 2
        center_y = zoomed_h / 2

        best_meas = None
        best_dist = float('inf')

        for meas in measurements:
            # Find this measurement in annotations
            for ann in annotations[1:]:
                text = ann['description'].strip()
                if text in meas or meas.startswith(text):
                    vertices = ann.get('boundingPoly', {}).get('vertices', [])
                    if len(vertices) >= 4:
                        x_coords = [v.get('x', 0) for v in vertices]
                        y_coords = [v.get('y', 0) for v in vertices]
                        bbox_x = sum(x_coords) / 4
                        bbox_y = sum(y_coords) / 4

                        dist = ((bbox_x - center_x)**2 + (bbox_y - center_y)**2)**0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_meas = meas

        actual_bounds = None
        if best_meas:
            print(f"  Chose closest: '{best_meas}'")

        # Rename temp files with detected measurement
        if best_meas:
            meas_str = best_meas.replace("/", "-").replace(" ", "_")
            final_file_orig = temp_file_orig.replace("TEMP_", "DEBUG_").replace("_original.png", f"_FOUND_{meas_str}_original.png")
            final_file_hsv = temp_file_hsv.replace("TEMP_", "DEBUG_").replace("_hsv.png", f"_FOUND_{meas_str}_hsv.png")

            if os.path.exists(temp_file_orig):
                os.rename(temp_file_orig, final_file_orig)
            if os.path.exists(temp_file_hsv):
                os.rename(temp_file_hsv, final_file_hsv)
            print(f"    Saved: {os.path.basename(final_file_hsv)}")
        else:
            # No measurement found - rename to show that
            final_file_orig = temp_file_orig.replace("TEMP_", "DEBUG_").replace("_original.png", "_FOUND_NONE_original.png")
            final_file_hsv = temp_file_hsv.replace("TEMP_", "DEBUG_").replace("_hsv.png", "_FOUND_NONE_hsv.png")

            if os.path.exists(temp_file_orig):
                os.rename(temp_file_orig, final_file_orig)
            if os.path.exists(temp_file_hsv):
                os.rename(temp_file_hsv, final_file_hsv)

        return (best_meas, actual_bounds)


def verify_measurement_at_center_with_logic(image_path, center, bounds, texts, api_key, center_index=0, save_debug=False, non_measurement_text_exclusions=None):
    """Same as verify_measurement_at_center but returns (measurement, logic_description)

    Returns: ((measurement_value, actual_bounds, special_notation, raw_ocr_text), logic_description)
    """
    logic_steps = []

    # Phase 2: Zoom and verify measurement at a specific center
    cx, cy = center
    x_coords = [bounds['left'], bounds['right']]
    y_coords = [bounds['top'], bounds['bottom']]

    # Define crop region with different horizontal and vertical padding
    padding_v = ZOOM_CONFIG['padding']  # Vertical padding
    padding_h = ZOOM_CONFIG.get('padding_horizontal', padding_v)  # Horizontal padding (wider)
    crop_x1 = max(0, int(min(x_coords) - padding_h))
    crop_y1 = max(0, int(min(y_coords) - padding_v))
    crop_x2 = int(max(x_coords) + padding_h)
    crop_y2 = int(max(y_coords) + padding_v)

    # Convert Windows network paths to Unix-style for OpenCV
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    # Load and crop image
    image = cv2.imread(image_path)
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Apply zoom to make text larger for better OCR
    zoom_factor = ZOOM_CONFIG.get('zoom_factor', 3)
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Apply HSV preprocessing to zoomed image
    hsv = cv2.cvtColor(zoomed, cv2.COLOR_BGR2HSV)
    lower_green = np.array(HSV_CONFIG['lower_green'])
    upper_green = np.array(HSV_CONFIG['upper_green'])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    preprocessed = cv2.bitwise_not(mask_3channel)

    # Save debug images if requested
    if save_debug:
        # Extract page number from image path (handle both "page_5" and "page-5" formats)
        page_num = ""
        if "page" in image_path.lower():
            page_match = re.search(r'page[-_]?(\d+)', image_path.lower())
            if page_match:
                page_num = f"page{page_match.group(1)}_"

        # Create text string from texts list
        text_str = "_".join(texts[:3]).replace("/", "-").replace(" ", "")[:20]

        # Get directory of source image
        image_dir = os.path.dirname(os.path.abspath(image_path))

        # Save debug images - now including zoomed versions
        # Use center_index + 1 to match 1-based numbering in visualization (M1, M2, M3...)
        debug_base = f"DEBUG_{page_num}M{center_index + 1}_pos{int(cx)}x{int(cy)}_{text_str}"
        debug_crop = os.path.join(image_dir, f"{debug_base}_crop.png")
        debug_zoom = os.path.join(image_dir, f"{debug_base}_zoom{zoom_factor}x.png")
        debug_hsv = os.path.join(image_dir, f"{debug_base}_hsv_zoom{zoom_factor}x.png")

        cv2.imwrite(debug_crop, cropped)
        cv2.imwrite(debug_zoom, zoomed)
        cv2.imwrite(debug_hsv, preprocessed)
        print(f"    [DEBUG] Saved: {os.path.basename(debug_crop)}")
        print(f"    [DEBUG] Saved: {os.path.basename(debug_zoom)}")
        print(f"    [DEBUG] Saved: {os.path.basename(debug_hsv)}")

    # Run OCR on zoomed preprocessed region
    _, buffer = cv2.imencode('.png', preprocessed)
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
        return None, "OCR API failed"

    result = response.json()
    if 'responses' not in result or not result['responses']:
        return None, "No OCR response from API"

    annotations = result['responses'][0].get('textAnnotations', [])
    if not annotations:
        print(f"  OCR: [No text detected in zoomed region]")
        print(f"  → Flagging for Claude verification")
        # Return special flag so Claude can verify if there's actually nothing there
        return ('NEEDS_CLAUDE_VERIFICATION', None, None, '', False), "Phase 2 OCR found no text - needs Claude verification"

    full_text = annotations[0]['description']
    raw_full_text = full_text  # Store the completely raw OCR text before any processing
    # Debug: show raw text first
    print(f"  OCR raw: {repr(full_text)}")
    print(f"  OCR: '{full_text.replace(chr(10), ' ')}'")

    # Strip arrow symbols from full_text (but keep in raw_full_text for bounds calculation)
    # Arrow symbols: → ← ↑ ↓ (U+2192, U+2190, U+2191, U+2193)
    arrow_symbols = ['\u2192', '\u2190', '\u2191', '\u2193']  # → ← ↑ ↓
    original_full_text = full_text
    for arrow in arrow_symbols:
        full_text = full_text.replace(arrow, '')
    if full_text != original_full_text:
        print(f"  Stripped arrow symbols: '{original_full_text}' → '{full_text}'")

    # Skip if this is clearly not a measurement
    if re.match(r'^[A-Z]+\d+$', full_text.strip()):
        print(f"    Skipping non-measurement text: {full_text.strip()}")
        logic_steps.append(f"OCR saw '{full_text.strip()}'")
        logic_steps.append("Pattern matches label (letter+number), not measurement")
        return None, " | ".join(logic_steps)

    # Group annotations by proximity to form measurements
    # Similar to Phase 1 but with tighter thresholds for zoomed view
    individual_items = []
    for ann in annotations[1:]:  # Skip full text
        text = ann['description'].strip()
        # Clean up text - skip punctuation-only items (including various dash types)
        if text in ['.', ',', '-', '+', '—', '−', '–', '·']:  # Skip punctuation
            continue

        # Strip leading/trailing dashes from text for cleaner grouping
        cleaned_text = text.strip('—−–-')

        vertices = ann['boundingPoly']['vertices']
        x = sum(v.get('x', 0) for v in vertices) / 4
        y = sum(v.get('y', 0) for v in vertices) / 4
        individual_items.append({
            'text': cleaned_text,  # Use cleaned text without edge dashes
            'original_text': text,  # Keep original for reference
            'x': x,
            'y': y,
            'vertices': vertices  # Keep vertices for bounds calculation
        })

    # Debug: show what items were found
    if len(individual_items) > 0 and len(individual_items) < 10:
        print(f"    Found {len(individual_items)} individual items after filtering:")
        for item in individual_items:
            print(f"      '{item['text']}' at ({item['x']:.0f}, {item['y']:.0f})")

    # Filter out excluded patterns (like "OR" from "or 2")
    from measurement_config import EXCLUDE_PATTERNS
    filtered_by_exclude = []
    for item in individual_items:
        text_upper = item['text'].upper()
        if text_upper in EXCLUDE_PATTERNS:
            print(f"    Excluding '{item['text']}' - matches exclude pattern")
            continue
        filtered_by_exclude.append(item)
    individual_items = filtered_by_exclude

    # Group items that are horizontally adjacent (for measurements like "13 7/8")
    # Adjust thresholds for zoomed image
    y_threshold = 15 * zoom_factor  # Scale for zoom
    x_threshold = 200 * zoom_factor  # Increased to handle wider spaced measurements with dashes

    # Filter out false positive "1"s (vertical lines detected as "1")
    filtered_items = []
    for item in individual_items:
        if item['text'] == '1':
            # Check if there's a fraction nearby
            has_nearby_fraction = False
            for other in individual_items:
                # Check for complete fractions or fraction parts
                if other['text'] in ['/2', '/4', '/8', '/16', '1/2', '1/4', '3/8', '5/8', '3/4', '7/8', '1/16', '3/16', '5/16', '7/16', '9/16', '11/16', '13/16', '15/16']:
                    x_dist = abs(other['x'] - item['x'])
                    y_dist = abs(other['y'] - item['y'])
                    if x_dist < x_threshold and y_dist < y_threshold:
                        has_nearby_fraction = True
                        break
                # Also check for standalone "/" which indicates the "1" is part of a fraction
                if other['text'] == '/':
                    x_dist = abs(other['x'] - item['x'])
                    y_dist = abs(other['y'] - item['y'])
                    if x_dist < x_threshold and y_dist < y_threshold:
                        has_nearby_fraction = True
                        break

            # Check aspect ratio if we have bounding box info
            if not has_nearby_fraction:
                # For now, skip isolated "1"s as they're likely false positives
                print(f"    Filtered out isolated '1' at ({item['x']:.0f}, {item['y']:.0f}) - likely a false positive")
                continue

        filtered_items.append(item)

    individual_items = filtered_items

    # Remove duplicate numbers that are very close together (OCR duplicates)
    deduplicated_items = []
    skip_indices = set()

    for i, item in enumerate(individual_items):
        if i in skip_indices:
            continue

        # Check for duplicates (same text, nearby position)
        for j, other in enumerate(individual_items):
            if j <= i or j in skip_indices:
                continue

            # If same text and within reasonable distance, likely a duplicate
            if item['text'] == other['text']:
                x_dist = abs(other['x'] - item['x'])
                y_dist = abs(other['y'] - item['y'])
                # Increased threshold to 100 pixels to catch duplicates on different lines
                if x_dist < 100 and y_dist < 100:
                    # Keep the first occurrence, skip the duplicate
                    skip_indices.add(j)
                    print(f"    Filtered out duplicate '{other['text']}' at ({other['x']:.0f}, {other['y']:.0f}) - duplicate of ({item['x']:.0f}, {item['y']:.0f})")

        deduplicated_items.append(item)

    individual_items = deduplicated_items

    measurement_groups = []
    used_indices = set()

    for i, item in enumerate(individual_items):
        if i in used_indices:
            continue

        group = [item]
        used_indices.add(i)

        # Look for items on same horizontal line
        for j, other in enumerate(individual_items):
            if j in used_indices or j <= i:
                continue

            # Check if on same line (y within threshold) and close horizontally (x within threshold)
            y_dist = abs(other['y'] - item['y'])
            if y_dist <= y_threshold:
                # Check if it's to the right and close enough
                x_dist = other['x'] - group[-1]['x']  # Distance from rightmost item in group
                if 0 < x_dist < x_threshold:
                    group.append(other)
                    used_indices.add(j)

        # Build measurement text from group
        group.sort(key=lambda x: x['x'])  # Sort by x position
        # Join components, but don't add spaces around "/" (for fractions)
        parts = []
        for i, g in enumerate(group):
            text = g['text']
            if i == 0:
                parts.append(text)
            elif text == '/':
                # Don't add space before slash
                parts.append('/')
            elif i > 0 and group[i-1]['text'] == '/':
                # Don't add space after slash
                parts.append(text)
            else:
                # Normal case - add space before
                parts.append(' ' + text)
        measurement_text = ''.join(parts)

        # Calculate group center
        avg_x = sum(g['x'] for g in group) / len(group)
        avg_y = sum(g['y'] for g in group) / len(group)

        measurement_groups.append({
            'text': measurement_text,
            'x': avg_x,
            'y': avg_y,
            'components': group,
            'raw_full_text': raw_full_text  # Store the original full OCR text
        })

    print(f"  Formed {len(measurement_groups)} measurement groups:")
    for mg in measurement_groups:
        print(f"    '{mg['text']}' at ({mg['x']:.0f}, {mg['y']:.0f})")

    # Accept all measurement groups - just clean the text
    valid_measurements = []

    for mg in measurement_groups:
        # Clean text (remove leading/trailing punctuation)
        cleaned_text = mg['text'].strip()
        # Remove leading/trailing dashes, commas, and asterisks (including Unicode minus sign and en dash)
        cleaned_text = cleaned_text.strip('—−–-.,*')
        # Strip arrow symbols (→ ← ↑ ↓)
        for arrow_sym in ['\u2192', '\u2190', '\u2191', '\u2193']:
            cleaned_text = cleaned_text.replace(arrow_sym, '')
        cleaned_text = cleaned_text.strip()
        # Replace internal commas with spaces
        cleaned_text = cleaned_text.replace(',', ' ')
        # Remove apostrophes (they shouldn't be in measurements)
        cleaned_text = cleaned_text.replace("'", " ")
        # Remove " F" or "F" suffix (finished size indicator) from measurement text
        # It will be kept as a flag, but removed from displayed text
        cleaned_text = cleaned_text.rstrip(' F').rstrip('F')
        # Clean up any double spaces
        cleaned_text = ' '.join(cleaned_text.split())
        cleaned_text = cleaned_text.strip()

        # Filter out components with leading/trailing dashes or punctuation-only
        # BUT keep special notation components (F, O) for bounds calculation
        cleaned_components = []

        # Find first component that contains a digit
        first_digit_index = None
        for idx, comp in enumerate(mg['components']):
            if any(c.isdigit() for c in comp['text']):
                first_digit_index = idx
                break

        for idx, comp in enumerate(mg['components']):
            comp_text = comp['text']

            # Skip leading text-only items (before first digit)
            if first_digit_index is not None and idx < first_digit_index:
                # This component comes BEFORE the first digit
                if not any(c.isdigit() for c in comp_text):
                    # Text-only component before number - skip it
                    print(f"    Filtered leading notation '{comp_text}' (before first number)")
                    continue

            # Special notation (F for finished size, O for opening size) - always keep
            if comp_text in ['F', 'O']:
                cleaned_components.append(comp)
                continue

            # Skip arrow symbols (→ ← ↑ ↓)
            if comp_text in ['\u2192', '\u2190', '\u2191', '\u2193', '→', '←', '↑', '↓']:
                print(f"    Filtered arrow symbol: '{comp_text}'")
                continue

            # Skip if starts or ends with dash (vertices include the dash)
            if comp_text and (comp_text[0] in '—−–-' or comp_text[-1] in '—−–-'):
                continue
            # Skip if it's only punctuation/slashes
            if not any(c.isalnum() for c in comp_text):
                continue
            cleaned_components.append(comp)

        if len(cleaned_components) != len(mg['components']):
            print(f"    Filtered components: {len(mg['components'])} -> {len(cleaned_components)}")
        mg['components'] = cleaned_components

        # Rebuild cleaned_text from filtered components
        if len(cleaned_components) > 0:
            parts = []
            for i, comp in enumerate(cleaned_components):
                text = comp['text']
                if i == 0:
                    parts.append(text)
                elif text == '/':
                    parts.append('/')
                elif i > 0 and cleaned_components[i-1]['text'] == '/':
                    parts.append(text)
                else:
                    parts.append(' ' + text)
            cleaned_text = ''.join(parts)
            # Remove trailing F/O notations from displayed text
            cleaned_text = cleaned_text.rstrip(' F').rstrip('F')
            cleaned_text = cleaned_text.rstrip(' O').rstrip('O')
            cleaned_text = cleaned_text.strip()

        # If it has any numbers, accept it as a measurement
        if any(c.isdigit() for c in cleaned_text):
            mg['cleaned_text'] = cleaned_text
            valid_measurements.append(mg)

    if not valid_measurements:
        # Check if Phase 1 had digits but Phase 2 doesn't - this indicates OCR misread
        phase1_had_digits = any(any(c.isdigit() for c in text) for text in texts)

        if phase1_had_digits:
            # Phase 1 detected numbers but Phase 2 verification failed
            # Return special flag for Claude verification
            print(f"  [OCR MISMATCH] Phase 1 had digits {texts} but Phase 2 found: {[mg['text'] for mg in measurement_groups]}")
            print(f"  → Flagging for Claude verification")
            logic_steps.append("Phase 1 had digits but Phase 2 verification failed")
            logic_steps.append("Needs Claude verification")
            # Return special tuple with 'NEEDS_CLAUDE_VERIFICATION' flag
            return ('NEEDS_CLAUDE_VERIFICATION', None, None, full_text, False), " | ".join(logic_steps)
        else:
            logic_steps.append(f"Position-grouped: {[mg['text'] for mg in measurement_groups]}")
            logic_steps.append("No measurements with numbers found")
            return None, " | ".join(logic_steps)

    # Check for special notations
    special_notation = None
    is_finished_size = False

    for mg in measurement_groups:
        # NH can be anywhere - check in original text
        if 'NH' in mg['text']:
            special_notation = 'NH'
            break

        # F must be AFTER first digit - check in cleaned components
        if 'components' in mg and len(mg['components']) > 0:
            # Find first component with digit
            first_digit_idx = None
            for idx, comp in enumerate(mg['components']):
                if any(c.isdigit() for c in comp['text']):
                    first_digit_idx = idx
                    break

            # Check for F only AFTER first digit
            if first_digit_idx is not None:
                for idx, comp in enumerate(mg['components']):
                    if idx > first_digit_idx and comp['text'] == 'F':
                        is_finished_size = True
                        print(f"  Found finished size indicator: F")
                        break

    # Check if multiple measurements are vertically separated (on different lines)
    # If yes, return ALL of them; if no, pick closest to avoid duplicates
    if len(valid_measurements) > 1:
        # Calculate Y distances between measurements
        y_positions = [vm['y'] for vm in valid_measurements]
        max_y_diff = max(y_positions) - min(y_positions)
        # Threshold: 30px unscaled (measurements on different lines should be ~30-40px apart in zoomed view)
        y_separation_threshold = 30 * zoom_factor

        if max_y_diff > y_separation_threshold:
            # Measurements on different lines - return ALL
            print(f"  Multiple measurements on different lines (Y-diff={max_y_diff:.0f}px > {y_separation_threshold:.0f}px)")
            print(f"  Returning ALL {len(valid_measurements)} measurements")

            # Helper function to calculate bounds for a measurement
            def calculate_bounds_for_measurement(meas):
                measurement_value = meas.get('cleaned_text', meas['text'])
                raw_ocr_text = meas.get('raw_full_text', measurement_value)

                actual_bounds = None
                if 'components' in meas and meas['components']:
                    min_x = float('inf')
                    max_x = float('-inf')
                    min_y = float('inf')
                    max_y = float('-inf')

                    comp_bboxes = []
                    for comp in meas['components']:
                        if 'vertices' in comp:
                            v_x_coords = [v.get('x', 0) for v in comp['vertices']]
                            v_y_coords = [v.get('y', 0) for v in comp['vertices']]

                            comp_min_x = min(v_x_coords)
                            comp_max_x = max(v_x_coords)
                            comp_min_y = min(v_y_coords)
                            comp_max_y = max(v_y_coords)

                            comp_bboxes.append({
                                'min_x': comp_min_x,
                                'max_x': comp_max_x,
                                'min_y': comp_min_y,
                                'max_y': comp_max_y
                            })

                            min_y = min(min_y, comp_min_y)
                            max_y = max(max_y, comp_max_y)

                    if comp_bboxes:
                        leftmost_x = min(bbox['min_x'] for bbox in comp_bboxes)
                        rightmost_x = max(bbox['max_x'] for bbox in comp_bboxes)
                        ocr_span = rightmost_x - leftmost_x
                        ocr_center = (leftmost_x + rightmost_x) / 2

                        # Use OCR span directly since components now include F/O notation
                        estimated_text_width = ocr_span

                        min_x = ocr_center - estimated_text_width / 2
                        max_x = ocr_center + estimated_text_width / 2

                        actual_bounds = {
                            'left': crop_x1 + (min_x / zoom_factor),
                            'right': crop_x1 + (max_x / zoom_factor),
                            'top': crop_y1 + (min_y / zoom_factor),
                            'bottom': crop_y1 + (max_y / zoom_factor)
                        }

                return (measurement_value, actual_bounds, special_notation, raw_ocr_text, is_finished_size)

            # Build list of all measurements
            results = []
            for vm in valid_measurements:
                result = calculate_bounds_for_measurement(vm)
                results.append(result)
                print(f"    Returning: '{result[0]}'")

            logic_steps.append(f"Found {len(valid_measurements)} measurements on different lines")
            logic_steps.append(f"Returning all {len(results)} measurements")
            return results, " | ".join(logic_steps)

    # Single measurement OR multiple on same line (pick closest to avoid duplicates)
    crop_cx = (cx - crop_x1) * zoom_factor
    crop_cy = (cy - crop_y1) * zoom_factor

    best_meas = None
    best_dist = float('inf')

    for vm in valid_measurements:
        dist = ((vm['x'] - crop_cx) ** 2 + (vm['y'] - crop_cy) ** 2) ** 0.5
        measurement_value = vm.get('cleaned_text', vm['text'])
        print(f"    Distance for '{measurement_value}': {dist:.1f}")
        if dist < best_dist:
            best_dist = dist
            best_meas = vm

    if best_meas:
        # Use cleaned text for the measurement value
        measurement_value = best_meas.get('cleaned_text', best_meas['text'])
        raw_ocr_text = best_meas.get('raw_full_text', measurement_value)
        print(f"  Chose closest: '{measurement_value}'")

        # Check if this measurement should be excluded (e.g., X2, OL, room names)
        if non_measurement_text_exclusions:
            for exc in non_measurement_text_exclusions:
                if 'text' in exc and 'x' in exc and 'y' in exc:
                    # Check if extracted text matches excluded text and position is close
                    # Convert zoomed coordinates back to original image coordinates
                    original_x = crop_x1 + (best_meas['x'] / zoom_factor)
                    original_y = crop_y1 + (best_meas['y'] / zoom_factor)
                    if (exc['text'].upper() == measurement_value.upper() and
                        abs(exc['x'] - original_x) < 50 and
                        abs(exc['y'] - original_y) < 50):
                        print(f"  Excluding: '{measurement_value}' matches non_measurement_text_exclusions")
                        return None, "Excluded by non_measurement_text_exclusions (matches room name, overlay, or label)"

        # Also check if this measurement matches overlay pattern (e.g., "3/4 01", "3/4 OL")
        if re.search(OVERLAY_PATTERN, measurement_value, re.IGNORECASE):
            print(f"  Excluding: '{measurement_value}' matches overlay pattern")
            return None, "Excluded by overlay pattern"

        # Calculate actual bounds from the components
        actual_bounds = None
        if 'components' in best_meas and best_meas['components']:
            # Find min/max bounds from all components
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')

            # IMPROVED APPROACH: Calculate bounds from component vertices to get actual character widths
            # Use OCR vertices to find actual character bounding boxes, not just center positions

            # Get component bounding boxes from vertices
            comp_bboxes = []
            comp_widths = []
            total_cleaned_text_len = 0

            for comp in best_meas['components']:
                # Use cleaned text length
                total_cleaned_text_len += len(comp.get('text', ''))

                # Calculate bounding box from vertices for both X and Y
                if 'vertices' in comp:
                    v_x_coords = [v.get('x', 0) for v in comp['vertices']]
                    v_y_coords = [v.get('y', 0) for v in comp['vertices']]

                    comp_min_x = min(v_x_coords)
                    comp_max_x = max(v_x_coords)
                    comp_min_y = min(v_y_coords)
                    comp_max_y = max(v_y_coords)

                    comp_bboxes.append({
                        'min_x': comp_min_x,
                        'max_x': comp_max_x,
                        'min_y': comp_min_y,
                        'max_y': comp_max_y,
                        'width': comp_max_x - comp_min_x,
                        'text': comp.get('text', '')
                    })
                    comp_widths.append(comp_max_x - comp_min_x)

                    min_y = min(min_y, comp_min_y)
                    max_y = max(max_y, comp_max_y)

            # Calculate bounds from actual component bounding boxes
            if comp_bboxes:
                # Find leftmost and rightmost edges from OCR vertices
                leftmost_x = min(bbox['min_x'] for bbox in comp_bboxes)
                rightmost_x = max(bbox['max_x'] for bbox in comp_bboxes)

                # PROBLEM: OCR includes extra characters (dashes, etc) making bounding boxes inconsistent
                # SOLUTION: Calculate pixels/char from OCR, then use actual cleaned character count

                # Get the OCR span and center
                ocr_span = rightmost_x - leftmost_x
                ocr_center = (leftmost_x + rightmost_x) / 2

                # Get character counts for both raw OCR and cleaned text
                char_count = len(measurement_value)  # Cleaned text (e.g., "14 7/16")
                raw_char_count = len(raw_ocr_text)   # Raw OCR text (e.g., "-14 7/16-")

                # Calculate pixels per character from OCR span and raw character count
                if raw_char_count > 0:
                    pixels_per_char = ocr_span / raw_char_count
                else:
                    pixels_per_char = 20  # Fallback

                # Calculate width based on OCR component vertices (which now include F/O notation)
                # Use the OCR span directly since components include all necessary characters
                estimated_text_width = ocr_span

                # Center the bounds around the OCR center
                min_x = ocr_center - estimated_text_width / 2
                max_x = ocr_center + estimated_text_width / 2

                print(f"    OCR span: {ocr_span:.0f}px, center={ocr_center:.0f}")
                print(f"    Raw OCR: '{raw_ocr_text}' ({raw_char_count} chars)")
                print(f"    Cleaned: '{measurement_value}' ({char_count} chars)")
                print(f"    Pixels per char: {pixels_per_char:.1f}")
                print(f"    Final bounds: [{min_x:.0f}, {max_x:.0f}] width={estimated_text_width:.0f}")

            # Scale bounds back to original image coordinates
            # The bounds are in zoomed image coords, need to unzoom and uncrop
            print(f"    Transform: min_x={min_x:.0f}, max_x={max_x:.0f}, crop_x1={crop_x1:.0f}, zoom={zoom_factor:.2f}")

            actual_bounds = {
                'left': crop_x1 + (min_x / zoom_factor),
                'right': crop_x1 + (max_x / zoom_factor),
                'top': crop_y1 + (min_y / zoom_factor),
                'bottom': crop_y1 + (max_y / zoom_factor)
            }

            calculated_width = actual_bounds['right'] - actual_bounds['left']
            calculated_height = actual_bounds['bottom'] - actual_bounds['top']
            print(f"  Actual OCR bounds: left={actual_bounds['left']:.0f}, right={actual_bounds['right']:.0f}, top={actual_bounds['top']:.0f}, bottom={actual_bounds['bottom']:.0f}")
            print(f"  Calculated dimensions: WIDTH={calculated_width:.0f}px, HEIGHT={calculated_height:.0f}px")
            print(f"  *** THIS WIDTH WILL BE USED FOR H-ROI CALCULATION ***")
        else:
            # Fallback if no components (shouldn't happen)
            actual_bounds = None

        if len(valid_measurements) > 1:
            logic_steps.append(f"Position-grouped measurements: {[vm.get('cleaned_text', vm['text']) for vm in valid_measurements]}")
            logic_steps.append(f"Chose closest to center: '{measurement_value}'")
        else:
            logic_steps.append(f"Found measurement: '{measurement_value}'")

        # Return tuple of (cleaned_text, bounds, notation, raw_text, is_finished_size)
        if special_notation:
            print(f"  Found special notation: {special_notation}")
        if is_finished_size:
            print(f"    Finished size (no overlay to be added)")
        return (measurement_value, actual_bounds, special_notation, raw_ocr_text, is_finished_size), " | ".join(logic_steps)

    return (None, None, None, None, False), "No measurements found"
