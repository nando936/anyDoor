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
import sys

sys.stdout.reconfigure(encoding='utf-8')

def verify_measurement_with_zoom(image_path, x, y, text, api_key):
    """Verify a measurement by zooming in on its location"""
    import cv2

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return text  # Return original if can't load

    h, w = image.shape[:2]

    # Create a padded crop area (3x wider than typical text)
    padding = 100  # Pixels to pad around the text location
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding/2))
    x2 = min(w, int(x + padding))
    y2 = min(h, int(y + padding/2))

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
                    # Check if it matches measurement patterns
                    if re.match(r'^\d+$', line) or \
                       re.match(r'^\d+\s+\d+/\d+', line) or \
                       re.match(r'^\d+-\d+/\d+', line) or \
                       re.match(r'^\d+/\d+/\d+$', line):
                        print(f"    Verification: '{text}' -> '{line}'")
                        return line

    return text  # Return original if verification doesn't find anything clear

def get_measurements_from_vision_api(image_path, api_key, debug_text=None):
    """Get all measurements with positions from Vision API"""
    import re

    with open(image_path, 'rb') as f:
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

    if response.status_code == 200:
        result = response.json()
        if 'responses' in result and result['responses']:
            annotations = result['responses'][0].get('textAnnotations', [])

            if annotations:
                # First annotation is full text
                full_text = annotations[0]['description']
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

                    all_text_items[text] = {'x': x, 'y': y, 'vertices': vertices}

                # Now match measurements to positions
                for measurement in measurement_patterns:
                    # Check if it's a simple whole number
                    if re.match(r'^\d+$', measurement):
                        # Whole number - should exist as-is
                        if measurement in all_text_items:
                            measurements.append({
                                'text': measurement,
                                'x': all_text_items[measurement]['x'],
                                'y': all_text_items[measurement]['y']
                            })
                            print(f"  Matched whole number '{measurement}' at ({all_text_items[measurement]['x']:.0f}, {all_text_items[measurement]['y']:.0f})")
                        continue

                    # Check if this measurement exists as single item with dash format (like "31-1/4 F")
                    dash_format = measurement.replace(" ", "-", 1)  # Convert "31 1/4 F" to "31-1/4 F"

                    # Check if this measurement exists as single item (like "5/1/2")
                    original_format = measurement.replace(" ", "/")  # Convert "5 1/2" to "5/1/2"

                    if dash_format in all_text_items:
                        # Found as dash format (like "31-1/4 F")
                        measurements.append({
                            'text': measurement,
                            'x': all_text_items[dash_format]['x'],
                            'y': all_text_items[dash_format]['y']
                        })
                    elif original_format in all_text_items:
                        # Found as single item (like "5/1/2")
                        measurements.append({
                            'text': measurement,
                            'x': all_text_items[original_format]['x'],
                            'y': all_text_items[original_format]['y']
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
                                measurements.append({
                                    'text': measurement,
                                    'x': all_text_items[dash_number]['x'],
                                    'y': all_text_items[dash_number]['y']
                                })
                                print(f"  Matched '{measurement}' via dash format at ({all_text_items[dash_number]['x']:.0f}, {all_text_items[dash_number]['y']:.0f})")
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
                                second_part_pos = (all_text_items[parts[1]]['x'], all_text_items[parts[1]]['y'])
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
                                        'width': actual_width  # Store actual width
                                    })
                                    print(f"  Matched '{measurement}' at ({avg_x:.0f}, {avg_y:.0f})")

    # Verify measurements with zoom - especially whole numbers which might be OCR errors
    print("\n  Verifying measurements with zoom...")
    verified_measurements = []
    for meas in measurements:
        # Verify suspicious measurements (like standalone numbers)
        if re.match(r'^\d+$', meas['text']) or len(meas['text']) <= 3:
            print(f"  Verifying '{meas['text']}' at ({meas['x']:.0f}, {meas['y']:.0f})...")
            verified_text = verify_measurement_with_zoom(image_path, meas['x'], meas['y'], meas['text'], api_key)
            if verified_text != meas['text']:
                if verified_text == "" or not re.match(r'\d', verified_text):
                    print(f"    [REMOVED] '{meas['text']}' - no valid measurement found at zoom")
                    continue  # Skip this measurement
                else:
                    print(f"    [CORRECTED] '{meas['text']}' -> '{verified_text}'")
                    meas['text'] = verified_text
            else:
                print(f"    [CONFIRMED] '{meas['text']}'")
        verified_measurements.append(meas)

    return verified_measurements, full_text

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

def find_lines_near_measurement(image, measurement, debug_image=None, save_edges=False):
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
    v_strip_height = text_height * 2  # Look 2x text height above/below
    v_strip_width = text_width + 10  # Width centered on text X

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

    # Debug: Save ROIs for "20 3/16"
    if "20" in str(measurement.get('text', '')):
        # Combine left and right horizontal ROIs for visualization
        h_left_roi = image[h_left_y1:h_left_y2, h_left_x1:h_left_x2]
        h_right_roi = image[h_right_y1:h_right_y2, h_right_x1:h_right_x2]

        # Combine top and bottom vertical ROIs for visualization
        v_top_roi = image[v_top_y1:v_top_y2, v_top_x1:v_top_x2]
        v_bottom_roi = image[v_bottom_y1:v_bottom_y2, v_bottom_x1:v_bottom_x2]

        # Create combined images for debug
        h_combined = np.zeros((h_left_y2-h_left_y1, h_right_x2-h_left_x1, 3), dtype=np.uint8)
        h_combined[:, :h_left_x2-h_left_x1] = h_left_roi
        h_combined[:, -(h_right_x2-h_right_x1):] = h_right_roi

        v_combined = np.zeros((v_bottom_y2-v_top_y1, v_top_x2-v_top_x1, 3), dtype=np.uint8)
        v_combined[:v_top_y2-v_top_y1, :] = v_top_roi
        v_combined[-(v_bottom_y2-v_bottom_y1):, :] = v_bottom_roi

        cv2.imwrite("debug_20_roi_horizontal.png", h_combined)
        cv2.imwrite("debug_20_roi_vertical.png", v_combined)
        print(f"    [DEBUG] Horizontal ROI: Left({h_left_x2-h_left_x1}x{h_left_y2-h_left_y1}) + Right({h_right_x2-h_right_x1}x{h_right_y2-h_right_y1})")
        print(f"    [DEBUG] Vertical ROI: Top({v_top_x2-v_top_x1}x{v_top_y2-v_top_y1}) + Bottom({v_bottom_x2-v_bottom_x1}x{v_bottom_y2-v_bottom_y1})")

    # Process horizontal and vertical ROIs separately
    horizontal_lines = []
    vertical_lines = []

    # Helper function to detect lines in an ROI
    def detect_lines_in_roi(roi_image, roi_x_offset, roi_y_offset):
        if text_color is not None:
            # Create color mask for lines matching text color
            text_bgr = np.uint8([[text_color]])
            text_hsv = cv2.cvtColor(text_bgr.reshape(1,1,3), cv2.COLOR_BGR2HSV)[0,0]

            # Convert ROI to HSV
            roi_hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

            # Create mask for similar colors
            lower = np.array([max(0, text_hsv[0] - 10), max(0, text_hsv[1] - 50), max(0, text_hsv[2] - 50)])
            upper = np.array([min(179, text_hsv[0] + 10), min(255, text_hsv[1] + 50), min(255, text_hsv[2] + 50)])
            color_mask = cv2.inRange(roi_hsv, lower, upper)

            # Apply morphology
            kernel = np.ones((3,3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            # Edge detection
            edges = cv2.Canny(color_mask, 50, 150)
        else:
            # Fallback to grayscale
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=30,
                                minLineLength=30,
                                maxLineGap=20)

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

    left_h_lines = detect_lines_in_roi(h_left_roi, h_left_x1, h_left_y1)
    right_h_lines = detect_lines_in_roi(h_right_roi, h_right_x1, h_right_y1)

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

    top_v_lines = detect_lines_in_roi(v_top_roi, v_top_x1, v_top_y1)
    bottom_v_lines = detect_lines_in_roi(v_bottom_roi, v_bottom_x1, v_bottom_y1)

    # Filter for vertical lines (angle between 70 and 110 degrees)
    for line in top_v_lines + bottom_v_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
        if 70 < angle < 110:
            vertical_lines.append({
                'coords': (lx1, ly1, lx2, ly2),
                'distance': abs(x - (lx1 + lx2) / 2),
                'type': 'vertical_line'
            })

    # Debug output for "20 3/16"
    if "20" in str(measurement.get('text', '')):
        print(f"    [DEBUG] Found {len(horizontal_lines)} horizontal lines in H-ROIs")
        print(f"    [DEBUG] Found {len(vertical_lines)} vertical lines in V-ROIs")

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

def classify_measurements_by_local_lines(image, measurements):
        lx1 += x1
        ly1 += y1
        lx2 += x1
        ly2 += y1

        # Calculate angle
        angle = np.abs(np.degrees(np.arctan2(ly2-ly1, lx2-lx1)))

        # Define text bounding box (rectangular region around text)
        text_left = x - text_width / 2
        text_right = x + text_width / 2
        text_top = y - text_height / 2
        text_bottom = y + text_height / 2

        # Categorize by orientation
        # Horizontal lines are for WIDTH measurements
        if angle < 20 or angle > 160:
            line_y = (ly1 + ly2) / 2
            y_distance = abs(y - line_y)

            # Check if line passes through the text box
            # For horizontal lines, the line's Y should be in the same range as text Y
            # This means the line should be horizontally aligned with the text
            line_in_text_box = (text_top <= line_y <= text_bottom)

            # For horizontal dimension lines (WIDTH), the text should be ALONG the line
            # The text center Y should align with the line Y (very close, within a few pixels)
            # Skip lines that are not aligned with the text center
            if abs(y - line_y) > 5:  # Allow only 5 pixel tolerance
                if "20" in str(measurement.get('text', '')):
                    print(f"      Skipping horizontal line at Y={line_y:.1f} - too far from text Y={y:.1f} (distance={abs(y-line_y):.1f} > {text_height/2:.1f})")
                continue

            # Check for arrows at line endpoints (indicates dimension line)
            # Arrow should be on the end furthest from text
            has_arrow = False
            # Convert back to ROI coordinates for arrow check
            roi_lx1 = int(lx1 - x1)
            roi_ly1 = int(ly1 - y1)
            roi_lx2 = int(lx2 - x1)
            roi_ly2 = int(ly2 - y1)

            # Determine which endpoint is furthest from text (in ROI coordinates)
            text_roi_x = x - x1
            text_roi_y = y - y1
            dist1 = ((roi_lx1 - text_roi_x)**2 + (roi_ly1 - text_roi_y)**2) ** 0.5
            dist2 = ((roi_lx2 - text_roi_x)**2 + (roi_ly2 - text_roi_y)**2) ** 0.5

            # Check the furthest endpoint for arrow
            if dist1 > dist2:
                has_arrow = check_for_arrow_at_endpoint(edges, roi_lx1, roi_ly1)
            else:
                has_arrow = check_for_arrow_at_endpoint(edges, roi_lx2, roi_ly2)

            # For horizontal lines (WIDTH measurements), only accept if:
            # The line Y is VERY close to text Y center (text sits ON the line)
            # This filters out horizontal lines that are near but not aligned
            if abs(y - line_y) <= 2 and y_distance <= text_height_with_padding:
                horizontal_lines.append({
                    'coords': (lx1, ly1, lx2, ly2),
                    'distance': y_distance,
                    'type': 'horizontal_line',
                    'has_arrow': has_arrow
                })
                if "33" in str(measurement.get('text', '')) or "20" in str(measurement.get('text', '')):
                    line_length = ((lx2 - lx1)**2 + (ly2 - ly1)**2) ** 0.5
                    arrow_str = " (with arrow)" if has_arrow else ""
                    # Sample color at line midpoint
                    line_mid_x = int((lx1 + lx2) / 2)
                    line_mid_y = int(line_y)
                    if 0 <= line_mid_y < image.shape[0] and 0 <= line_mid_x < image.shape[1]:
                        line_color = image[line_mid_y, line_mid_x]
                        print(f"      Found horizontal line: Y={line_y:.1f}, distance={y_distance:.1f}, length={line_length:.1f}, color BGR={line_color}{arrow_str}")
                    else:
                        print(f"      Found horizontal line: Y={line_y:.1f}, distance={y_distance:.1f}, length={line_length:.1f}{arrow_str}")

                    # Draw debug circles for "20 3/16" lines
                    if debug_image is not None and "20" in str(measurement.get('text', '')):
                        # Draw the line in blue (BGR format)
                        cv2.line(debug_image, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 0, 0), 2)
                        # Draw circles at endpoints
                        cv2.circle(debug_image, (int(lx1), int(ly1)), 5, (255, 0, 0), -1)
                        cv2.circle(debug_image, (int(lx2), int(ly2)), 5, (255, 0, 0), -1)
                        # Add detailed label with Y position
                        cv2.putText(debug_image, f"Y={line_y:.0f}", (int((lx1+lx2)/2), int(line_y-10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        cv2.putText(debug_image, f"dist={y_distance:.0f}px", (int((lx1+lx2)/2), int(line_y+10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            elif "20" in str(measurement.get('text', '')) and line_in_text_box and not has_arrow:
                print(f"      Skipped horizontal line at Y={line_y:.1f} - inside text box, no arrow")

        # Vertical lines are for HEIGHT measurements
        elif 70 < angle < 110:
            line_x = (lx1 + lx2) / 2
            x_distance = abs(x - line_x)

            # Check if line passes through the text box
            # For vertical lines, the line's X should be in the same range as text X
            # This means the line should be vertically aligned with the text
            line_in_text_box = (text_left <= line_x <= text_right)

            # For vertical dimension lines (HEIGHT), the text should be ALONG the line
            # The text center X should align with the line X (very close, within a few pixels)
            # Skip lines that are not aligned with the text center
            if abs(x - line_x) > 10:  # Allow 10 pixel tolerance for vertical (text might be slightly offset)
                if "20" in str(measurement.get('text', '')):
                    print(f"      Skipping vertical line at X={line_x:.1f} - too far from text X={x:.1f} (distance={abs(x-line_x):.1f} > {text_width/2:.1f})")
                continue

            # Check for arrows at line endpoints (indicates dimension line)
            # Arrow should be on the end furthest from text
            has_arrow = False
            # Convert back to ROI coordinates for arrow check
            roi_lx1 = int(lx1 - x1)
            roi_ly1 = int(ly1 - y1)
            roi_lx2 = int(lx2 - x1)
            roi_ly2 = int(ly2 - y1)

            # Determine which endpoint is furthest from text (in ROI coordinates)
            text_roi_x = x - x1
            text_roi_y = y - y1
            dist1 = ((roi_lx1 - text_roi_x)**2 + (roi_ly1 - text_roi_y)**2) ** 0.5
            dist2 = ((roi_lx2 - text_roi_x)**2 + (roi_ly2 - text_roi_y)**2) ** 0.5

            # Check the furthest endpoint for arrow
            if dist1 > dist2:
                has_arrow = check_for_arrow_at_endpoint(edges, roi_lx1, roi_ly1)
            else:
                has_arrow = check_for_arrow_at_endpoint(edges, roi_lx2, roi_ly2)

            # For vertical lines (HEIGHT measurements), accept if:
            # 1. Line is very close to text X (text sits ON the line), OR
            # 2. Line is outside text box but within padding distance
            # This filters out unrelated vertical lines passing through
            if (abs(x - line_x) <= 8 or not line_in_text_box) and x_distance <= text_width_with_padding:
                vertical_lines.append({
                    'coords': (lx1, ly1, lx2, ly2),
                    'distance': x_distance,
                    'type': 'vertical_line',
                    'has_arrow': has_arrow
                })
                if "33" in str(measurement.get('text', '')) or "20" in str(measurement.get('text', '')):
                    line_length = ((lx2 - lx1)**2 + (ly2 - ly1)**2) ** 0.5
                    arrow_str = " (with arrow)" if has_arrow else ""
                    # Sample color at line midpoint
                    line_mid_x = int(line_x)
                    line_mid_y = int((ly1 + ly2) / 2)
                    if 0 <= line_mid_y < image.shape[0] and 0 <= line_mid_x < image.shape[1]:
                        line_color = image[line_mid_y, line_mid_x]
                        print(f"      Found vertical line: X={line_x:.1f}, distance={x_distance:.1f}, length={line_length:.1f}, color BGR={line_color}{arrow_str}")
                    else:
                        print(f"      Found vertical line: X={line_x:.1f}, distance={x_distance:.1f}, length={line_length:.1f}{arrow_str}")

                    # Draw debug circles for "20 3/16" lines
                    if debug_image is not None and "20" in str(measurement.get('text', '')):
                        # Draw the line in red (BGR format)
                        cv2.line(debug_image, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 0, 255), 2)
                        # Draw circles at endpoints
                        cv2.circle(debug_image, (int(lx1), int(ly1)), 5, (0, 0, 255), -1)
                        cv2.circle(debug_image, (int(lx2), int(ly2)), 5, (0, 0, 255), -1)
                        # Add detailed label with X position
                        cv2.putText(debug_image, f"X={line_x:.0f}", (int(line_x+5), int((ly1+ly2)/2-10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        cv2.putText(debug_image, f"dist={x_distance:.0f}px", (int(line_x+5), int((ly1+ly2)/2+10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            elif "20" in str(measurement.get('text', '')) and line_in_text_box and not has_arrow:
                print(f"      Skipped vertical line at X={line_x:.1f} - inside text box, no arrow")

    # Choose the closest line of each type
    # For HEIGHT measurements: vertical lines are closer to the text
    # For WIDTH measurements: horizontal lines are closer to the text
    best_line = None
    min_distance = float('inf')
    orientation = None

    # Get closest of each type
    closest_h = min(horizontal_lines, key=lambda l: l['distance']) if horizontal_lines else None
    closest_v = min(vertical_lines, key=lambda l: l['distance']) if vertical_lines else None

    # Debug output
    if "20" in str(measurement.get('text', '')):
        print(f"    DEBUG: After filtering - Found {len(horizontal_lines)} horiz lines, {len(vertical_lines)} vert lines")
        if closest_h:
            print(f"    Closest horiz at distance: {closest_h['distance']:.1f}")
        if closest_v:
            print(f"    Closest vert at distance: {closest_v['distance']:.1f}")

    # If we have both, choose based on which is significantly closer
    if closest_h and closest_v:
        # If vertical line is much closer, it's a height
        if closest_v['distance'] < closest_h['distance'] - 10:
            min_distance = closest_v['distance']
            best_line = closest_v['coords']
            orientation = 'vertical_line'  # Vertical line = HEIGHT measurement
        # If horizontal line is much closer, it's a width
        elif closest_h['distance'] < closest_v['distance'] - 10:
            min_distance = closest_h['distance']
            best_line = closest_h['coords']
            orientation = 'horizontal_line'  # Horizontal line = WIDTH measurement
        else:
            # Similar distances, use the closer one
            if closest_v['distance'] < closest_h['distance']:
                min_distance = closest_v['distance']
                best_line = closest_v['coords']
                orientation = 'vertical_line'
            else:
                min_distance = closest_h['distance']
                best_line = closest_h['coords']
                orientation = 'horizontal_line'
    elif closest_h:
        min_distance = closest_h['distance']
        best_line = closest_h['coords']
        orientation = 'horizontal_line'
    elif closest_v:
        min_distance = closest_v['distance']
        best_line = closest_v['coords']
        orientation = 'vertical_line'

    if best_line and orientation:
        return {
            'line': best_line,
            'orientation': orientation,
            'distance': min_distance
        }

    return None

def classify_measurements_by_local_lines(image, measurements):
    """Classify each measurement based on lines found near it"""
    classified = {
        'vertical': [],
        'horizontal': [],
        'unclassified': []
    }

    measurement_details = []

    # Create debug image for "20 3/16"
    debug_image = image.copy()

    for meas in measurements:
        print(f"\n  Analyzing {meas['text']} at ({meas['x']:.0f}, {meas['y']:.0f})")

        # Find lines near this measurement (pass debug_image for "20 3/16")
        if "20" in str(meas['text']):
            line_info = find_lines_near_measurement(image, meas, debug_image)

            # Draw text bounding box in yellow
            text_width = len(meas['text']) * 15
            text_height = 30
            x, y = int(meas['x']), int(meas['y'])
            text_left = x - text_width // 2
            text_right = x + text_width // 2
            text_top = y - text_height // 2
            text_bottom = y + text_height // 2

            # Draw the text bounding box
            cv2.rectangle(debug_image, (text_left, text_top), (text_right, text_bottom), (0, 255, 255), 2)

            # Mark text center with green cross
            cv2.line(debug_image, (x-30, y), (x+30, y), (0, 255, 0), 2)  # Horizontal line through center
            cv2.line(debug_image, (x, y-30), (x, y+30), (0, 255, 0), 2)  # Vertical line through center

            # Add labels
            cv2.putText(debug_image, f"Text: 20 3/16", (x+40, y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Center: ({x},{y})", (x+40, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(debug_image, f"Box: Y{text_top}-{text_bottom}", (x+40, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            line_info = find_lines_near_measurement(image, meas)

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

    # Save debug image for "20 3/16" lines
    cv2.imwrite("debug_20_lines.png", debug_image)
    print("\n[DEBUG] Saved debug_20_lines.png showing all lines found near '20 3/16'")
    print("  - Blue lines = Horizontal lines found")
    print("  - Red lines = Vertical lines found")

    return classified, measurement_details

def main():
    import os
    print("=" * 80)
    print("MEASUREMENT-BASED DIMENSION DETECTOR")
    print("Finds measurements first, then looks for lines near them")
    print("=" * 80)
    print()

    API_KEY = "AIzaSyBMvrdMTEh9buLqKjmr5zoUONTcj3YcSsA"

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
    measurements, full_text = get_measurements_from_vision_api(image_path, API_KEY)
    print(f"\n  Found {len(measurements)} measurements:")
    for m in measurements:
        print(f"    {m['text']} at ({m['x']:.0f}, {m['y']:.0f})")

    # Step 2: Find lines near each measurement and classify
    print("\n[2/2] Finding lines near each measurement...")
    classified, details = classify_measurements_by_local_lines(image, measurements)

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

    total = len(classified['vertical']) * len(classified['horizontal'])
    print(f"\nTOTAL OPENINGS: {total}")

    if total > 0:
        print("\nOPENING SPECIFICATIONS:")
        n = 1
        for h in classified['horizontal']:  # Width
            for v in classified['vertical']:  # Height
                print(f"  Opening {n}: {h} W × {v} H")
                n += 1

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

    # Look for overlay/hinge notations in the full text
    overlay_info = ""
    if full_text:
        # Look for common overlay patterns
        overlay_patterns = [
            r'(\d+/\d+)\s+OL',  # Like "5/8 OL" or "1/2 OL"
            r'(\d+/\d+)\s+OL\s+\w+',  # Like "5/8 OL KITCHEN"
            r'FULL\s+OL',  # Full overlay
            r'PARTIAL\s+OL',  # Partial overlay
        ]
        for pattern in overlay_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                overlay_info = match.group(0)
                print(f"\n[OK] Found overlay info: {overlay_info}")
                break

    # Save results
    results = {
        'measurements': [{'text': m['text'], 'position': [m['x'], m['y']]} for m in measurements],
        'vertical': classified['vertical'],
        'horizontal': classified['horizontal'],
        'unclassified': classified['unclassified'],
        'total_openings': total,
        'overlay_info': overlay_info
    }

    output_json = os.path.join(image_dir, f"{base_name}_measurements_data.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved: {output_json}")

if __name__ == "__main__":
    main()