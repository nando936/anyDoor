#!/usr/bin/env python3
"""
Test detector with new algorithm:
1. Find interest areas (centers) from initial OCR
2. Merge close centers to prevent duplicates
3. One zoom verification per unique center
4. Pick closest measurement to center if multiple found
"""

import cv2
import numpy as np
import base64
import requests
import json
import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv

# Fix Unicode issues on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
# All configurable parameters in one place

# HSV color range for green text on brown backgrounds
HSV_CONFIG = {
    'lower_green': [40, 40, 40],
    'upper_green': [80, 255, 255]
}

# Text grouping proximity thresholds
GROUPING_CONFIG = {
    'x_distance': 100,   # Max horizontal distance for grouping (increased to 100)
    'y_distance': 25,   # Max vertical distance for grouping (reduced to prevent vertical over-grouping)
    'merge_threshold': 70  # Distance threshold for merging centers
}

# Zoom verification parameters
ZOOM_CONFIG = {
    'padding': 30,           # Vertical padding (top/bottom)
    'padding_horizontal': 50, # Horizontal padding (left/right) - wider to capture full characters
    'zoom_factor': 3         # Magnification factor
}

# Measurement validation
VALIDATION_CONFIG = {
    'min_value': 2,     # Minimum valid measurement value
    'max_value': 100    # Maximum valid measurement value
}

# Room name patterns to exclude
ROOM_PATTERNS = [
    r'(MASTER\s+BATH)',
    r'(MASTER\s+CLOSET)',
    r'(GUEST\s+BATH)',
    r'(POWDER\s+ROOM)',
    r'(POWDER)',
    r'(UTILITY)',
    r'(KITCHEN)',
    r'(LAUNDRY)',
    r'(PANTRY)',
    r'((?:UPSTAIRS|DOWNSTAIRS|MAIN\s+FLOOR)\s+BATH\s*\d*)',  # Location prefix + BATH
    r'((?:UPSTAIRS|DOWNSTAIRS|MAIN\s+FLOOR)\s+BEDROOM\s*\d*)',  # Location prefix + BEDROOM
    r'((?:UPSTAIRS|DOWNSTAIRS|MAIN\s+FLOOR)\s+CLOSET\s*\d*)',  # Location prefix + CLOSET
    r'(BATH\s*\d*)',
    r'(BEDROOM\s*\d*)',
    r'(CLOSET\s*\d*)',
    r'(UPSTAIRS\s+BATH)',
    r'(DOWNSTAIRS\s+BATH)',
]

# Non-measurement text to exclude
EXCLUDE_PATTERNS = ['H2', 'NH', 'C', 'UPPERS', 'BASE']

# Overlay notation pattern
OVERLAY_PATTERN = r'(\d+/\d+\s+OL)'

# ==================== END CONFIGURATION ====================

def apply_hsv_preprocessing(image):
    """Apply HSV preprocessing to isolate green text on brown backgrounds"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array(HSV_CONFIG['lower_green'])
    upper_green = np.array(HSV_CONFIG['upper_green'])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    return mask_inv

def find_room_and_overlay(annotations):
    """Extract room name and overlay info from annotations"""
    room_name = ""
    overlay_info = ""

    if not annotations:
        return room_name, overlay_info, []

    # Get full text
    full_text = annotations[0].get('description', '') if annotations else ''

    # Items to exclude (their text and positions)
    exclude_items = []

    if full_text:
        # Search for overlay notation
        overlay_match = re.search(OVERLAY_PATTERN, full_text, re.IGNORECASE)
        if overlay_match:
            overlay_info = overlay_match.group(1).strip()
            print(f"Found overlay: {overlay_info}")

            # Find overlay in individual annotations to exclude
            for ann in annotations[1:]:
                text = ann['description']
                if 'OL' in text.upper() or text in overlay_info:
                    vertices = ann['boundingPoly']['vertices']
                    x = sum(v.get('x', 0) for v in vertices) / 4
                    y = sum(v.get('y', 0) for v in vertices) / 4
                    exclude_items.append({'text': text, 'x': x, 'y': y})

        # Look for room names using configured patterns
        for pattern in ROOM_PATTERNS:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                room_name = match.group(1)
                print(f"Found room: {room_name}")

                # Find room name parts in annotations to exclude
                # Include ALL parts of room name, including numbers
                room_words = room_name.split()  # Include room numbers too
                for ann in annotations[1:]:
                    text = ann['description']
                    # Exclude if it's an exact match to any room word (including numbers)
                    if text.upper() in [w.upper() for w in room_words]:
                        vertices = ann['boundingPoly']['vertices']
                        x = sum(v.get('x', 0) for v in vertices) / 4
                        y = sum(v.get('y', 0) for v in vertices) / 4
                        exclude_items.append({'text': text, 'x': x, 'y': y})
                break

        # Also exclude common non-measurement text
        for ann in annotations[1:]:
            text = ann['description'].upper()
            if text in EXCLUDE_PATTERNS:
                vertices = ann['boundingPoly']['vertices']
                x = sum(v.get('x', 0) for v in vertices) / 4
                y = sum(v.get('y', 0) for v in vertices) / 4
                exclude_items.append({'text': ann['description'], 'x': x, 'y': y})

    return room_name, overlay_info, exclude_items

def find_opencv_supplemental_regions_better(hsv_image, all_vision_detections):
    """Find text regions using better text detection methods - not lines!"""

    # Apply threshold to get binary image
    _, binary = cv2.threshold(hsv_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological gradient to enhance text edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    additional_regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Basic size filter for text
        if not (8 < w < 150 and 8 < h < 60):
            continue

        # Aspect ratio for text
        aspect_ratio = w / h
        if not (0.2 < aspect_ratio < 8):
            continue

        # Skip very small areas
        if area < 80:
            continue

        center_x = x + w//2
        center_y = y + h//2

        # Check if overlaps with Vision API detections
        is_covered = False
        for vision_item in all_vision_detections:
            if not (x + w < vision_item['x_min'] or
                    x > vision_item['x_max'] or
                    y + h < vision_item['y_min'] or
                    y > vision_item['y_max']):
                is_covered = True
                break

        if not is_covered:
            # Extract the region
            roi = binary[y:y+h, x:x+w]

            # Text validation using stroke analysis
            # Apply distance transform to get stroke widths
            dist_transform = cv2.distanceTransform(roi, cv2.DIST_L2, 3)

            # Get stroke width statistics
            if dist_transform.size > 0:
                non_zero = dist_transform[dist_transform > 0]
                if len(non_zero) > 0:
                    mean_stroke = np.mean(non_zero)
                    std_stroke = np.std(non_zero)

                    # Text has consistent stroke width (low std/mean ratio)
                    # Lines have very uniform stroke (very low ratio) or very high ratio
                    stroke_consistency = std_stroke / (mean_stroke + 0.001)

                    # Text typically has stroke consistency between 0.3 and 0.8
                    if not (0.25 < stroke_consistency < 0.85):
                        continue

                    # Text strokes are typically 2-15 pixels thick
                    if not (1.5 < mean_stroke < 15):
                        continue

            # Check edge density - text has moderate edge density
            edges = cv2.Canny(hsv_image[y:y+h, x:x+w], 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)

            # Text has moderate edge density (not too low like solid areas, not too high like noise)
            if not (0.05 < edge_density < 0.4):
                continue

            # Count connected components in the region
            num_labels, _ = cv2.connectedComponents(roi, connectivity=8)

            # Text regions typically have 2-20 components (letters/numbers)
            # Single component = likely a line or solid shape
            # Too many = likely noise
            if not (2 <= num_labels <= 20):
                continue

            # Passed all checks - likely text!
            additional_regions.append({
                'text': 'OPENCV',
                'center': (center_x, center_y),
                'x': center_x,
                'y': center_y,
                'source': 'opencv'
            })

    return additional_regions

def find_opencv_supplemental_regions(hsv_image, all_vision_detections):
    """Find additional text regions using OpenCV in areas Vision API missed

    Focus: Detect text regions while excluding lines using multiple validation techniques
    """

    # Apply threshold to get binary image
    _, binary = cv2.threshold(hsv_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    additional_regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Basic size filter for text (measurements are typically this size)
        if not (10 < w < 120 and 10 < h < 45):
            continue

        # Skip if area too small
        if area < 100:
            continue

        center_x = x + w//2
        center_y = y + h//2

        # Check if overlaps with Vision API detections or their zoom areas
        is_covered = False
        # Use the same padding that will be used for zoom ROI areas
        zoom_padding_h = 50  # From ZOOM_CONFIG['padding_horizontal']
        zoom_padding_v = 30  # From ZOOM_CONFIG['padding']

        for vision_item in all_vision_detections:
            # Expand Vision API bounds by zoom padding to exclude zoom ROI areas
            vision_left = vision_item['x_min'] - zoom_padding_h
            vision_right = vision_item['x_max'] + zoom_padding_h
            vision_top = vision_item['y_min'] - zoom_padding_v
            vision_bottom = vision_item['y_max'] + zoom_padding_v

            # Check if OpenCV region overlaps with expanded Vision area (includes zoom ROI)
            if not (x + w < vision_left or
                    x > vision_right or
                    y + h < vision_top or
                    y > vision_bottom):
                is_covered = True
                break

        if is_covered:
            continue

        # Extract region for analysis
        roi = binary[y:y+h, x:x+w]

        # === ADVANCED LINE DETECTION ===

        # 1. Hough Line Detection - if we can fit a single strong line through most pixels, it's a line
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=min(w, h)//2,
                                minLineLength=min(w, h)*0.7, maxLineGap=5)

        if lines is not None and len(lines) > 0:
            # Check if a single line covers most of the region
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                diagonal = np.sqrt(w**2 + h**2)

                # If a single line spans most of the diagonal, it's likely just a line
                if line_length > diagonal * 0.75:
                    continue  # Skip this region, it's a line

        # 2. Profile Analysis - check intensity profile perpendicular to main axis
        aspect_ratio = w / h

        if aspect_ratio > 2.5:  # Horizontal orientation
            # Check vertical profile - lines have uniform profile
            vertical_profile = np.mean(roi, axis=1)
            profile_std = np.std(vertical_profile)

            # Lines have very low standard deviation in perpendicular profile
            if profile_std < 10:
                continue  # It's a horizontal line

        elif aspect_ratio < 0.4:  # Vertical orientation
            # Check horizontal profile
            horizontal_profile = np.mean(roi, axis=0)
            profile_std = np.std(horizontal_profile)

            if profile_std < 10:
                continue  # It's a vertical line

        # 3. Skeleton Analysis - lines have simple skeletons
        skeleton = cv2.ximgproc.thinning(roi)

        # Count skeleton endpoints and junctions
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
        filtered = cv2.filter2D(skeleton, cv2.CV_8U, kernel)

        # Endpoints have value 11 (1 neighbor + 10 center)
        # Junctions have value > 12 (2+ neighbors + 10 center)
        endpoints = np.sum(filtered == 11)
        junctions = np.sum(filtered > 12)

        # Lines have exactly 2 endpoints and 0 junctions
        # Text has multiple endpoints and junctions
        if endpoints <= 2 and junctions == 0:
            # Simple structure - likely a line
            skeleton_points = np.sum(skeleton > 0)
            if skeleton_points > 0:
                # Check if skeleton is mostly straight
                skeleton_coords = np.column_stack(np.where(skeleton > 0))
                if len(skeleton_coords) > 2:
                    # Fit a line to skeleton points
                    vx, vy, cx, cy = cv2.fitLine(skeleton_coords, cv2.DIST_L2, 0, 0.01, 0.01)

                    # Calculate how well points fit the line
                    distances = []
                    for point in skeleton_coords:
                        # Distance from point to line
                        d = abs((point[1] - cx) * vy[0] - (point[0] - cy) * vx[0]) / np.sqrt(vx[0]**2 + vy[0]**2)
                        distances.append(d)

                    mean_dist = np.mean(distances)

                    # If skeleton points are very close to a straight line, it's a line
                    if mean_dist < 2:
                        continue

        # 4. Contour Complexity - text contours are more complex than line contours
        perimeter = cv2.arcLength(contour, True)
        complexity = perimeter**2 / (4 * np.pi * area) if area > 0 else 0

        # Lines have low complexity (close to 1 for perfect rectangle)
        if complexity < 1.5:
            continue

        # 5. Connected Components Analysis
        num_labels, labels = cv2.connectedComponents(roi, connectivity=8)

        # Single component often means a line or solid shape
        if num_labels <= 2:  # Only background and 1 component
            # But check if it could be a single digit like "5"
            # Single digits are small and have moderate complexity
            if w > 30 or h > 30:  # Too large for single digit
                continue

        # 6. Pixel Distribution - text has more varied distribution than lines
        # Calculate histogram of row and column sums
        row_sums = np.sum(roi, axis=1)
        col_sums = np.sum(roi, axis=0)

        # Coefficient of variation (CV) - std/mean
        # Lines have low CV, text has higher CV
        if len(row_sums) > 0 and np.mean(row_sums) > 0:
            row_cv = np.std(row_sums) / np.mean(row_sums)
        else:
            row_cv = 0

        if len(col_sums) > 0 and np.mean(col_sums) > 0:
            col_cv = np.std(col_sums) / np.mean(col_sums)
        else:
            col_cv = 0

        # Lines have very low CV in perpendicular direction
        if aspect_ratio > 2 and row_cv < 0.2:  # Horizontal line
            continue
        if aspect_ratio < 0.5 and col_cv < 0.2:  # Vertical line
            continue

        # Passed all line detection checks - this is likely text!
        additional_regions.append({
            'text': 'OPENCV',
            'center': (center_x, center_y),
            'x': center_x,
            'y': center_y,
            'source': 'opencv'
        })

    return additional_regions

def find_interest_areas(image_path, api_key):
    """Phase 1: Find areas of interest using initial OCR"""
    print("\n=== PHASE 1: Finding Interest Areas ===")


    # Convert Windows network paths to Unix-style for OpenCV compatibility
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    # Load and preprocess image
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Could not load image from: {image_path}")
        return [], "", "", []

    # Use HSV preprocessing
    use_hsv = True  # Changed back to True for better detection

    if use_hsv:
        processed_image = apply_hsv_preprocessing(image)
        print("Using HSV preprocessing")
    else:
        processed_image = image
        print("Using original image (no HSV preprocessing)")

    # Encode for Vision API
    _, buffer = cv2.imencode('.png', processed_image)
    content = base64.b64encode(buffer).decode('utf-8')

    # Run initial OCR
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
        return [], "", ""

    result = response.json()
    if 'responses' not in result or not result['responses']:
        return [], "", ""

    annotations = result['responses'][0].get('textAnnotations', [])
    if not annotations:
        return [], "", ""

    # Extract room and overlay info first
    room_name, overlay_info, exclude_items = find_room_and_overlay(annotations)

    if room_name:
        print(f"Room name: {room_name}")
    if overlay_info:
        print(f"Overlay info: {overlay_info}")
    if exclude_items:
        print(f"Excluding {len(exclude_items)} non-measurement items")

    # Collect ALL Vision API detections with bounding boxes for OpenCV to exclude
    all_vision_detections = []
    for ann in annotations[1:]:  # Skip full text annotation
        vertices = ann['boundingPoly']['vertices']
        x_coords = [v.get('x', 0) for v in vertices]
        y_coords = [v.get('y', 0) for v in vertices]

        all_vision_detections.append({
            'text': ann['description'],
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords),
            'center_x': sum(x_coords) / 4,
            'center_y': sum(y_coords) / 4
        })

    # Get all text items with numbers (but exclude room/overlay items)
    text_items = []
    for ann in annotations[1:]:  # Skip full text
        text = ann['description']
        original_text = text
        # Clean text by removing leading/trailing hyphens
        cleaned_text = text.strip('-')

        # Only process if cleaned text still contains digits
        if cleaned_text and any(char.isdigit() for char in cleaned_text):
            vertices = ann['boundingPoly']['vertices']

            # Adjust bounding box if we removed characters
            if original_text != cleaned_text:
                # Calculate how much we removed
                removed_left = len(original_text) - len(original_text.lstrip('-'))
                removed_right = len(original_text) - len(original_text.rstrip('-'))
                total_chars = len(original_text)

                # Get original bounds
                x_coords = [v.get('x', 0) for v in vertices]
                y_coords = [v.get('y', 0) for v in vertices]
                left = min(x_coords)
                right = max(x_coords)
                top = min(y_coords)
                bottom = max(y_coords)
                width = right - left

                # Adjust bounds based on removed characters
                if removed_left > 0:
                    # Removed from left, shift left boundary right
                    char_width = width / total_chars
                    left += char_width * removed_left

                if removed_right > 0:
                    # Removed from right, shift right boundary left
                    char_width = width / total_chars
                    right -= char_width * removed_right

                # Create adjusted vertices
                adjusted_vertices = [
                    {'x': left, 'y': top},      # Top-left
                    {'x': right, 'y': top},     # Top-right
                    {'x': right, 'y': bottom},  # Bottom-right
                    {'x': left, 'y': bottom}    # Bottom-left
                ]

                # Overwrite original vertices with adjusted ones
                vertices = adjusted_vertices

                # Recalculate center with adjusted bounds
                x = (left + right) / 2
                y = (top + bottom) / 2
            else:
                # No adjustment needed, use original center
                x = sum(v.get('x', 0) for v in vertices) / 4
                y = sum(v.get('y', 0) for v in vertices) / 4

            # Check if this item should be excluded
            should_exclude = False
            for exc in exclude_items:
                if (exc['text'] == original_text and
                    abs(exc['x'] - x) < 10 and
                    abs(exc['y'] - y) < 10):
                    should_exclude = True
                    # Debug exclusion
                    print(f"  Excluding: '{original_text}' at ({x:.0f}, {y:.0f})")
                    break

            if not should_exclude:
                text_items.append({
                    'text': cleaned_text,  # Use cleaned text
                    'x': x,
                    'y': y,
                    'vertices': vertices
                })

    print(f"Found {len(text_items)} text items with numbers (after filtering)")

    # Debug: show text items
    for item in text_items:
        print(f"  Item: '{item['text']}' at ({item['x']:.0f}, {item['y']:.0f})")

    # === ADD OPENCV SUPPLEMENTATION ===
    print("\nSupplemental OpenCV detection...")

    # Find additional regions with OpenCV - pass ALL Vision detections to exclude them
    opencv_regions = find_opencv_supplemental_regions(processed_image, all_vision_detections)

    # Create visualization
    if len(processed_image.shape) == 2:
        viz = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    else:
        viz = processed_image.copy()

    # Draw Vision API detections in GREEN
    for item in text_items:
        x, y = int(item['x']), int(item['y'])
        cv2.circle(viz, (x, y), 8, (0, 255, 0), 2)
        cv2.putText(viz, item['text'][:10], (x-30, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if opencv_regions:
        print(f"  OpenCV found {len(opencv_regions)} additional potential text regions")

        # Draw OpenCV additions in RED
        for region in opencv_regions:
            cx, cy = region['center']
            cv2.circle(viz, (int(cx), int(cy)), 8, (0, 0, 255), 2)
            cv2.putText(viz, "OPENCV+", (int(cx)-30, int(cy)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Check for left 23 area
            if 250 < cx < 400 and 700 < cy < 850:
                print(f"    - Potential left '23' at ({cx}, {cy})")
                # Add special marker for potential left 23
                cv2.rectangle(viz, (int(cx)-20, int(cy)-20),
                            (int(cx)+20, int(cy)+20), (255, 0, 255), 2)

        # Add OpenCV regions to text_items
        for region in opencv_regions:
            text_items.append({
                'text': region['text'],
                'x': region['center'][0],
                'y': region['center'][1],
                'vertices': None,  # No vertices for OpenCV regions
                'source': 'opencv'  # Mark source
            })

        print(f"  Total text items after OpenCV: {len(text_items)}")
    else:
        print("  No additional regions found by OpenCV")

    # Don't save separate visualization - it will be included in main test_viz

    # Group nearby items (simple proximity grouping)
    groups = []
    used = set()

    for i, item in enumerate(text_items):
        if i in used:
            continue

        group = [item]
        used.add(i)

        # Find nearby items - check against ALL items already in group
        for j, other in enumerate(text_items):
            if j in used:
                continue

            # Check proximity to ANY item in the current group
            is_close = False
            for group_item in group:
                x_dist = abs(other['x'] - group_item['x'])
                y_dist = abs(other['y'] - group_item['y'])

                if x_dist < GROUPING_CONFIG['x_distance'] and y_dist < GROUPING_CONFIG['y_distance']:
                    is_close = True
                    break

            if is_close:
                group.append(other)
                used.add(j)

        groups.append(group)

    print(f"Formed {len(groups)} initial groups")

    # Extract interest areas (centers and bounds)
    interest_areas = []
    for group in groups:
        # Calculate bounds
        all_x = []
        all_y = []
        texts = []

        for item in group:
            vertices = item['vertices']
            if vertices:  # Check if vertices exist
                for v in vertices:
                    all_x.append(v.get('x', item['x']))
                    all_y.append(v.get('y', item['y']))
            else:  # For OpenCV regions without vertices
                all_x.extend([item['x'] - 10, item['x'] + 10])  # Approximate bounds
                all_y.extend([item['y'] - 10, item['y'] + 10])
            texts.append(item['text'])

        bounds = {
            'left': min(all_x),
            'right': max(all_x),
            'top': min(all_y),
            'bottom': max(all_y)
        }

        center_x = (bounds['left'] + bounds['right']) / 2
        center_y = (bounds['top'] + bounds['bottom']) / 2

        interest_areas.append({
            'center': (center_x, center_y),
            'bounds': bounds,
            'texts': texts
        })

    return interest_areas, room_name, overlay_info, opencv_regions  # Return opencv_regions too

def merge_close_centers(interest_areas, threshold=None):
    """Merge centers that are too close together"""
    if threshold is None:
        threshold = GROUPING_CONFIG['merge_threshold']
    print(f"\n=== Merging Close Centers (threshold={threshold}px) ===")
    print(f"Starting with {len(interest_areas)} areas")

    # Debug: show centers before merging
    for i, area in enumerate(interest_areas):
        print(f"  Area {i+1}: center ({area['center'][0]:.0f}, {area['center'][1]:.0f})")

    merged = []
    used = set()

    for i, area in enumerate(interest_areas):
        if i in used:
            continue

        # Start a new merged area
        merged_area = {
            'centers': [area['center']],
            'all_bounds': [area['bounds']],
            'all_texts': area['texts'].copy()
        }
        used.add(i)

        # Find nearby areas to merge
        for j, other in enumerate(interest_areas):
            if j in used:
                continue

            # Check distance between centers - use tighter Y threshold since text is horizontal
            x_dist = abs(area['center'][0] - other['center'][0])
            y_dist = abs(area['center'][1] - other['center'][1])

            # Use different thresholds for X and Y
            # Y should be tight (max 30px) since text on same line should have similar Y
            # X can be more relaxed (use full threshold)
            if x_dist < threshold and y_dist < 30:
                merged_area['centers'].append(other['center'])
                merged_area['all_bounds'].append(other['bounds'])
                merged_area['all_texts'].extend(other['texts'])
                used.add(j)
                print(f"  Merging areas at {area['center']} and {other['center']} (x_dist={x_dist:.1f}, y_dist={y_dist:.1f})")

        # Calculate merged center and bounds
        avg_x = sum(c[0] for c in merged_area['centers']) / len(merged_area['centers'])
        avg_y = sum(c[1] for c in merged_area['centers']) / len(merged_area['centers'])

        all_lefts = [b['left'] for b in merged_area['all_bounds']]
        all_rights = [b['right'] for b in merged_area['all_bounds']]
        all_tops = [b['top'] for b in merged_area['all_bounds']]
        all_bottoms = [b['bottom'] for b in merged_area['all_bounds']]

        merged.append({
            'center': (avg_x, avg_y),
            'bounds': {
                'left': min(all_lefts),
                'right': max(all_rights),
                'top': min(all_tops),
                'bottom': max(all_bottoms)
            },
            'texts': merged_area['all_texts']
        })

    print(f"Merged to {len(merged)} unique areas")
    return merged

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
        import re as regex
        page_match = regex.search(r'page_(\d+)', image_path.lower())
        if page_match:
            page_num = f"page{page_match.group(1)}_"

    # Create readable text string from texts list
    text_str = "_".join(texts[:3]).replace("/", "-").replace(" ", "")
    if len(text_str) > 20:
        text_str = text_str[:20]

    # Get the directory of the source image to save debug images in the same location
    import os
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

            # Find the actual OCR bounds for this measurement
            for ann in annotations[1:]:  # Skip first annotation which is full text
                text = ann['description'].strip()
                # Check if this annotation is part of our measurement
                if text in best_meas or best_meas in text or any(part in text for part in best_meas.split()):
                    vertices = ann.get('boundingPoly', {}).get('vertices', [])
                    if len(vertices) >= 4:
                        # Get the bounding box in zoomed coordinates
                        x_coords = [v.get('x', 0) for v in vertices]
                        y_coords = [v.get('y', 0) for v in vertices]

                        # Convert to original image coordinates
                        x_coords = [crop_x1 + x for x in x_coords]
                        y_coords = [crop_y1 + y for y in y_coords]

                        if actual_bounds is None:
                            actual_bounds = {
                                'left': min(x_coords),
                                'right': max(x_coords),
                                'top': min(y_coords),
                                'bottom': max(y_coords)
                            }
                        else:
                            # Expand bounds to include all parts of the measurement
                            actual_bounds['left'] = min(actual_bounds['left'], min(x_coords))
                            actual_bounds['right'] = max(actual_bounds['right'], max(x_coords))
                            actual_bounds['top'] = min(actual_bounds['top'], min(y_coords))
                            actual_bounds['bottom'] = max(actual_bounds['bottom'], max(y_coords))

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

def extract_measurements_from_text(text):
    """Extract measurement patterns from text"""
    import re
    # Pattern for measurements: "24 3/4", "11 1/4", "5/8", "20", "6", etc.
    # Matches: whole + fraction | fraction | decimal | any whole number
    pattern = r'\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\b\d+\b'
    matches = re.findall(pattern, text)
    return matches

def verify_measurement_at_center_with_logic(image_path, center, bounds, texts, api_key, center_index=0, save_debug=False):
    """Same as verify_measurement_at_center but returns (measurement, logic_description)"""
    import re
    import os
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
        # Extract page number from image path
        page_num = ""
        if "page_" in image_path.lower():
            page_match = re.search(r'page_(\d+)', image_path.lower())
            if page_match:
                page_num = f"page{page_match.group(1)}_"

        # Create text string from texts list
        text_str = "_".join(texts[:3]).replace("/", "-").replace(" ", "")[:20]

        # Get directory of source image
        image_dir = os.path.dirname(os.path.abspath(image_path))

        # Save debug images - now including zoomed versions
        debug_base = f"DEBUG_{page_num}M{center_index}_pos{int(cx)}x{int(cy)}_{text_str}"
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
        return None, "Zoom OCR detected no text (likely not a measurement)"

    full_text = annotations[0]['description']
    raw_full_text = full_text  # Store the completely raw OCR text before any processing
    # Debug: show raw text first
    print(f"  OCR raw: {repr(full_text)}")
    print(f"  OCR: '{full_text.replace(chr(10), ' ')}'")

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
        # (vertices still span full text, but we'll handle that in bounds calculation)
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
            print(f"      '{item['text']}'")

    # Group items that are horizontally adjacent (for measurements like "13 7/8")
    # Adjust thresholds for zoomed image
    y_threshold = 15 * zoom_factor  # Scale for zoom
    x_threshold = 100 * zoom_factor  # Increased from 90 to handle wider spaced measurements

    # Filter out false positive "1"s (vertical lines detected as "1")
    filtered_items = []
    for item in individual_items:
        if item['text'] == '1':
            # Check if there's a fraction nearby
            has_nearby_fraction = False
            for other in individual_items:
                if other['text'] in ['/2', '/4', '/8', '/16', '1/2', '1/4', '3/8', '5/8', '3/4', '7/8', '1/16', '3/16', '5/16', '7/16', '9/16', '11/16', '13/16', '15/16']:
                    x_dist = abs(other['x'] - item['x'])
                    y_dist = abs(other['y'] - item['y'])
                    # Check if fraction is nearby (within grouping distance)
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
            if abs(other['y'] - item['y']) < y_threshold:
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
        # Remove leading/trailing dashes and commas (including Unicode minus sign and en dash)
        cleaned_text = cleaned_text.strip('—−–-.,')  # Added − (Unicode minus) and – (en dash)
        # Replace internal commas with spaces
        cleaned_text = cleaned_text.replace(',', ' ')
        # Remove apostrophes (they shouldn't be in measurements)
        cleaned_text = cleaned_text.replace("'", " ")
        # Clean up any double spaces
        cleaned_text = ' '.join(cleaned_text.split())
        cleaned_text = cleaned_text.strip()

        # Filter out components with leading/trailing dashes or punctuation-only
        # This ensures bounds calculations don't include vertices with dashes
        cleaned_components = []
        for comp in mg['components']:
            comp_text = comp['text']
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

        # If it has any numbers, accept it as a measurement
        if any(c.isdigit() for c in cleaned_text):
            mg['cleaned_text'] = cleaned_text
            valid_measurements.append(mg)

    if not valid_measurements:
        logic_steps.append(f"Position-grouped: {[mg['text'] for mg in measurement_groups]}")
        logic_steps.append("No measurements with numbers found")
        return None, " | ".join(logic_steps)

    # Find closest to center - adjust for zoom
    crop_cx = (cx - crop_x1) * zoom_factor  # Scale center position for zoomed image
    crop_cy = (cy - crop_y1) * zoom_factor

    best_meas = None
    best_dist = float('inf')

    # Also check for special notations like "NH"
    special_notation = None

    for vm in valid_measurements:
        dist = ((vm['x'] - crop_cx) ** 2 + (vm['y'] - crop_cy) ** 2) ** 0.5
        measurement_value = vm.get('cleaned_text', vm['text'])
        print(f"    Distance for '{measurement_value}': {dist:.1f}")
        if dist < best_dist:
            best_dist = dist
            best_meas = vm

    # Check if "NH" appears in the original OCR text (before measurement groups)
    for mg in measurement_groups:
        if 'NH' in mg['text']:
            special_notation = 'NH'
            break

    if best_meas:
        # Use cleaned text for the measurement value
        measurement_value = best_meas.get('cleaned_text', best_meas['text'])
        raw_ocr_text = best_meas.get('raw_full_text', measurement_value)  # Get the actual raw OCR text with dashes
        print(f"  Chose closest: '{measurement_value}'")

        # Calculate actual bounds from the components in the zoomed image
        # Then scale back to original image coordinates
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

                # Calculate width for just the cleaned text characters
                estimated_text_width = char_count * pixels_per_char

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

        # Return tuple of (cleaned_text, bounds, notation, raw_text)
        if special_notation:
            print(f"  Found special notation: {special_notation}")
        return (measurement_value, actual_bounds, special_notation, raw_ocr_text), " | ".join(logic_steps)

    return (None, None, None, None), "No measurements found"

def find_lines_near_measurement(image, measurement, save_roi_debug=False):
    """Find lines near a specific measurement position that match the text color"""
    x = int(measurement['position'][0])
    y = int(measurement['position'][1])

    # Enable debug for troubleshooting
    DEBUG_MODE = True  # Enable to see width calculation and H-ROI positioning details

    # Use bounds if available
    if 'bounds' in measurement:
        bounds = measurement['bounds']
        text_left = int(bounds['left'])
        text_right = int(bounds['right'])
        text_top = int(bounds['top'])
        text_bottom = int(bounds['bottom'])

        text_width = text_right - text_left
        text_height = text_bottom - text_top

        # Update x,y to be the actual center of the text bounds
        x = int((text_left + text_right) / 2)
        y = int((text_top + text_bottom) / 2)
    else:
        # Fallback to estimates
        text_height = 30
        text_width = len(measurement.get('text', '')) * 15

    # Add padding for image skew or misalignment
    padding = 10
    text_height_with_padding = text_height + padding
    text_width_with_padding = text_width + padding

    # Create ROIs for horizontal and vertical line detection
    # For horizontal lines: Look LEFT and RIGHT of the text
    # Use 0.75x text width but ensure minimum of 40 pixels for short text
    h_strip_extension = max(40, int(text_width * 0.75))

    # IMPROVED: Ensure minimum gap between text edge and H-ROI to avoid false line detection
    min_gap = 10  # Minimum 10px gap between text and H-ROI

    # Use normal H-ROI height
    h_roi_height_multiplier = 1.0  # Normal height (1x text height)

    # Left horizontal ROI - with validated gap from text edge
    text_left_edge = int(x - text_width//2)
    h_left_x2 = max(0, text_left_edge - min_gap)  # Ensure gap from text
    h_left_x1 = max(0, h_left_x2 - h_strip_extension)  # Extend left from x2
    h_left_y1 = int(y - text_height * h_roi_height_multiplier)  # Adjustable height
    h_left_y2 = int(y + text_height * h_roi_height_multiplier)

    # Right horizontal ROI - with validated gap from text edge
    text_right_edge = int(x + text_width//2)
    h_right_x1 = min(image.shape[1], text_right_edge + min_gap)  # Ensure gap from text
    h_right_x2 = min(image.shape[1], h_right_x1 + h_strip_extension)  # Extend right from x1
    h_right_y1 = h_left_y1
    h_right_y2 = h_left_y2

    if DEBUG_MODE:
        print(f"  Text bounds: left={text_left_edge}, right={text_right_edge}, center_x={x}, width={text_width}")
        print(f"  H-strip extension: {h_strip_extension}px (0.75x text_width)")
        print(f"  Left H-ROI: x=[{h_left_x1}, {h_left_x2}], width={h_left_x2-h_left_x1}, gap_from_text={text_left_edge - h_left_x2}")
        print(f"  Right H-ROI: x=[{h_right_x1}, {h_right_x2}], width={h_right_x2-h_right_x1}, gap_from_text={h_right_x1 - text_right_edge}")

    # For vertical lines: Look ABOVE and BELOW the text
    # Make ROIs TALLER and NARROWER for better vertical line detection
    v_strip_extension = int(text_height * 4)  # Increased from 2x to 4x height for taller ROI

    # Use full text width for vertical ROIs
    # Top vertical ROI - full width of text
    v_top_x1 = int(x - text_width//2)
    v_top_x2 = int(x + text_width//2)
    v_top_y1 = max(0, int(y - text_height//2 - v_strip_extension))
    v_top_y2 = int(y - text_height//2 - 5)

    # Bottom vertical ROI - full width
    v_bottom_x1 = v_top_x1
    v_bottom_x2 = v_top_x2
    v_bottom_y1 = int(y + text_height//2 + 5)
    v_bottom_y2 = min(image.shape[0], int(y + text_height//2 + v_strip_extension))

    # Process horizontal and vertical ROIs separately
    horizontal_lines = []
    vertical_lines = []

    # Helper function to detect arrows in an ROI
    def detect_arrow_in_roi(roi_image, direction):
        """
        Detect arrow pointing in specified direction by looking for converging lines
        direction: 'up', 'down', 'left', 'right'
        """
        if roi_image.size == 0:
            return False

        # Apply HSV filter to detect only green dimension arrows (not cabinet edges)
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array(HSV_CONFIG['lower_green'])
        upper_green = np.array(HSV_CONFIG['upper_green'])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply morphological operations to connect nearby pixels
        kernel = np.ones((2,2), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Detect edges only on the green-filtered image
        edges = cv2.Canny(green_mask, 30, 100)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)

        if lines is None or len(lines) < 2:
            return False

        # Debug: Show what lines we found in arrow detection
        if DEBUG_MODE and direction == 'up':
            print(f"        Arrow detection found {len(lines)} lines in ROI")

        # Look for converging lines that could form an arrow
        # For up arrow: lines should converge at top (Y decreases as they meet)
        # For down arrow: lines should converge at bottom (Y increases as they meet)

        converging_pairs = 0
        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            for j in range(i + 1, len(lines)):
                x3, y3, x4, y4 = lines[j][0]
                angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3))

                # Check if lines have opposite angles (forming a V or ^ shape)
                angle_diff = abs(angle1 + angle2)  # If one is positive and one negative, sum is small

                if direction == 'up':
                    # Debug angle pairs
                    if DEBUG_MODE and i == 0 and j == 1:
                        print(f"        First pair angles: {angle1:.1f}° and {angle2:.1f}°, diff={angle_diff:.1f}")

                    # For up arrow, look for lines that converge upward (like ^)
                    # Check for any pair of lines with significantly different angles
                    # that could form the two sides of an arrow
                    if abs(angle1 - angle2) > 15:  # Lines with different angles
                        # Could be arrow sides
                        converging_pairs += 1
                        if DEBUG_MODE:
                            print(f"        Found converging pair: {angle1:.1f}° and {angle2:.1f}°")

                elif direction == 'down':
                    # For down arrow, look for lines that converge downward (like v)
                    # Angles should be roughly -135 and +135 (or similar)
                    if angle_diff > 150:  # Angles pointing down from opposite sides
                        converging_pairs += 1

                elif direction == 'left':
                    # For left arrow, look for lines that converge leftward (like <)
                    # Angles should form a < shape (one line angled up-left, one down-left)
                    if abs(angle1 - angle2) > 15:  # Lines with different angles
                        converging_pairs += 1

                elif direction == 'right':
                    # For right arrow, look for lines that converge rightward (like >)
                    # Angles should form a > shape (one line angled up-right, one down-right)
                    if abs(angle1 - angle2) > 15:  # Lines with different angles
                        converging_pairs += 1

        return converging_pairs > 0

    # Helper function to detect lines in an ROI
    def detect_lines_in_roi(roi_image, roi_x_offset, roi_y_offset):
        if roi_image.size == 0:
            return []

        # Apply HSV filter to detect only green dimension lines (not cabinet edges)
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array(HSV_CONFIG['lower_green'])
        upper_green = np.array(HSV_CONFIG['upper_green'])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply morphological operations to connect nearby pixels
        kernel = np.ones((3,3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Detect edges only on the green-filtered image
        edges = cv2.Canny(green_mask, 30, 100)

        # Find lines with HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=15,
                                minLineLength=20,
                                maxLineGap=30)

        # Convert line coordinates back to full image coordinates
        if lines is not None:
            adjusted_lines = []
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                adjusted_lines.append([lx1 + roi_x_offset, ly1 + roi_y_offset,
                                      lx2 + roi_x_offset, ly2 + roi_y_offset])
            return adjusted_lines
        return []

    # Helper function to extract pixels from a rotated rectangular region
    def extract_rotated_roi(image, x1, y1, x2, y2, angle_degrees, center_point):
        """Extract pixels within a rotated rectangle from the image"""
        # Calculate the four corners of the rectangle
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)

        # Get center point for rotation
        cx, cy = center_point

        # Rotate corners around center point
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotated_corners = []
        for px, py in corners:
            # Translate to origin
            px_rel = px - cx
            py_rel = py - cy
            # Rotate
            px_rot = px_rel * cos_a - py_rel * sin_a
            py_rot = px_rel * sin_a + py_rel * cos_a
            # Translate back
            rotated_corners.append([px_rot + cx, py_rot + cy])

        rotated_corners = np.array(rotated_corners, dtype=np.float32)

        # Get the bounding box of the rotated rectangle
        min_x = int(np.floor(rotated_corners[:, 0].min()))
        max_x = int(np.ceil(rotated_corners[:, 0].max()))
        min_y = int(np.floor(rotated_corners[:, 1].min()))
        max_y = int(np.ceil(rotated_corners[:, 1].max()))

        # Clamp to image bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(image.shape[1], max_x)
        max_y = min(image.shape[0], max_y)

        # Calculate the dimensions for the output ROI
        width = x2 - x1
        height = y2 - y1

        # Define destination points (unrotated rectangle of same size)
        dst_corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)

        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(rotated_corners, dst_corners)

        # Apply perspective transform to get the rotated region
        roi = cv2.warpPerspective(image, M, (width, height))

        return roi

    # Check if we need to apply rotation to ROIs
    roi_rotation_angle = measurement.get('roi_rotation_angle', 0.0)

    # Store the actual ROI coordinates in the measurement for visualization
    measurement['actual_h_left_roi'] = (h_left_x1, h_left_y1, h_left_x2, h_left_y2)
    measurement['actual_h_right_roi'] = (h_right_x1, h_right_y1, h_right_x2, h_right_y2)
    measurement['actual_v_top_roi'] = (v_top_x1, v_top_y1, v_top_x2, v_top_y2)
    measurement['actual_v_bottom_roi'] = (v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2)

    # Initialize arrow detection flags
    has_left_arrow = False
    has_right_arrow = False

    # Search for horizontal lines
    if h_left_x2 > h_left_x1 and h_left_y2 > h_left_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Left H-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            h_left_roi = extract_rotated_roi(image, h_left_x1, h_left_y1, h_left_x2, h_left_y2,
                                             roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            h_left_roi = image[h_left_y1:h_left_y2, h_left_x1:h_left_x2]

        if DEBUG_MODE:
            print(f"      ACTUAL Left H-ROI coords: x=[{h_left_x1}, {h_left_x2}], y=[{h_left_y1}, {h_left_y2}]")

        left_h_lines = detect_lines_in_roi(h_left_roi, h_left_x1, h_left_y1)
        print(f"      Left H-ROI: shape={h_left_roi.shape}, found {len(left_h_lines)} lines")

        # If no lines found, try arrow detection
        if not left_h_lines:
            has_left_arrow = detect_arrow_in_roi(h_left_roi, 'left')
            if has_left_arrow:
                print(f"        Arrow detection found left arrow")

        if left_h_lines:
            for line in left_h_lines:
                lx1, ly1, lx2, ly2 = line
                angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                print(f"        Raw line: angle={angle:.1f}°")
    else:
        left_h_lines = []
        print(f"      Left H-ROI: Invalid bounds")

    if h_right_x2 > h_right_x1 and h_right_y2 > h_right_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Right H-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            h_right_roi = extract_rotated_roi(image, h_right_x1, h_right_y1, h_right_x2, h_right_y2,
                                              roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            h_right_roi = image[h_right_y1:h_right_y2, h_right_x1:h_right_x2]

        if DEBUG_MODE:
            print(f"      ACTUAL Right H-ROI coords: x=[{h_right_x1}, {h_right_x2}], y=[{h_right_y1}, {h_right_y2}]")

        right_h_lines = detect_lines_in_roi(h_right_roi, h_right_x1, h_right_y1)
        print(f"      Right H-ROI: shape={h_right_roi.shape}, found {len(right_h_lines)} lines")

        # If no lines found, try arrow detection
        if not right_h_lines:
            has_right_arrow = detect_arrow_in_roi(h_right_roi, 'right')
            if has_right_arrow:
                print(f"        Arrow detection found right arrow")

        if right_h_lines:
            for line in right_h_lines:
                lx1, ly1, lx2, ly2 = line
                angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                print(f"        Raw line: angle={angle:.1f}°")
    else:
        right_h_lines = []
        print(f"      Right H-ROI: Invalid bounds")

    # Filter for horizontal lines (more tolerant angles)
    # When ROI is rotated, adjust the expected angle
    for line in left_h_lines + right_h_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))

        # Try both angle adjustments to see which makes the line horizontal
        # Adjustment 1: angle + rotation (lines rotated opposite direction)
        adjusted_angle_1 = angle + roi_rotation_angle
        while adjusted_angle_1 > 180:
            adjusted_angle_1 -= 360
        while adjusted_angle_1 < -180:
            adjusted_angle_1 += 360

        # Adjustment 2: angle - rotation (lines rotated same direction)
        adjusted_angle_2 = angle - roi_rotation_angle
        while adjusted_angle_2 > 180:
            adjusted_angle_2 -= 360
        while adjusted_angle_2 < -180:
            adjusted_angle_2 += 360

        abs_adjusted_1 = abs(adjusted_angle_1)
        abs_adjusted_2 = abs(adjusted_angle_2)

        # Check if either adjustment makes it horizontal
        # More tolerant: 0-35° or 145-180° for "generally horizontal"
        is_horizontal_1 = abs_adjusted_1 < 35 or abs_adjusted_1 > 145
        is_horizontal_2 = abs_adjusted_2 < 35 or abs_adjusted_2 > 145

        if is_horizontal_1 or is_horizontal_2:
            # Use whichever adjustment is closer to horizontal
            if is_horizontal_1 and (not is_horizontal_2 or abs_adjusted_1 < abs_adjusted_2):
                adjusted_angle = adjusted_angle_1
            else:
                adjusted_angle = adjusted_angle_2

            horizontal_lines.append({
                'coords': (lx1, ly1, lx2, ly2),
                'distance': abs(y - (ly1 + ly2) / 2),
                'type': 'horizontal_line',
                'angle': angle,
                'adjusted_angle': adjusted_angle
            })

    if horizontal_lines:
        print(f"      Found {len(horizontal_lines)} horizontal line candidates")

    # Search for vertical lines and arrows
    has_up_arrow = False
    has_down_arrow = False

    if v_top_x2 > v_top_x1 and v_top_y2 > v_top_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Top V-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            v_top_roi = extract_rotated_roi(image, v_top_x1, v_top_y1, v_top_x2, v_top_y2,
                                            roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            v_top_roi = image[v_top_y1:v_top_y2, v_top_x1:v_top_x2]

        top_v_lines = detect_lines_in_roi(v_top_roi, v_top_x1, v_top_y1)
        has_up_arrow = detect_arrow_in_roi(v_top_roi, 'up')
        print(f"      Top V-ROI: shape={v_top_roi.shape}, found {len(top_v_lines)} lines, up-arrow={has_up_arrow}")
    else:
        top_v_lines = []
        print(f"      Top V-ROI: Invalid bounds")

    if v_bottom_x2 > v_bottom_x1 and v_bottom_y2 > v_bottom_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Bottom V-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            v_bottom_roi = extract_rotated_roi(image, v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2,
                                               roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            v_bottom_roi = image[v_bottom_y1:v_bottom_y2, v_bottom_x1:v_bottom_x2]

        bottom_v_lines = detect_lines_in_roi(v_bottom_roi, v_bottom_x1, v_bottom_y1)
        has_down_arrow = detect_arrow_in_roi(v_bottom_roi, 'down')
        print(f"      Bottom V-ROI: shape={v_bottom_roi.shape}, found {len(bottom_v_lines)} lines, down-arrow={has_down_arrow}")
    else:
        bottom_v_lines = []
        print(f"      Bottom V-ROI: Invalid bounds")

    # Store original line counts before filtering
    original_top_line_count = len(top_v_lines)
    original_bottom_line_count = len(bottom_v_lines)

    # Filter for vertical lines (more tolerant angles) and track their source
    # When ROI is rotated, adjust the expected angle
    for line in top_v_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))

        # Adjust angle for rotation
        adjusted_angle = angle + roi_rotation_angle

        # Normalize to -180 to 180
        while adjusted_angle > 180:
            adjusted_angle -= 360
        while adjusted_angle < -180:
            adjusted_angle += 360

        abs_adjusted_angle = abs(adjusted_angle)

        # Debug angle detection
        if DEBUG_MODE and measurement.get('text') == '5 1/2':
            x_dist = abs(x - (lx1 + lx2) / 2)
            print(f"        Top line: angle={angle:.1f}°, adjusted={adjusted_angle:.1f}°, x_dist={x_dist:.1f}, coords=({lx1},{ly1})-({lx2},{ly2})")
        # More tolerant: 55-125° for "generally vertical"
        if 55 < abs_adjusted_angle < 125:
            x_distance = abs(x - (lx1 + lx2) / 2)
            # Keep the distance check but make it less strict
            if x_distance > 2:  # Reduced from 5 to 2
                vertical_lines.append({
                    'coords': (lx1, ly1, lx2, ly2),
                    'distance': x_distance,
                    'type': 'vertical_line',
                    'source': 'top',  # Track that this came from top ROI
                    'angle': angle,  # Store the original angle
                    'adjusted_angle': adjusted_angle  # Store adjusted angle for skew detection
                })
                if DEBUG_MODE:
                    print(f"        Top V-line: angle={angle:.1f}°, adjusted={adjusted_angle:.1f}°, x_dist={x_distance:.1f}")

    for line in bottom_v_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))

        # Adjust angle for rotation
        adjusted_angle = angle + roi_rotation_angle

        # Normalize to -180 to 180
        while adjusted_angle > 180:
            adjusted_angle -= 360
        while adjusted_angle < -180:
            adjusted_angle += 360

        abs_adjusted_angle = abs(adjusted_angle)

        # More tolerant: 55-125° for "generally vertical"
        if 55 < abs_adjusted_angle < 125:
            x_distance = abs(x - (lx1 + lx2) / 2)
            # Keep the distance check but make it less strict
            if x_distance > 2:  # Reduced from 5 to 2
                vertical_lines.append({
                    'coords': (lx1, ly1, lx2, ly2),
                    'distance': x_distance,
                    'type': 'vertical_line',
                    'source': 'bottom',  # Track that this came from bottom ROI
                    'angle': angle,  # Store the original angle
                    'adjusted_angle': adjusted_angle  # Store adjusted angle for skew detection
                })
                if DEBUG_MODE:
                    print(f"        Bottom V-line: angle={angle:.1f}°, adjusted={adjusted_angle:.1f}°, x_dist={x_distance:.1f}")

    if vertical_lines:
        print(f"      Found {len(vertical_lines)} vertical line candidates")

    # NEW LOGIC: Sequential check requiring HORIZONTAL lines on BOTH sides for WIDTH,
    # or VERTICAL lines on BOTH sides for HEIGHT

    # Step 1: Check for WIDTH - must have HORIZONTAL lines on BOTH left AND right
    # horizontal_lines already contains only horizontal lines (angle filtered)
    # Check which side of the text center each horizontal line is on
    left_h_lines = [l for l in horizontal_lines if l['coords'][0] < x and l['coords'][2] < x]  # Both endpoints left of center
    right_h_lines = [l for l in horizontal_lines if l['coords'][0] > x and l['coords'][2] > x]  # Both endpoints right of center

    has_left_horizontal = len(left_h_lines) > 0 or has_left_arrow
    has_right_horizontal = len(right_h_lines) > 0 or has_right_arrow

    if has_left_horizontal and has_right_horizontal:
        # Found HORIZONTAL lines or arrows on BOTH left and right - classify as WIDTH and stop
        best_h = min(horizontal_lines, key=lambda l: l['distance']) if horizontal_lines else None
        arrow_msg = ""
        if has_left_arrow:
            arrow_msg += " (left arrow)"
        if has_right_arrow:
            arrow_msg += " (right arrow)"
        print(f"      Found HORIZONTAL lines/arrows on BOTH left and right{arrow_msg} → WIDTH")

        # Store all horizontal lines for pairing logic (to extend lines and check for heights)
        measurement['h_lines'] = horizontal_lines

        # Calculate average angle of horizontal lines for intersection calculation
        if horizontal_lines:
            avg_angle = sum(l['angle'] for l in horizontal_lines) / len(horizontal_lines)
            measurement['h_line_angle'] = avg_angle

        if best_h:
            return {
                'line': best_h['coords'],
                'orientation': 'horizontal_line',
                'distance': best_h['distance'],
                'angle': best_h.get('angle', 0)
            }
        else:
            # Arrows detected but no actual lines - still classify as WIDTH
            return {
                'line': None,
                'orientation': 'horizontal_line',
                'distance': 0,
                'angle': 0
            }

    # Step 2: If not WIDTH, check for HEIGHT
    # Use the 'source' field to determine which ROI each line came from
    top_v_lines = [l for l in vertical_lines if l.get('source') == 'top']
    bottom_v_lines = [l for l in vertical_lines if l.get('source') == 'bottom']

    # Debug: Show what lines we found
    if DEBUG_MODE and len(vertical_lines) > 0:
        print(f"      Vertical lines analysis:")
        print(f"        Lines from TOP ROI: {len(top_v_lines)}")
        print(f"        Lines from BOTTOM ROI: {len(bottom_v_lines)}")

    # Check for vertical lines - must have lines from BOTH top AND bottom ROIs
    has_top_vertical = len(top_v_lines) > 0
    has_bottom_vertical = len(bottom_v_lines) > 0

    if has_top_vertical and has_bottom_vertical:
        # Found VERTICAL lines on BOTH top and bottom - classify as HEIGHT
        best_v = min(vertical_lines, key=lambda l: l['distance'])
        print(f"      Found VERTICAL lines on BOTH top and bottom → HEIGHT")

        # Calculate average angle of vertical lines for intersection calculation
        if vertical_lines:
            avg_angle = sum(l['angle'] for l in vertical_lines) / len(vertical_lines)
            measurement['v_line_angle'] = avg_angle

        return {
            'line': best_v['coords'],
            'orientation': 'vertical_line',
            'distance': best_v['distance'],
            'angle': best_v.get('angle', 90)
        }

    # Step 2b: If no vertical lines on both sides, check for arrows as fallback
    if has_up_arrow and has_down_arrow:
        print(f"      Found vertical arrows (up and down) → HEIGHT")
        # Use any vertical lines if they exist, otherwise create placeholder
        if vertical_lines:
            best_v = min(vertical_lines, key=lambda l: l['distance'])
            return {
                'line': best_v['coords'],
                'orientation': 'vertical_line',
                'distance': best_v['distance'],
                'arrow_based': True
            }
        else:
            # No vertical lines but arrows detected - use center position
            return {
                'line': (x, y-20, x, y+20),
                'orientation': 'vertical_line',
                'distance': 0,
                'arrow_based': True
            }

    # Step 2c: Additional fallback for small measurements with single-sided arrow
    # Parse the measurement value to check if it's less than 10"
    try:
        # Extract numeric value from measurement text (e.g., "5 1/2" -> 5.5)
        import re
        text = measurement.get('text', '')
        # Match patterns like "5 1/2", "10 3/4", etc.
        match = re.match(r'^(\d+)\s*(\d+)?/?(\d+)?', text)
        if match:
            whole = float(match.group(1))
            if match.group(2) and match.group(3):
                # Has fraction part
                numerator = float(match.group(2))
                denominator = float(match.group(3))
                value = whole + (numerator / denominator)
            else:
                value = whole

            # If measurement is less than 10" and has vertical arrow on at least one side
            if value < 10 and (has_up_arrow or has_down_arrow):
                print(f"      Small measurement ({value:.1f}\") with vertical arrow → HEIGHT (fallback)")
                # Use any vertical lines if they exist
                if vertical_lines:
                    best_v = min(vertical_lines, key=lambda l: l['distance'])
                    return {
                        'line': best_v['coords'],
                        'orientation': 'vertical_line',
                        'distance': best_v['distance'],
                        'small_height_fallback': True
                    }
                else:
                    # No vertical lines but arrow detected - use center position
                    return {
                        'line': (x, y-20, x, y+20),
                        'orientation': 'vertical_line',
                        'distance': 0,
                        'small_height_fallback': True
                    }
    except:
        pass  # If parsing fails, continue to UNCLASSIFIED

    # Step 3: Neither condition met - UNCLASSIFIED
    print(f"      No lines on both sides (L-horiz:{has_left_horizontal} R-horiz:{has_right_horizontal} T-vert:{has_top_vertical} B-vert:{has_bottom_vertical}) → UNCLASSIFIED")

    # Calculate skew angle from detected vertical lines for fallback retry
    skew_angle = None
    if vertical_lines:
        # Use the closest vertical line's angle to estimate skew
        closest_vline = min(vertical_lines, key=lambda l: l['distance'])
        vline_angle = closest_vline.get('angle', 90.0)
        # Skew is deviation from perfect vertical (90°)
        skew_angle = 90.0 - vline_angle
        print(f"      Detected vertical line angle: {vline_angle:.1f}°, skew from vertical: {skew_angle:.1f}°")

    # Return None but include info about whether vertical lines were found
    # This info will be used by the classification function for fallback retry
    return {
        'unclassified': True,
        'has_vertical_lines': has_top_vertical or has_bottom_vertical,
        'has_left_horizontal': has_left_horizontal,
        'has_right_horizontal': has_right_horizontal,
        'skew_angle': skew_angle  # Include skew for second fallback
    }

def classify_measurements_by_lines(image, measurements):
    """Classify measurements as WIDTH, HEIGHT, or UNCLASSIFIED based on nearby dimension lines"""
    classified = {
        'width': [],
        'height': [],
        'unclassified': []
    }

    measurement_categories = []

    for i, meas in enumerate(measurements):
        print(f"\n  Analyzing measurement {i+1}: '{meas['text']}' at ({meas['position'][0]:.0f}, {meas['position'][1]:.0f})")

        # Find lines near this measurement
        line_info = find_lines_near_measurement(image, meas)

        # Check if it's actually classified or unclassified
        if line_info and not line_info.get('unclassified'):
            print(f"    Found {line_info['orientation']} at distance {line_info['distance']:.1f} pixels")
            # Horizontal line = WIDTH measurement
            # Vertical line = HEIGHT measurement
            if line_info['orientation'] == 'horizontal_line':
                classified['width'].append(meas['text'])
                measurement_categories.append('width')
                print(f"    → Classified as WIDTH")
            elif line_info['orientation'] == 'vertical_line':
                classified['height'].append(meas['text'])
                measurement_categories.append('height')
                print(f"    → Classified as HEIGHT")
        else:
            # UNCLASSIFIED - try clockwise rotation fallback
            print(f"    UNCLASSIFIED - Trying clockwise rotation fallback...")

            # Try clockwise rotation (22.5°)
            print(f"    Attempting +22.5° rotation...")
            meas_cw = meas.copy()
            meas_cw['roi_rotation_angle'] = 22.5
            line_info_cw = find_lines_near_measurement(image, meas_cw)

            if line_info_cw and not line_info_cw.get('unclassified'):
                print(f"    ROTATION FALLBACK SUCCESS: Found {line_info_cw['orientation']} with +22.5° rotation!")
                if line_info_cw['orientation'] == 'horizontal_line':
                    classified['width'].append(meas['text'])
                    measurement_categories.append('width')
                    meas['roi_rotation_angle'] = 22.5
                    meas['actual_h_left_roi'] = meas_cw.get('actual_h_left_roi')
                    meas['actual_h_right_roi'] = meas_cw.get('actual_h_right_roi')
                    meas['actual_v_top_roi'] = meas_cw.get('actual_v_top_roi')
                    meas['actual_v_bottom_roi'] = meas_cw.get('actual_v_bottom_roi')
                    print(f"    → Classified as WIDTH (via +22.5° rotation)")
                elif line_info_cw['orientation'] == 'vertical_line':
                    classified['height'].append(meas['text'])
                    measurement_categories.append('height')
                    meas['roi_rotation_angle'] = 22.5
                    meas['actual_h_left_roi'] = meas_cw.get('actual_h_left_roi')
                    meas['actual_h_right_roi'] = meas_cw.get('actual_h_right_roi')
                    meas['actual_v_top_roi'] = meas_cw.get('actual_v_top_roi')
                    meas['actual_v_bottom_roi'] = meas_cw.get('actual_v_bottom_roi')
                    print(f"    → Classified as HEIGHT (via +22.5° rotation)")
            else:
                # Still unclassified after rotation - reset to 0° for viz
                meas['roi_rotation_angle'] = 0.0  # Reset to regular ROIs in viz
                meas['rotation_failed'] = True
                # Don't store the rotated ROI coords - keep original non-rotated ones
                print(f"    No dimension lines found even with rotation")
                classified['unclassified'].append(meas['text'])
                measurement_categories.append('unclassified')
                print(f"    → Classified as UNCLASSIFIED (rotation attempted but failed, ROIs reset to 0°)")

    return classified, measurement_categories

def pair_measurements_by_proximity(classified_measurements, all_measurements):
    """
    Pair width and height measurements based on proximity to form cabinet openings
    Returns list of paired openings
    """
    print("\n=== PAIRING MEASUREMENTS INTO OPENINGS ===")

    # Initialize variables
    unpaired_heights = []
    used_heights = set()

    # Extract measurements by type with their positions
    widths = []
    heights = []

    for i, meas in enumerate(all_measurements):
        if i < len(classified_measurements):
            category = classified_measurements[i]
            if category == 'width':
                widths.append({
                    'text': meas['text'],
                    'x': meas['position'][0],
                    'y': meas['position'][1],
                    'bounds': meas.get('bounds', None),
                    'h_line_angle': meas.get('h_line_angle', 0)  # Store the horizontal line angle
                })
            elif category == 'height':
                height_data = {
                    'text': meas['text'],
                    'x': meas['position'][0],
                    'y': meas['position'][1],
                    'bounds': meas.get('bounds', None),
                    'v_line_angle': meas.get('v_line_angle', 90)  # Store the vertical line angle
                }
                if 'notation' in meas:
                    height_data['notation'] = meas['notation']
                heights.append(height_data)

    if not widths or not heights:
        print("  Cannot pair - need both widths and heights")
        return [], []

    # Sort by Y position for analysis
    widths.sort(key=lambda w: w['y'])
    heights.sort(key=lambda h: h['y'])

    print(f"\n  Widths to pair ({len(widths)}):")
    for w in widths:
        print(f"    {w['text']} at ({w['x']:.0f}, {w['y']:.0f})")

    print(f"\n  Heights to pair ({len(heights)}):")
    for h in heights:
        print(f"    {h['text']} at ({h['x']:.0f}, {h['y']:.0f})")

    # Check if dimensions are stacked (similar X positions)
    STACKING_TOLERANCE = 100  # pixels

    # Check width stacking
    width_x_positions = [w['x'] for w in widths]
    width_x_range = max(width_x_positions) - min(width_x_positions) if widths else 0
    widths_stacked = width_x_range < STACKING_TOLERANCE and len(widths) > 1

    # Check height stacking
    height_x_positions = [h['x'] for h in heights]
    height_x_range = max(height_x_positions) - min(height_x_positions) if heights else 0
    heights_stacked = height_x_range < STACKING_TOLERANCE and len(heights) > 1

    print(f"\n  Arrangement Analysis:")
    print(f"    Widths X-range: {width_x_range:.0f}px - {'STACKED' if widths_stacked else 'SIDE-BY-SIDE'}")
    print(f"    Heights X-range: {height_x_range:.0f}px - {'STACKED' if heights_stacked else 'SIDE-BY-SIDE'}")

    openings = []
    used_heights = set()

    print(f"\n  Pairing Strategy: {'Stacked widths' if widths_stacked else 'Side-by-side widths'}")

    if widths_stacked:
        # Each height finds its closest width
        print("  → Each height will find its closest width")

        for height in heights:
            best_width = None
            best_distance = float('inf')

            for width in widths:
                # Calculate Euclidean distance
                distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)

                if distance < best_distance:
                    best_distance = distance
                    best_width = width

            if best_width:
                print(f"\n  Height '{height['text']}' pairs with width '{best_width['text']}'")
                print(f"    Distance: {best_distance:.0f}px")
                opening_data = {
                    'width': best_width['text'],
                    'height': height['text'],
                    'width_pos': (best_width['x'], best_width['y']),
                    'height_pos': (height['x'], height['y']),
                    'distance': best_distance,
                    'width_angle': best_width.get('h_line_angle', 0),  # Store width's h-line angle
                    'height_angle': height.get('v_line_angle', 90)  # Store height's v-line angle
                }
                if 'notation' in height:
                    opening_data['notation'] = height['notation']
                openings.append(opening_data)
                # Track by position, not text
                height_id = (height['x'], height['y'])
                used_heights.add(height_id)

    else:
        # Side-by-side: each width finds closest height above it
        print("  → Each width will find closest height above it")

        for width in widths:
            best_height = None
            best_distance = float('inf')

            # First try: Look for height above this width
            for height in heights:
                # Only consider heights above this width (at least 20px above)
                if height['y'] >= width['y'] - 20:
                    continue

                # Calculate weighted distance (X distance weighted more)
                x_dist = abs(height['x'] - width['x'])
                y_dist = abs(height['y'] - width['y'])
                distance = np.sqrt(x_dist**2 + y_dist**2)
                weighted_distance = distance + (x_dist * 0.5)

                if weighted_distance < best_distance:
                    best_distance = weighted_distance
                    best_height = height

            # Second try: If no height found above, extend H-lines and check if height is above extended line
            if not best_height:
                print(f"\n  Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): No height found above, extending H-lines...")

                # Get the H-lines from this width measurement
                width_measurement = None
                for i, meas in enumerate(all_measurements):
                    if abs(meas['position'][0] - width['x']) < 5 and abs(meas['position'][1] - width['y']) < 5:
                        width_measurement = meas
                        break

                if width_measurement and 'h_lines' in width_measurement:
                    h_lines = width_measurement['h_lines']
                    print(f"    Found {len(h_lines)} H-lines to extend")

                    # Filter to only use H-lines on the side TOWARD the height
                    # If height is to the left of width, use left H-lines; if right, use right H-lines
                    for height in heights:
                        if height['x'] < width['x']:
                            # Height is to the left, use H-lines on left side
                            relevant_h_lines = [hl for hl in h_lines if (hl['coords'][0] + hl['coords'][2])/2 < width['x']]
                        else:
                            # Height is to the right, use H-lines on right side
                            relevant_h_lines = [hl for hl in h_lines if (hl['coords'][0] + hl['coords'][2])/2 > width['x']]

                        if not relevant_h_lines:
                            continue

                        print(f"    Using {len(relevant_h_lines)} H-lines on {'left' if height['x'] < width['x'] else 'right'} side toward height")

                        # For each relevant H-line, calculate where it would be at the height's X position
                        for h_line in relevant_h_lines:
                            # Get line coordinates and angle
                            lx1, ly1, lx2, ly2 = h_line['coords']
                            if 'angle' in h_line:
                                angle_deg = h_line['angle']
                            else:
                                angle_deg = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))

                            # Use the line endpoints to calculate slope more accurately
                            # The line goes from (lx1, ly1) to (lx2, ly2)
                            # Calculate slope: dy/dx
                            if lx2 - lx1 != 0:
                                slope = (ly2 - ly1) / (lx2 - lx1)
                            else:
                                slope = 0

                            # Use the RIGHTMOST point as reference when extending LEFT
                            # (We're on the left side of the width, extending toward the height on the left)
                            if lx1 > lx2:
                                ref_x, ref_y = lx1, ly1
                            else:
                                ref_x, ref_y = lx2, ly2

                            # Extend line to height's X position using slope
                            # Since we're extending left from the right reference point,
                            # delta_x will be negative
                            delta_x = height['x'] - ref_x
                            delta_y = slope * delta_x
                            line_y_at_height = ref_y + delta_y

                            # Check if height is above this extended line (with tolerance)
                            tolerance = 200  # pixels tolerance above the line
                            print(f"      H-line angle={angle_deg:.1f}° slope={slope:.3f}: from ({lx1:.0f},{ly1:.0f})-({lx2:.0f},{ly2:.0f}), ref=({ref_x:.0f},{ref_y:.0f}), extends to X={height['x']:.0f} Y={line_y_at_height:.0f}, height is at Y={height['y']:.0f}")
                            if height['y'] < line_y_at_height and (line_y_at_height - height['y']) < tolerance:
                                # Height is above the extended H-line - valid geometric pair!
                                x_dist = abs(height['x'] - width['x'])
                                y_dist = abs(height['y'] - width['y'])
                                distance = np.sqrt(x_dist**2 + y_dist**2)
                                weighted_distance = distance + (x_dist * 0.5)

                                print(f"    Extended H-line (angle={angle_deg:.1f}°) from width reaches Y={line_y_at_height:.0f} at height's X={height['x']:.0f}")
                                print(f"    Height at Y={height['y']:.0f} is {line_y_at_height - height['y']:.0f}px above extended line - VALID PAIR!")

                                if weighted_distance < best_distance:
                                    best_distance = weighted_distance
                                    best_height = height
                                    break  # Found via extended line, stop checking more lines

                        if best_height:
                            break  # Found a height, stop checking more heights

            # Third fallback: If still no height, look for height below
            if not best_height:
                print(f"\n  Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): No height via extended lines, trying below...")
                for height in heights:
                    # Only consider heights below this width (at least 20px below)
                    if height['y'] <= width['y'] + 20:
                        continue

                    # Calculate weighted distance (X distance weighted more)
                    x_dist = abs(height['x'] - width['x'])
                    y_dist = abs(height['y'] - width['y'])
                    distance = np.sqrt(x_dist**2 + y_dist**2)
                    weighted_distance = distance + (x_dist * 0.5)

                    if weighted_distance < best_distance:
                        best_distance = weighted_distance
                        best_height = height

            if best_height:
                print(f"\n  Width '{width['text']}' pairs with height '{best_height['text']}'")
                print(f"    Distance: {best_distance:.0f}px")
                opening_data = {
                    'width': width['text'],
                    'height': best_height['text'],
                    'width_pos': (width['x'], width['y']),
                    'height_pos': (best_height['x'], best_height['y']),
                    'distance': best_distance,
                    'width_angle': width.get('h_line_angle', 0),  # Store width's h-line angle
                    'height_angle': best_height.get('v_line_angle', 90)  # Store height's v-line angle
                }
                if 'notation' in best_height:
                    opening_data['notation'] = best_height['notation']
                openings.append(opening_data)
                # Track this specific height instance (by position) as used
                if 'used_heights' not in locals():
                    used_heights = set()
                # Use position tuple as unique identifier
                height_id = (best_height['x'], best_height['y'])
                used_heights.add(height_id)

        # Second pass for unpaired heights (check by position, not text)
        if 'used_heights' not in locals():
            used_heights = set()
        unpaired_heights = [h for h in heights if (h['x'], h['y']) not in used_heights]
        if unpaired_heights:
            print(f"\n  Second pass for {len(unpaired_heights)} unpaired heights:")

            for height in unpaired_heights:
                # First: Look for width directly below (within X-range of height's text width)
                best_width = None
                best_distance = float('inf')

                # Estimate text width for the height (using actual bounds if available)
                if 'bounds' in height and height['bounds']:
                    height_text_width = height['bounds']['right'] - height['bounds']['left']
                else:
                    height_text_width = len(height['text']) * 15  # Fallback estimate

                # Define X-range for "directly below" search
                height_x = height['x']
                x_tolerance = height_text_width / 2 + 50  # Increased tolerance for better pairing

                # First try: Find width directly below within X-range
                for width in widths:
                    # Check if width is below this height
                    if width['y'] <= height['y']:
                        continue

                    # Check if width is within X-range (directly below)
                    x_diff = abs(width['x'] - height_x)
                    if x_diff <= x_tolerance:
                        distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)
                        print(f"        Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): x_diff={x_diff:.0f}, distance={distance:.0f} - WITHIN X-RANGE")
                        if distance < best_distance:
                            best_distance = distance
                            best_width = width
                            print(f"          -> New best width!")
                    else:
                        print(f"        Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): x_diff={x_diff:.0f} - OUTSIDE X-RANGE")

                # Second try: If no width directly below in X-range, try width above within X-range
                if not best_width:
                    print(f"      Second try: Looking for width above within X-range...")
                    for width in widths:
                        # Check if width is above this height
                        if width['y'] >= height['y']:
                            continue

                        # Check if width is within X-range (directly above)
                        x_diff = abs(width['x'] - height_x)
                        print(f"        Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): x_diff={x_diff:.0f}, x_tolerance={x_tolerance:.0f}, in_range={x_diff <= x_tolerance + 1}")
                        if x_diff <= x_tolerance + 1:  # Add 1px tolerance for floating point precision
                            # Weight X-distance more heavily (3x) for side-by-side cabinet alignment
                            x_distance = (height['x'] - width['x']) * 3
                            y_distance = height['y'] - width['y']
                            distance = np.sqrt(x_distance**2 + y_distance**2)
                            print(f"          Distance={distance:.0f} (x_weighted), best_distance={best_distance:.0f}")
                            if distance < best_distance:
                                best_distance = distance
                                best_width = width
                                print(f"          -> New best width found!")

                # Third try: If no width directly above in X-range, try ANY width below
                if not best_width:
                    for width in widths:
                        # Check if width is below this height
                        if width['y'] <= height['y']:
                            continue

                        # No X-range check - just find closest below
                        distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)
                        if distance < best_distance:
                            best_distance = distance
                            best_width = width

                # Fourth try: If still no width below, find ANY width above
                if not best_width:
                    for width in widths:
                        # Only consider widths that are above this height
                        if width['y'] >= height['y']:
                            continue

                        distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)
                        if distance < best_distance:
                            best_distance = distance
                            best_width = width

                if best_width:  # Always pair with closest width, no max distance
                    search_type = "below" if best_width['y'] > height['y'] else "above"
                    print(f"    Height '{height['text']}' pairs with width '{best_width['text']}' {search_type} (distance: {best_distance:.0f}px)")
                    opening_data = {
                        'width': best_width['text'],
                        'height': height['text'],
                        'width_pos': (best_width['x'], best_width['y']),
                        'height_pos': (height['x'], height['y']),
                        'distance': best_distance
                    }
                    if 'notation' in height:
                        opening_data['notation'] = height['notation']
                    openings.append(opening_data)

    print(f"\n  Total openings paired: {len(openings)}")

    # Sort openings by X position (left to right) then Y position (top to bottom) for consistent numbering
    if len(openings) > 0:
        # Sort by X position (horizontal) first, then Y position (vertical) second
        # Use average X and Y positions from width and height measurements
        openings.sort(key=lambda o: (
            (o['width_pos'][0] + o['height_pos'][0]) / 2,  # Average X position (primary - left to right)
            (o['width_pos'][1] + o['height_pos'][1]) / 2   # Average Y position (secondary - top to bottom)
        ))
        print("  Sorted by X position (left to right), then Y position (top to bottom)")

    # Collect unpaired heights info for visualization
    unpaired_heights_info = []
    if unpaired_heights:
        for height in unpaired_heights:
            # Estimate text width for X-range calculation
            if 'bounds' in height and height['bounds']:
                height_text_width = height['bounds']['right'] - height['bounds']['left']
            else:
                height_text_width = len(height['text']) * 15  # Fallback estimate

            x_tolerance = height_text_width / 2 + 50  # Must match pairing logic
            unpaired_heights_info.append({
                'x': height['x'],
                'y': height['y'],
                'text': height['text'],
                'x_tolerance': x_tolerance
            })

    return openings, unpaired_heights_info

def add_fraction_to_measurement(measurement, fraction_to_add):
    """
    Add a fraction to a measurement string (e.g., "23 15/16" + "5/8" = "24 9/16")
    """
    import re
    from fractions import Fraction

    if not fraction_to_add:
        return measurement

    # Parse the measurement (e.g., "23 15/16" or "23" or "15/16")
    match = re.match(r'(\d+)?\s*(\d+/\d+)?', measurement.strip())
    if not match:
        return measurement

    whole_part = match.group(1)
    fraction_part = match.group(2)

    # Convert to Fraction
    total = Fraction(0)
    if whole_part:
        total += int(whole_part)
    if fraction_part:
        total += Fraction(fraction_part)

    # Parse the overlay fraction (e.g., "5/8 OL" -> "5/8")
    overlay_match = re.match(r'(\d+/\d+)', fraction_to_add.strip())
    if not overlay_match:
        return measurement

    overlay_frac = Fraction(overlay_match.group(1))

    # Add them together
    result = total + overlay_frac

    # Convert back to mixed number format
    whole = result.numerator // result.denominator
    remainder = result.numerator % result.denominator

    if remainder == 0:
        return str(whole)
    elif whole == 0:
        return f"{remainder}/{result.denominator}"
    else:
        return f"{whole} {remainder}/{result.denominator}"

def find_clear_position_for_marker(intersection_x, intersection_y, width_pos, height_pos, measurements_list, opening, image, existing_markers=None, overlay_info=None, marker_radius=35):
    """
    Find the best available clear position for the opening number marker.
    Analyzes available space and picks the closest spot to the intersection that fits.
    """
    # Collect all occupied regions
    occupied_regions = []

    # Add regions for all measurement text
    for meas in measurements_list:
        if 'bounds' in meas and meas['bounds']:
            bounds = meas['bounds']
            occupied_regions.append({
                'left': bounds['left'],  # No padding - exact bounds
                'right': bounds['right'],
                'top': bounds['top'],
                'bottom': bounds['bottom']
            })

    # Add regions for existing opening markers
    if existing_markers:
        for marker in existing_markers:
            marker_x, marker_y = marker['position']
            # Add circular marker region (use stored radius if available, otherwise default)
            existing_radius = marker.get('radius', 35)
            occupied_regions.append({
                'left': marker_x - existing_radius,
                'right': marker_x + existing_radius,
                'top': marker_y - existing_radius,
                'bottom': marker_y + existing_radius
            })

            # Add dimension text below marker
            dim_text = f"{marker['opening']['width']} W x {marker['opening']['height']} H"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.7, 2)
            occupied_regions.append({
                'left': marker_x - text_width//2 - 1,  # Minimal 1px padding
                'right': marker_x + text_width//2 + 1,
                'top': marker_y + 48,  # Text starts at +50, minimal padding
                'bottom': marker_y + 50 + text_height
            })

    # Calculate dimension text size for the marker
    dim_text = f"{opening['width']} W x {opening['height']} H"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.7, 2)

    # Start from the center of the opening (intersection point)
    # Then try small offsets within the opening area
    # Only move outside if necessary
    test_positions = [
        (intersection_x, intersection_y),  # Center of opening (preferred)
    ]

    # Scan in expanding circles around the intersection point
    # Start with small movements, prioritizing upward movement
    max_radius = 250  # Increased from 120 to allow more spread

    # First, try moving up pixel by pixel (prefer upward movement)
    for offset in range(1, 100):  # Increased from 50 to 100
        test_positions.append((intersection_x, intersection_y - offset))

    # Then try other directions pixel by pixel
    for offset in range(1, 100):  # Increased from 50 to 100
        test_positions.extend([
            (intersection_x + offset, intersection_y),  # Right
            (intersection_x - offset, intersection_y),  # Left
            (intersection_x, intersection_y + offset),  # Down (last priority)
        ])

    # Now scan in expanding squares for remaining positions
    for radius in range(5, max_radius, 5):
        # Generate points in a square perimeter
        # Top edge (left to right)
        for x in range(intersection_x - radius, intersection_x + radius + 1, 5):
            test_positions.append((x, intersection_y - radius))
        # Right edge (top to bottom)
        for y in range(intersection_y - radius + 5, intersection_y + radius, 5):
            test_positions.append((intersection_x + radius, y))
        # Bottom edge (right to left)
        for x in range(intersection_x + radius, intersection_x - radius - 1, -5):
            test_positions.append((x, intersection_y + radius))
        # Left edge (bottom to top)
        for y in range(intersection_y + radius - 5, intersection_y - radius, -5):
            test_positions.append((intersection_x - radius, y))

    # Find the best clear position closest to intersection
    best_position = None
    best_distance = float('inf')

    for test_x, test_y in test_positions:
        # Check if position is within image bounds
        if test_x - marker_radius < 0 or test_x + marker_radius >= image.shape[1]:
            continue
        if test_y - marker_radius < 0 or test_y + marker_radius + 50 + text_height >= image.shape[0]:
            continue

        # Calculate marker bounds (including text below)
        marker_left = test_x - max(marker_radius, text_width // 2)
        marker_right = test_x + max(marker_radius, text_width // 2)
        marker_top = test_y - marker_radius
        marker_bottom = test_y + marker_radius + 50 + text_height  # Account for circle + text below

        # Calculate overlap with occupied regions
        total_overlap = 0
        for region in occupied_regions:
            # Check for overlap
            if (marker_left < region['right'] and marker_right > region['left'] and
                marker_top < region['bottom'] and marker_bottom > region['top']):
                # Calculate overlap area
                overlap_x = min(marker_right, region['right']) - max(marker_left, region['left'])
                overlap_y = min(marker_bottom, region['bottom']) - max(marker_top, region['top'])
                total_overlap += overlap_x * overlap_y

        # If no overlap, check distance to intersection
        if total_overlap == 0:
            distance = np.sqrt((test_x - intersection_x)**2 + (test_y - intersection_y)**2)
            if distance < best_distance:
                best_distance = distance
                best_position = (test_x, test_y)
            # Use first clear position at this distance
            if distance == 0:  # At intersection
                break

    # If no clear position found, default to offset
    if best_position is None:
        best_position = (intersection_x + 80, intersection_y)

    return int(best_position[0]), int(best_position[1])

def create_visualization(image_path, groups, measurement_texts, measurement_logic=None, save_viz=True, opencv_regions=None, measurement_categories=None, measurements_list=None, show_rois=False, paired_openings=None, show_groups=False, show_opencv=False, show_line_rois=True, show_panel=True, show_pairing=True, show_classification=True, room_name=None, overlay_info=None, unpaired_heights_info=None, page_number=None, start_opening_number=1):
    """Create visualization showing groups and measurements side by side"""

    # Colors for different groups (BGR format - muted professional colors)
    COLORS = [
        (0, 0, 200),      # Dark Red
        (0, 100, 180),    # Dark Orange
        (180, 0, 0),      # Dark Blue
        (128, 0, 128),    # Purple
        (150, 0, 150),    # Dark Magenta
        (128, 64, 0),     # Dark blue
        (0, 128, 128),    # Dark olive/brown
        (100, 0, 100),    # Dark purple
        (0, 80, 160),     # Dark orange-red
        (80, 40, 120),    # Dark red-purple
        (150, 75, 0),     # Navy blue
        (75, 0, 130),     # Indigo
    ]

    # Convert Windows network paths to Unix-style for OpenCV
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    # Load original image
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Could not load image for visualization")
        return

    vis_image = image.copy()
    h, w = image.shape[:2]

    # Draw ROIs for line detection (part of classification visualization)
    # Line ROIs are shown when classification is enabled (unless explicitly disabled)
    if show_classification and show_line_rois and measurements_list and len(measurements_list) > 0:
        for i, meas in enumerate(measurements_list[:len(groups)]):  # Match measurements to groups
            if i >= len(groups):
                break

            # Get measurement position and bounds
            x = int(meas['position'][0])
            y = int(meas['position'][1])

            # Get bounds for text size estimation
            if 'bounds' in meas:
                bounds = meas['bounds']
                text_width = int(bounds['right'] - bounds['left'])
                text_height = int(bounds['bottom'] - bounds['top'])
                # Update x,y to be the actual center of the text bounds (same as detection logic)
                x = int((bounds['left'] + bounds['right']) / 2)
                y = int((bounds['top'] + bounds['bottom']) / 2)
            else:
                # Estimate
                text_height = 30
                text_width = len(meas.get('text', '')) * 15

            # Calculate ROIs EXACTLY same as in find_lines_near_measurement (UPDATED VERSION)
            # Horizontal ROIs (for WIDTH detection)
            # Use 0.75x text width but ensure minimum of 40 pixels for short text
            h_strip_extension = max(40, int(text_width * 0.75))

            # IMPROVED: Match the corrected H-ROI logic with proper gaps
            min_gap = 10  # Minimum 10px gap between text and H-ROI

            # Check if this measurement used rotation fallback
            viz_rotation_angle = meas.get('roi_rotation_angle', 0.0)

            # Use the ACTUAL ROI coordinates stored during detection
            if 'actual_h_left_roi' in meas:
                h_left_x1, h_left_y1, h_left_x2, h_left_y2 = meas['actual_h_left_roi']
            else:
                # Fallback to recalculation if not stored
                text_left_edge = int(x - text_width//2)
                h_left_x2 = max(0, text_left_edge - min_gap)
                h_left_x1 = max(0, h_left_x2 - h_strip_extension)
                h_left_y1 = int(y - text_height * 2.0)
                h_left_y2 = int(y + text_height * 2.0)

            if 'actual_h_right_roi' in meas:
                h_right_x1, h_right_y1, h_right_x2, h_right_y2 = meas['actual_h_right_roi']
            else:
                # Fallback to recalculation if not stored
                text_right_edge = int(x + text_width//2)
                h_right_x1 = min(w, text_right_edge + min_gap)
                h_right_x2 = min(w, h_right_x1 + h_strip_extension)
                h_right_y1 = h_left_y1
                h_right_y2 = h_left_y2

            # Add label if rotation was used
            if viz_rotation_angle != 0:
                rotation_failed = meas.get('rotation_failed', False)
                if rotation_failed:
                    rotation_label = f"ROI ROTATED {viz_rotation_angle:+.1f}° FAILED"
                    label_color = (0, 0, 255)  # Red for failed
                else:
                    rotation_label = f"ROI ROTATED {viz_rotation_angle:+.1f}° SUCCESS"
                    label_color = (0, 255, 0)  # Green for success

                label_x = x - 80
                label_y = y - text_height - 15
                # Draw label background
                cv2.rectangle(vis_image, (label_x - 5, label_y - 15), (label_x + 200, label_y + 5), (0, 0, 0), -1)
                cv2.rectangle(vis_image, (label_x - 5, label_y - 15), (label_x + 200, label_y + 5), label_color, 1)
                # Draw label text
                cv2.putText(vis_image, rotation_label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

            # Use the ACTUAL V-ROI coordinates stored during detection
            if 'actual_v_top_roi' in meas:
                v_top_x1, v_top_y1, v_top_x2, v_top_y2 = meas['actual_v_top_roi']
            else:
                # Fallback to recalculation if not stored
                v_strip_extension = int(text_height * 4)
                v_top_x1 = int(x - text_width//2)
                v_top_x2 = int(x + text_width//2)
                v_top_y1 = max(0, int(y - text_height//2 - v_strip_extension))
                v_top_y2 = int(y - text_height//2 - 5)

            if 'actual_v_bottom_roi' in meas:
                v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2 = meas['actual_v_bottom_roi']
            else:
                # Fallback to recalculation if not stored
                v_strip_extension = int(text_height * 4)
                v_bottom_x1 = v_top_x1 if 'actual_v_top_roi' in meas else int(x - text_width//2)
                v_bottom_x2 = v_top_x2 if 'actual_v_top_roi' in meas else int(x + text_width//2)
                v_bottom_y1 = int(y + text_height//2 + 5)
                v_bottom_y2 = min(h, int(y + text_height//2 + v_strip_extension))

            # Helper function to draw rotated rectangle
            def draw_rotated_rect(image, x1, y1, x2, y2, angle, color, center_point):
                """Draw a rotated rectangle given corners and rotation angle around center point"""
                if angle == 0:
                    # No rotation - draw normal rectangle with dotted lines
                    for px in range(x1, x2, 8):
                        cv2.line(image, (px, y1), (min(px+4, x2), y1), color, 1)
                        cv2.line(image, (px, y2), (min(px+4, x2), y2), color, 1)
                    for py in range(y1, y2, 8):
                        cv2.line(image, (x1, py), (x1, min(py+4, y2)), color, 1)
                        cv2.line(image, (x2, py), (x2, min(py+4, y2)), color, 1)
                else:
                    # Rotate the corners around the center point
                    cx, cy = center_point
                    angle_rad = np.radians(angle)
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)

                    # Original corners
                    corners = [
                        (x1, y1),  # Top-left
                        (x2, y1),  # Top-right
                        (x2, y2),  # Bottom-right
                        (x1, y2)   # Bottom-left
                    ]

                    # Rotate each corner around center
                    rotated_corners = []
                    for px, py in corners:
                        # Translate to origin
                        px_rel = px - cx
                        py_rel = py - cy
                        # Rotate
                        px_rot = px_rel * cos_a - py_rel * sin_a
                        py_rot = px_rel * sin_a + py_rel * cos_a
                        # Translate back
                        rotated_corners.append((int(px_rot + cx), int(py_rot + cy)))

                    # Draw the rotated rectangle with solid lines (easier than dotted for rotated)
                    pts = np.array(rotated_corners, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, color, 2)

            # Draw horizontal ROIs with dotted rectangles (for WIDTH detection) - RED
            # Left horizontal ROI
            if h_left_x2 > h_left_x1 and h_left_y2 > h_left_y1:
                draw_rotated_rect(vis_image, h_left_x1, h_left_y1, h_left_x2, h_left_y2,
                                 viz_rotation_angle, (0, 0, 200), (x, y))
                cv2.putText(vis_image, "W", (h_left_x1+2, h_left_y1-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)

            # Right horizontal ROI
            if h_right_x2 > h_right_x1 and h_right_y2 > h_right_y1:
                draw_rotated_rect(vis_image, h_right_x1, h_right_y1, h_right_x2, h_right_y2,
                                 viz_rotation_angle, (0, 0, 200), (x, y))
                cv2.putText(vis_image, "W", (h_right_x2-10, h_right_y1-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)

            # Draw vertical ROIs with dotted rectangles (for HEIGHT detection) - BLUE
            # Top vertical ROI
            if v_top_x2 > v_top_x1 and v_top_y2 > v_top_y1:
                draw_rotated_rect(vis_image, v_top_x1, v_top_y1, v_top_x2, v_top_y2,
                                 viz_rotation_angle, (200, 0, 0), (x, y))
                cv2.putText(vis_image, "H", (v_top_x1+2, v_top_y1+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 1)

            # Bottom vertical ROI
            if v_bottom_x2 > v_bottom_x1 and v_bottom_y2 > v_bottom_y1:
                draw_rotated_rect(vis_image, v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2,
                                 viz_rotation_angle, (200, 0, 0), (x, y))
                cv2.putText(vis_image, "H", (v_bottom_x1+2, v_bottom_y2-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 1)

    # Draw groups with different colors (just groups, no classification)
    if show_groups:
      for i, group in enumerate(groups):
        # Use cycling colors for groups (no classification here)
        color = COLORS[i % len(COLORS)]
        label_suffix = ""  # No classification labels in groups

        bounds = group['bounds']
        texts = group.get('texts', [])

        # Draw rectangle for group
        left = int(bounds['left'])
        right = int(bounds['right'])
        top = int(bounds['top'])
        bottom = int(bounds['bottom'])

        cv2.rectangle(vis_image, (left - 5, top - 5), (right + 5, bottom + 5), color, 3)

        # Draw group number with category
        group_label = f"G{i+1}{label_suffix}"
        label_x = max(10, left - 35)
        label_y = max(30, top - 10)
        cv2.putText(vis_image, group_label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw center point
        center = group['center']
        cv2.circle(vis_image, (int(center[0]), int(center[1])), 5, color, -1)

        # Draw zoom region as dashed rectangle
        # Calculate zoom bounds (using same padding as in verify_measurement_at_center_with_logic)
        padding = ZOOM_CONFIG['padding']  # Should be 150 based on config
        zoom_left = max(0, int(min(bounds['left'], bounds['right']) - padding))
        zoom_top = max(0, int(min(bounds['top'], bounds['bottom']) - padding))
        zoom_right = min(w, int(max(bounds['left'], bounds['right']) + padding))
        zoom_bottom = min(h, int(max(bounds['top'], bounds['bottom']) + padding))

        # Draw dashed rectangle for zoom region
        # Create a lighter version of the color for the zoom region
        lighter_color = tuple(min(255, c + 100) for c in color)

        # Draw dashed lines manually (OpenCV doesn't have built-in dashed lines)
        dash_length = 10
        gap_length = 5

        # Top edge
        x = zoom_left
        while x < zoom_right:
            x_end = min(x + dash_length, zoom_right)
            cv2.line(vis_image, (x, zoom_top), (x_end, zoom_top), lighter_color, 1)
            x += dash_length + gap_length

        # Bottom edge
        x = zoom_left
        while x < zoom_right:
            x_end = min(x + dash_length, zoom_right)
            cv2.line(vis_image, (x, zoom_bottom), (x_end, zoom_bottom), lighter_color, 1)
            x += dash_length + gap_length

        # Left edge
        y = zoom_top
        while y < zoom_bottom:
            y_end = min(y + dash_length, zoom_bottom)
            cv2.line(vis_image, (zoom_left, y), (zoom_left, y_end), lighter_color, 1)
            y += dash_length + gap_length

        # Right edge
        y = zoom_top
        while y < zoom_bottom:
            y_end = min(y + dash_length, zoom_bottom)
            cv2.line(vis_image, (zoom_right, y), (zoom_right, y_end), lighter_color, 1)
            y += dash_length + gap_length

    # Draw OpenCV additions if provided
    if show_opencv and opencv_regions:
        for region in opencv_regions:
            cx, cy = region['center']
            # Draw with thick red circle and "OCV+" label
            cv2.circle(vis_image, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            cv2.circle(vis_image, (int(cx), int(cy)), 8, (0, 0, 255), 2)
            cv2.putText(vis_image, "OCV+", (int(cx)-20, int(cy)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            # Special marking for potential left 23
            if 250 < cx < 400 and 700 < cy < 850:
                cv2.rectangle(vis_image, (int(cx)-15, int(cy)-15),
                            (int(cx)+15, int(cy)+15), (255, 0, 255), 2)

    # Create info panel if enabled (this is for grouping, not classification/pairing)
    if show_panel and show_groups:
        panel_width = 400  # Narrower panel
        panel_height = 150  # Fixed height for summary
        panel_y = h - panel_height - 20  # Position at bottom with 20px margin
        panel_x = 20  # Left margin

        # Create semi-transparent black background for panel
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0, vis_image)

        # Draw panel border
        cv2.rectangle(vis_image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                      (200, 200, 200), 2)

        # Add title
        y_pos = panel_y + 30
        x_pos = panel_x + 10
        cv2.putText(vis_image, "MEASUREMENT CLASSIFICATION", (x_pos, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 30

        # Show category counts if available and classification is enabled
        if show_classification and measurement_categories:
            width_count = sum(1 for cat in measurement_categories if cat == 'width')
            height_count = sum(1 for cat in measurement_categories if cat == 'height')
            unclass_count = sum(1 for cat in measurement_categories if cat == 'unclassified')

            cv2.putText(vis_image, f"Widths: {width_count}", (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis_image, f"Heights: {height_count}", (x_pos + 120, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if unclass_count > 0:
                cv2.putText(vis_image, f"Unclassified: {unclass_count}", (x_pos + 240, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            y_pos += 35

        # Show pairing summary if available and pairing is enabled
        if show_pairing and paired_openings:
            cv2.putText(vis_image, f"Cabinet Openings: {len(paired_openings)}", (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30

        # Simple line separator at bottom of panel
        cv2.line(vis_image, (panel_x + 10, panel_y + panel_height - 20),
                 (panel_x + panel_width - 10, panel_y + panel_height - 20), (200, 200, 200), 1)

    # Draw classification labels on measurements when classification is enabled
    if show_classification and measurement_categories and measurements_list:
        for i, (meas, category) in enumerate(zip(measurements_list, measurement_categories)):
            if i >= len(measurement_categories):
                break

            x, y = int(meas['position'][0]), int(meas['position'][1])

            # DEBUG: Draw OCR text with box below the actual text for comparison
            # DISABLED by default - set SHOW_DEBUG_TEXT_BOXES = True to enable
            SHOW_DEBUG_TEXT_BOXES = False  # Set to True to show debug text boxes and OCR text

            if SHOW_DEBUG_TEXT_BOXES and 'bounds' in meas and meas['bounds']:
                bounds = meas['bounds']
                text_left = int(bounds['left'])
                text_right = int(bounds['right'])
                text_top = int(bounds['top'])
                text_bottom = int(bounds['bottom'])
                text_width_px = text_right - text_left
                text_height_px = text_bottom - text_top

                # Draw bounding box around actual text (cyan color)
                cv2.rectangle(vis_image, (text_left, text_top), (text_right, text_bottom), (255, 255, 0), 2)

                # Get the RAW OCR text (not cleaned) - this is what the bounds were calculated from
                measurement_text = meas.get('raw_ocr_text', meas.get('text', ''))

                # Draw the OCR text below with same dimensions
                # Position it below the actual text with some spacing
                draw_y_offset = text_bottom + 50  # 50px below actual text

                # Draw box with same dimensions as calculated bounds
                draw_box_left = text_left
                draw_box_top = draw_y_offset
                draw_box_right = text_left + text_width_px
                draw_box_bottom = draw_y_offset + text_height_px

                # Draw the comparison box (cyan)
                cv2.rectangle(vis_image, (draw_box_left, draw_box_top), (draw_box_right, draw_box_bottom), (255, 255, 0), 2)

                # Draw the OCR text inside the box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                # Get text size to center it in the box
                (text_w, text_h), baseline = cv2.getTextSize(measurement_text, font, font_scale, thickness)

                # Center the text in the drawn box
                text_x = draw_box_left + (text_width_px - text_w) // 2
                text_y = draw_box_top + (text_height_px + text_h) // 2

                # Draw the text
                cv2.putText(vis_image, measurement_text, (text_x, text_y),
                           font, font_scale, (0, 255, 255), thickness)

            # Determine color based on category
            if category == 'width':
                color = (0, 0, 255)  # Red for WIDTH
                label = "WIDTH"
            elif category == 'height':
                color = (255, 0, 0)  # Blue for HEIGHT
                label = "HEIGHT"
            else:
                color = (128, 128, 128)  # Gray for UNCLASSIFIED
                label = "UNCLASS"

            # Find clear position for classification label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Collect avoid regions from measurement bounds
            avoid_regions = []
            if 'bounds' in meas and meas['bounds']:
                bounds = meas['bounds']
                avoid_regions.append({
                    'left': bounds['left'],
                    'right': bounds['right'],
                    'top': bounds['top'],
                    'bottom': bounds['bottom']
                })

            # Try different positions around the measurement
            test_offsets = [
                (15, -10),   # Upper right
                (-text_width - 15, -10),  # Upper left
                (15, 20),    # Lower right
                (-text_width - 15, 20),   # Lower left
                (0, -25),    # Above center
                (0, 30),     # Below center
            ]

            best_pos = (x + 12, y + 5)  # Default position
            min_overlap = float('inf')

            for offset_x, offset_y in test_offsets:
                label_x = x + offset_x
                label_y = y + offset_y

                # Check if label fits in image
                if label_x < 0 or label_x + text_width >= w:
                    continue
                if label_y - text_height < 0 or label_y >= h:
                    continue

                # Calculate label bounds
                label_left = label_x
                label_right = label_x + text_width
                label_top = label_y - text_height
                label_bottom = label_y

                # Check overlap with measurement bounds
                total_overlap = 0
                for region in avoid_regions:
                    if (label_left < region['right'] and label_right > region['left'] and
                        label_top < region['bottom'] and label_bottom > region['top']):
                        overlap_x = min(label_right, region['right']) - max(label_left, region['left'])
                        overlap_y = min(label_bottom, region['bottom']) - max(label_top, region['top'])
                        total_overlap += overlap_x * overlap_y

                if total_overlap < min_overlap:
                    min_overlap = total_overlap
                    best_pos = (label_x, label_y)
                    if total_overlap == 0:
                        break

            # Draw classification label at best position
            cv2.putText(vis_image, label, best_pos,
                       font, font_scale, color, thickness)

    # Draw X-range visualization for unpaired heights
    if False and unpaired_heights_info:  # Disabled X-range visualization
        print(f"Drawing X-range zones for {len(unpaired_heights_info)} unpaired heights")
        for height_info in unpaired_heights_info:
            # Only draw for 9 1/4 measurements
            if '9 1/4' not in height_info.get('text', ''):
                continue

            height_x = height_info['x']
            height_y = height_info['y']
            x_tolerance = height_info.get('x_tolerance', 50)

            # Calculate X-range boundaries
            x_min = max(0, int(height_x - x_tolerance))
            x_max = min(w, int(height_x + x_tolerance))

            # Draw semi-transparent vertical band showing X-range
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (x_min, 0), (x_max, h),
                         (255, 200, 100), -1)  # Light blue color
            cv2.addWeighted(overlay, 0.15, vis_image, 0.85, 0, vis_image)

            # Draw vertical boundary lines
            cv2.line(vis_image, (x_min, 0), (x_min, h), (255, 200, 100), 2)
            cv2.line(vis_image, (x_max, 0), (x_max, h), (255, 200, 100), 2)

            # Draw horizontal line at height position
            cv2.line(vis_image, (x_min, int(height_y)), (x_max, int(height_y)),
                    (0, 255, 255), 2)

            # Add label
            label = f"X-range: {x_tolerance:.0f}px"
            cv2.putText(vis_image, label,
                       (x_min + 5, int(height_y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(vis_image, label,
                       (x_min + 5, int(height_y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw X-range for the 16 5/8 width above (at 940, 810)
    if False:  # Disabled width visualization
        # For the specific width at (940, 810)
        width_x = 940
        width_y = 810
        # Estimate text width for "16 5/8"
        width_text_width = 75  # Based on actual OCR bounds from output
        x_tolerance_width = width_text_width / 2 + 50  # Same formula as heights

        # Calculate X-range boundaries for this width
        x_min_w = max(0, int(width_x - x_tolerance_width))
        x_max_w = min(w, int(width_x + x_tolerance_width))

        # Draw semi-transparent vertical band for width's X-range (different color)
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (x_min_w, 0), (x_max_w, h),
                     (100, 255, 100), -1)  # Light green color for width
        cv2.addWeighted(overlay, 0.15, vis_image, 0.85, 0, vis_image)

        # Draw vertical boundary lines for width
        cv2.line(vis_image, (x_min_w, 0), (x_min_w, h), (100, 255, 100), 2)
        cv2.line(vis_image, (x_max_w, 0), (x_max_w, h), (100, 255, 100), 2)

        # Draw horizontal line at width position
        cv2.line(vis_image, (x_min_w, int(width_y)), (x_max_w, int(width_y)),
                (0, 255, 0), 2)

        # Add label for width
        label_w = f"Width X-range: {x_tolerance_width:.0f}px"
        cv2.putText(vis_image, label_w,
                   (x_min_w + 5, int(width_y) + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis_image, label_w,
                   (x_min_w + 5, int(width_y) + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw paired openings if provided
    if show_pairing and paired_openings and len(paired_openings) > 0:
        print(f"Drawing {len(paired_openings)} paired openings")

        # Define colors for different openings (muted professional colors)
        opening_colors = [
            (0, 0, 200),     # Dark Red
            (180, 0, 0),     # Dark Blue
            (0, 150, 0),     # Dark Green
            (128, 0, 128),   # Purple
            (0, 140, 140),   # Teal
            (150, 75, 0)     # Navy
        ]

        # Track placed markers to avoid overlaps
        placed_markers = []

        for idx, opening in enumerate(paired_openings):
            color = opening_colors[idx % len(opening_colors)]

            # Get positions
            width_x, width_y = opening['width_pos']
            height_x, height_y = opening['height_pos']

            # Calculate intersection point:
            # Start from height position, travel parallel to width's H-lines toward width's X position
            width_hline_angle = opening.get('width_angle', 0)

            # Calculate horizontal distance to travel from height to width's X position
            x_diff = width_x - height_x

            # Use the width's H-line angle to calculate Y offset
            # For a line at angle θ from horizontal: tan(θ) = y/x
            # We travel parallel to the H-line from height position
            angle_rad = np.radians(width_hline_angle)

            # Calculate Y offset based on the H-line angle
            # tan(angle) = opposite/adjacent = y_offset/x_diff
            # y_offset = x_diff * tan(angle)
            if np.abs(x_diff) > 0.01:  # Avoid near-zero horizontal distance
                y_offset = x_diff * np.tan(angle_rad)
            else:
                y_offset = 0  # No horizontal movement

            intersection_x = int(width_x)
            intersection_y = int(height_y + y_offset)

            # Draw lines from intersection to measurements (thin lines)
            cv2.line(vis_image, (intersection_x, intersection_y),
                    (int(width_x), int(width_y)), color, 1, cv2.LINE_AA)
            cv2.line(vis_image, (intersection_x, intersection_y),
                    (int(height_x), int(height_y)), color, 1, cv2.LINE_AA)

            # Calculate opening number text and radius BEFORE finding marker position
            text = f"#{start_opening_number + idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2

            # Get text size for proper centering
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Calculate dynamic circle radius based on text size (with proportional padding)
            # Use 30% padding around the text for a balanced appearance
            padding_factor = 1.30
            radius = int(max(text_width, text_height) * padding_factor / 2)

            # Find clear position for marker that avoids overlapping text and other markers
            marker_x, marker_y = find_clear_position_for_marker(
                intersection_x, intersection_y,
                (width_x, width_y), (height_x, height_y),
                measurements_list if measurements_list else [],
                opening, vis_image,
                placed_markers,  # Pass existing markers to avoid
                overlay_info,  # Pass overlay info for dimension text
                radius  # Pass the calculated radius
            )

            # Draw white filled circle with colored outline for opening number
            cv2.circle(vis_image, (marker_x, marker_y), radius, (255, 255, 255), -1)  # White fill
            cv2.circle(vis_image, (marker_x, marker_y), radius, color, 3)  # Colored outline

            # Calculate centered position (accounting for baseline)
            text_x = marker_x - text_width // 2
            text_y = marker_y + text_height // 2 - baseline // 2

            # Draw opening number
            cv2.putText(vis_image, text,
                       (text_x, text_y),
                       font, font_scale, color, thickness)

            # Draw line connecting marker to intersection if offset
            if marker_x != intersection_x or marker_y != intersection_y:
                cv2.line(vis_image, (intersection_x, intersection_y),
                        (marker_x, marker_y), color, 1, cv2.LINE_AA)

            # Add dimension label below marker
            dim_text = f"{opening['width']} W x {opening['height']} H"
            has_notation = 'notation' in opening and opening['notation'] == 'NH'
            notation_text = "NO HINGES" if has_notation else None

            label_y = marker_y + 50

            # Calculate background size based on whether we have notation
            (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.7, 2)

            if has_notation:
                (notation_width, notation_height), _ = cv2.getTextSize(notation_text, font, 0.6, 2)
                total_width = max(text_width, notation_width)
                total_height = text_height + notation_height + 5  # 5px spacing between lines
            else:
                total_width = text_width
                total_height = text_height

            # Background for dimension text
            cv2.rectangle(vis_image,
                         (marker_x - total_width//2 - 3, label_y - text_height - 3),
                         (marker_x + total_width//2 + 3, label_y + (notation_height + 5 if has_notation else 0) + 3),
                         (255, 255, 255), -1)

            # Draw dimension text
            cv2.putText(vis_image, dim_text,
                       (marker_x - text_width//2, label_y),
                       font, 0.7, color, 2)

            # Draw notation below if present
            if has_notation:
                notation_y = label_y + text_height + 5
                cv2.putText(vis_image, notation_text,
                           (marker_x - notation_width//2, notation_y),
                           font, 0.6, color, 2)

            # Add this marker to the placed markers list for collision avoidance
            placed_markers.append({
                'position': (marker_x, marker_y),
                'opening': opening,
                'radius': radius
            })

        # Add opening list panel/legend
        if room_name or overlay_info:
            legend_height = 60 + (len(paired_openings) * 35)

            # Calculate required width based on text content
            title_text = f"{room_name} - Finish Sizes"
            if overlay_info:
                title_text = f"{room_name} - Finish Sizes (including {overlay_info})"

            # Get title width
            (title_width, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            # Calculate max width needed for opening specifications
            max_spec_width = 0
            for opening in paired_openings:
                if overlay_info:
                    finish_width = add_fraction_to_measurement(opening['width'], overlay_info)
                    finish_height = add_fraction_to_measurement(opening['height'], overlay_info)
                    spec_text = f"{finish_width} W x {finish_height} H"
                else:
                    spec_text = f"{opening['width']} W x {opening['height']} H"

                if 'notation' in opening and opening['notation'] == 'NH':
                    spec_text += " NO HINGES"

                (spec_width, _), _ = cv2.getTextSize(spec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                max_spec_width = max(max_spec_width, spec_width)

            # Legend width = max of (title width, spec width + circle space) + padding
            legend_width = max(title_width + 20, max_spec_width + 60) + 20

            legend_x = 10
            legend_y = h - legend_height - 20

            # White background with black border
            cv2.rectangle(vis_image, (legend_x, legend_y),
                         (legend_x + legend_width, legend_y + legend_height),
                         (255, 255, 255), -1)
            cv2.rectangle(vis_image, (legend_x, legend_y),
                         (legend_x + legend_width, legend_y + legend_height),
                         (0, 0, 0), 2)

            # Title
            title_y = legend_y + 30
            cv2.putText(vis_image, title_text,
                       (legend_x + 10, title_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Opening list
            for i, opening in enumerate(paired_openings):
                opening_y = title_y + 35 + (i * 35)
                color = opening_colors[i % len(opening_colors)]

                # Calculate text size first for dynamic circle sizing
                legend_text = f"#{start_opening_number + i}"
                legend_font = cv2.FONT_HERSHEY_SIMPLEX
                legend_font_scale = 0.5
                legend_thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(legend_text, legend_font, legend_font_scale, legend_thickness)

                # Calculate dynamic radius based on text size (with proportional padding)
                # Use 40% padding for legend (slightly tighter than main markers)
                legend_padding_factor = 1.40
                legend_radius = int(max(text_w, text_h) * legend_padding_factor / 2)

                # Draw circle with number (dynamic size for better fit)
                circle_center_x = legend_x + 25
                circle_center_y = opening_y - 5
                cv2.circle(vis_image, (circle_center_x, circle_center_y), legend_radius, (255, 255, 255), -1)
                cv2.circle(vis_image, (circle_center_x, circle_center_y), legend_radius, color, 2)
                text_x = circle_center_x - text_w // 2
                text_y = circle_center_y + text_h // 2 - baseline // 2

                cv2.putText(vis_image, legend_text,
                           (text_x, text_y),
                           legend_font, legend_font_scale, color, legend_thickness)

                # Opening specification (calculate finish sizes with overlay)
                if overlay_info:
                    finish_width = add_fraction_to_measurement(opening['width'], overlay_info)
                    finish_height = add_fraction_to_measurement(opening['height'], overlay_info)
                    spec_text = f"{finish_width} W x {finish_height} H"
                else:
                    spec_text = f"{opening['width']} W x {opening['height']} H"

                # Add NO HINGES notation if present
                if 'notation' in opening and opening['notation'] == 'NH':
                    spec_text += " NO HINGES"

                cv2.putText(vis_image, spec_text,
                           (legend_x + 50, opening_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


    # No need to combine images since panel is overlaid
    combined = vis_image

    # Add page number and timestamp at bottom
    from datetime import datetime
    timestamp = datetime.now().strftime("%m-%d-%Y %I:%M:%S %p Central")

    # Extract page number from filename if not provided
    if page_number is None:
        import re
        filename = os.path.basename(image_path)
        match = re.search(r'page_(\d+)', filename)
        if match:
            page_number = int(match.group(1))

    # Create footer text
    if page_number is not None:
        footer_text = f"Page {page_number} - {timestamp}"
    else:
        footer_text = timestamp

    # Position at bottom right
    footer_font = cv2.FONT_HERSHEY_SIMPLEX
    footer_scale = 0.5
    footer_thickness = 1
    (footer_w, footer_h), footer_baseline = cv2.getTextSize(footer_text, footer_font, footer_scale, footer_thickness)

    footer_x = w - footer_w - 10
    footer_y = h - 10

    # Add white background behind text
    cv2.rectangle(combined, (footer_x - 5, footer_y - footer_h - 5),
                 (footer_x + footer_w + 5, footer_y + footer_baseline + 5),
                 (255, 255, 255), -1)

    # Add text
    cv2.putText(combined, footer_text,
               (footer_x, footer_y),
               footer_font, footer_scale, (0, 0, 0), footer_thickness)

    # Save visualization
    if save_viz:
        output_path = image_path.replace('.png', '_test_viz.png')
        cv2.imwrite(output_path, combined)
        # Convert to Windows path for display if needed
        display_path = output_path.replace('/', '\\') if output_path.startswith('//') else output_path
        print(f"\n[SAVED] Visualization: {display_path}")

    return combined

def main(start_opening_number=1):
    if len(sys.argv) < 2:
        print("Usage: python measurement_detector_test.py <image_path> [options]")
        print("\nVisualization options:")
        print("  --viz          : Create visualization (required for other viz options)")
        print("  --debug        : Save debug images showing zoom regions")
        print("  --show-groups  : Show group boxes and zoom regions")
        print("  --show-opencv  : Show OpenCV detected regions")
        print("  --no-rois      : Hide line detection ROIs (shown by default)")
        print("  --no-panel     : Hide the info panel")
        print("  --no-pairing   : Hide pairing visualization")
        print("  --no-class     : Hide classification visualization")
        print("  --minimal      : Show only classification and pairing")
        print("\nOpening number option:")
        print("  --start-num N  : Start opening numbers at N (default: 1)")
        sys.exit(1)

    image_path = sys.argv[1]

    # Check for --start-num option
    if '--start-num' in sys.argv:
        idx = sys.argv.index('--start-num')
        if idx + 1 < len(sys.argv):
            start_opening_number = int(sys.argv[idx + 1])
    # Always create visualization by default (use --no-viz to disable)
    create_viz = '--no-viz' not in sys.argv
    save_debug = '--debug' in sys.argv or '--save-debug' in sys.argv

    # Visualization flags
    show_groups = '--show-groups' in sys.argv
    show_opencv = '--show-opencv' in sys.argv
    show_line_rois = '--show-rois' in sys.argv  # ROIs disabled by default
    show_panel = '--no-panel' not in sys.argv
    show_pairing = '--no-pairing' not in sys.argv
    show_classification = '--no-class' not in sys.argv

    # Minimal mode: only pairing and classification (with ROIs)
    if '--minimal' in sys.argv:
        show_groups = False
        show_opencv = False
        # show_line_rois stays True (part of classification)

    api_key = os.environ.get('GOOGLE_VISION_API_KEY')

    if not api_key:
        print("[ERROR] Set GOOGLE_VISION_API_KEY environment variable")
        sys.exit(1)

    print("=" * 80)
    print("MEASUREMENT DETECTOR - NEW ALGORITHM TEST")
    print("=" * 80)

    # Phase 1: Find interest areas
    interest_areas, room_name, overlay_info, opencv_regions = find_interest_areas(image_path, api_key)

    # Merge close centers
    merged_areas = merge_close_centers(interest_areas)

    # Phase 2: Verify each unique center
    print("\n=== PHASE 2: Verifying Measurements ===")
    measurements = []
    measurement_texts = []  # For visualization
    measurement_logic = []  # Track any logic/corrections applied

    for i, area in enumerate(merged_areas):
        center = area['center']
        bounds = area['bounds']
        texts = area['texts']

        print(f"\n{i+1}. Center ({center[0]:.0f}, {center[1]:.0f}), texts: {' '.join(texts[:3])}...")

        result, logic = verify_measurement_at_center_with_logic(image_path, center, bounds, texts, api_key, i+1, save_debug=save_debug)

        # Handle both old format (string) and new format (tuple with notation and raw text)
        special_notation = None
        raw_text = None
        if isinstance(result, tuple):
            if len(result) == 4:
                measurement, actual_bounds, special_notation, raw_text = result
            elif len(result) == 3:
                measurement, actual_bounds, special_notation = result
            else:
                measurement, actual_bounds = result
        else:
            measurement = result
            actual_bounds = None

        if measurement:
            print(f"  → Measurement: '{measurement}'")

            # Use actual OCR bounds if available, otherwise estimate
            if actual_bounds:
                # Use the actual bounds from OCR
                updated_bounds = actual_bounds
                print(f"  Using actual OCR bounds: width={updated_bounds['right'] - updated_bounds['left']:.0f}")
            else:
                # Fallback to estimation (shouldn't happen with updated function)
                if bounds and texts and len(texts) > 0:
                    original_text_length = sum(len(t) for t in texts)
                    original_width = bounds['right'] - bounds['left']

                    if original_text_length > 0:
                        char_width = original_width / original_text_length
                    else:
                        char_width = 15  # Default estimate
                else:
                    char_width = 15  # Default estimate

                # Calculate new width based on actual measurement
                new_text_width = len(measurement) * char_width

                # Ensure minimum width for proper line detection (at least 60 pixels)
                min_width = 60
                new_text_width = max(new_text_width, min_width)

                # Update bounds to reflect actual measurement width
                center_x = center[0]
                updated_bounds = {
                    'left': center_x - new_text_width / 2,
                    'right': center_x + new_text_width / 2,
                    'top': bounds['top'],
                    'bottom': bounds['bottom']
                }

            measurement_data = {
                'text': measurement,
                'position': center,
                'bounds': updated_bounds
            }
            if special_notation:
                measurement_data['notation'] = special_notation
            if raw_text:
                measurement_data['raw_ocr_text'] = raw_text
            measurements.append(measurement_data)
            measurement_texts.append(measurement)
            measurement_logic.append(logic)
        else:
            print(f"  → No measurement found")
            measurement_texts.append(None)
            measurement_logic.append("")

    # Phase 3: Categorize measurements
    measurement_categories = None
    classified = None
    if measurements:
        print("\n=== PHASE 3: Categorizing Measurements ===")
        print("Finding dimension lines near each measurement...")

        # Convert Windows network paths to Unix-style for OpenCV
        img_path = image_path
        if img_path.startswith('\\\\'):
            img_path = img_path.replace('\\', '/')

        # Load image for line detection
        image = cv2.imread(img_path)
        if image is not None:
            classified, measurement_categories = classify_measurements_by_lines(image, measurements)

            print(f"\nCategorization Results:")
            print(f"  WIDTH measurements: {len(classified['width'])}")
            for w in classified['width']:
                print(f"    - {w}")
            print(f"  HEIGHT measurements: {len(classified['height'])}")
            for h in classified['height']:
                print(f"    - {h}")
            if classified['unclassified']:
                print(f"  UNCLASSIFIED: {len(classified['unclassified'])}")
                for u in classified['unclassified']:
                    print(f"    - {u}")

    # Phase 4: Pair measurements into openings
    paired_openings = []
    unpaired_heights_info = []
    if measurement_categories and measurements:
        print("\n=== PHASE 4: Pairing Measurements into Cabinet Openings ===")
        paired_openings, unpaired_heights_info = pair_measurements_by_proximity(measurement_categories, measurements)

        if paired_openings:
            print("\nCABINET OPENING SPECIFICATIONS:")
            print("-" * 60)
            for i, opening in enumerate(paired_openings, start_opening_number):
                print(f"Opening {i}: {opening['width']} W × {opening['height']} H")
                print(f"  Width at: ({opening['width_pos'][0]:.0f}, {opening['width_pos'][1]:.0f})")
                print(f"  Height at: ({opening['height_pos'][0]:.0f}, {opening['height_pos'][1]:.0f})")
                print(f"  Pairing distance: {opening['distance']:.0f}px")
        else:
            print("\nNo openings could be paired from the measurements")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("-" * 60)

    if room_name:
        print(f"Room: {room_name}")
    if overlay_info:
        print(f"Overlay: {overlay_info}")

    # Count measurements
    from collections import Counter
    counts = Counter(m['text'] for m in measurements)

    print(f"\nTOTAL MEASUREMENTS: {len(measurements)}")

    # Show categorized counts if available
    if classified:
        print(f"  Widths: {len(classified['width'])}")
        print(f"  Heights: {len(classified['height'])}")
        print(f"  Unclassified: {len(classified['unclassified'])}")

    print("\nMeasurement Counts:")
    for meas, count in sorted(counts.items()):
        # Find category for this measurement
        category = ""
        if classified:
            if meas in classified['width']:
                category = " [WIDTH]"
            elif meas in classified['height']:
                category = " [HEIGHT]"
            else:
                category = " [UNCLASS]"
        print(f"  {count}x {meas}{category}")

    # Create visualization if requested
    if create_viz:
        # Pass the actual measurements list so ROIs can be drawn
        create_visualization(image_path, merged_areas, measurement_texts, measurement_logic,
                           save_viz=True, opencv_regions=opencv_regions,
                           measurement_categories=measurement_categories,
                           measurements_list=measurements,
                           paired_openings=paired_openings,
                           show_groups=show_groups,
                           show_opencv=show_opencv,
                           show_line_rois=show_line_rois,
                           show_panel=show_panel,
                           show_pairing=show_pairing,
                           show_classification=show_classification,
                           room_name=room_name,
                           overlay_info=overlay_info,
                           unpaired_heights_info=unpaired_heights_info,
                           start_opening_number=start_opening_number)

    # Save pairing results to JSON
    if paired_openings:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(os.path.abspath(image_path))

        results_data = {
            'room_name': room_name if room_name else "",
            'overlay_info': overlay_info if overlay_info else "",
            'total_measurements': len(measurements),
            'total_openings': len(paired_openings),
            'measurements': {
                'widths': classified['width'] if classified else [],
                'heights': classified['height'] if classified else [],
                'unclassified': classified['unclassified'] if classified else []
            },
            'openings': [
                {
                    'number': i,
                    'width': opening['width'],
                    'height': opening['height'],
                    'specification': f"{opening['width']} W × {opening['height']} H" +
                                   (" NO HINGES" if opening.get('notation') == 'NH' else ""),
                    'width_position': opening['width_pos'],
                    'height_position': opening['height_pos'],
                    'pairing_distance': opening['distance'],
                    **({'notation': opening['notation']} if 'notation' in opening else {})
                }
                for i, opening in enumerate(paired_openings, start_opening_number)
            ]
        }

        output_json = os.path.join(output_dir, f"{base_name}_cabinet_openings.json")
        with open(output_json, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n[SAVED] Cabinet openings data: {output_json}")

if __name__ == "__main__":
    main()