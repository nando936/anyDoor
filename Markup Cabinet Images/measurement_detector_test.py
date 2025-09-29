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
    r'(KITCHEN)',
    r'(LAUNDRY)',
    r'(PANTRY)',
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
        # Clean up text
        if text in ['.', ',', '-', '+']:  # Skip punctuation
            continue
        vertices = ann['boundingPoly']['vertices']
        x = sum(v.get('x', 0) for v in vertices) / 4
        y = sum(v.get('y', 0) for v in vertices) / 4
        individual_items.append({
            'text': text,
            'x': x,
            'y': y
        })

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
            'components': group
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
        print(f"  Chose closest: '{measurement_value}'")

        # Calculate actual bounds from the components in the zoomed image
        # Then scale back to original image coordinates
        if 'components' in best_meas and best_meas['components']:
            # Find min/max bounds from all components
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')

            # Get bounds from individual annotations (which have boundingPoly)
            for comp in best_meas['components']:
                # We need to get bounds from the original annotations
                # Find matching annotation by text and position
                for ann in annotations[1:]:  # Skip full text
                    if ann['description'].strip() == comp['text']:
                        ann_x = sum(v.get('x', 0) for v in ann['boundingPoly']['vertices']) / 4
                        ann_y = sum(v.get('y', 0) for v in ann['boundingPoly']['vertices']) / 4
                        # Check if this is the same annotation (by position)
                        if abs(ann_x - comp['x']) < 5 and abs(ann_y - comp['y']) < 5:
                            vertices = ann['boundingPoly']['vertices']
                            for v in vertices:
                                x_coord = v.get('x', 0)
                                y_coord = v.get('y', 0)
                                min_x = min(min_x, x_coord)
                                max_x = max(max_x, x_coord)
                                min_y = min(min_y, y_coord)
                                max_y = max(max_y, y_coord)
                            break

            # Scale bounds back to original image coordinates
            # The bounds are in zoomed image coords, need to unzoom and uncrop
            actual_bounds = {
                'left': crop_x1 + (min_x / zoom_factor),
                'right': crop_x1 + (max_x / zoom_factor),
                'top': crop_y1 + (min_y / zoom_factor),
                'bottom': crop_y1 + (max_y / zoom_factor)
            }

            print(f"  Actual OCR bounds: left={actual_bounds['left']:.0f}, right={actual_bounds['right']:.0f}, top={actual_bounds['top']:.0f}, bottom={actual_bounds['bottom']:.0f}")
            print(f"  Width: {actual_bounds['right'] - actual_bounds['left']:.0f}, Height: {actual_bounds['bottom'] - actual_bounds['top']:.0f}")
        else:
            # Fallback if no components (shouldn't happen)
            actual_bounds = None

        if len(valid_measurements) > 1:
            logic_steps.append(f"Position-grouped measurements: {[vm.get('cleaned_text', vm['text']) for vm in valid_measurements]}")
            logic_steps.append(f"Chose closest to center: '{measurement_value}'")
        else:
            logic_steps.append(f"Found measurement: '{measurement_value}'")

        # Return tuple of (text, bounds, logic)
        return (measurement_value, actual_bounds), " | ".join(logic_steps)

    return (None, None), "No measurements found"

def find_lines_near_measurement(image, measurement, save_roi_debug=False):
    """Find lines near a specific measurement position that match the text color"""
    x = int(measurement['position'][0])
    y = int(measurement['position'][1])

    # Enable debug for troubleshooting
    DEBUG_MODE = True  # Force debug mode to see what's happening

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

    # Left horizontal ROI
    h_left_x1 = max(0, int(x - text_width//2 - h_strip_extension))
    h_left_x2 = int(x - text_width//2 - 5)
    h_left_y1 = int(y - text_height//2)
    h_left_y2 = int(y + text_height//2)

    # Right horizontal ROI
    h_right_x1 = int(x + text_width//2 + 5)
    h_right_x2 = min(image.shape[1], int(x + text_width//2 + h_strip_extension))
    h_right_y1 = h_left_y1
    h_right_y2 = h_left_y2

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

        # Convert to grayscale
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image

        # Detect edges
        edges = cv2.Canny(gray, 30, 100)

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

        return converging_pairs > 0

    # Helper function to detect lines in an ROI
    def detect_lines_in_roi(roi_image, roi_x_offset, roi_y_offset):
        if roi_image.size == 0:
            return []

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image
        edges = cv2.Canny(gray, 30, 100)

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

    # Search for horizontal lines
    if h_left_x2 > h_left_x1 and h_left_y2 > h_left_y1:
        h_left_roi = image[h_left_y1:h_left_y2, h_left_x1:h_left_x2]
        left_h_lines = detect_lines_in_roi(h_left_roi, h_left_x1, h_left_y1)
        print(f"      Left H-ROI: shape={h_left_roi.shape}, found {len(left_h_lines)} lines")
    else:
        left_h_lines = []
        print(f"      Left H-ROI: Invalid bounds")

    if h_right_x2 > h_right_x1 and h_right_y2 > h_right_y1:
        h_right_roi = image[h_right_y1:h_right_y2, h_right_x1:h_right_x2]
        right_h_lines = detect_lines_in_roi(h_right_roi, h_right_x1, h_right_y1)
        print(f"      Right H-ROI: shape={h_right_roi.shape}, found {len(right_h_lines)} lines")
    else:
        right_h_lines = []
        print(f"      Right H-ROI: Invalid bounds")

    # Filter for horizontal lines (more tolerant angles)
    for line in left_h_lines + right_h_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
        # More tolerant: 0-35° or 145-180° for "generally horizontal"
        if angle < 35 or angle > 145:
            horizontal_lines.append({
                'coords': (lx1, ly1, lx2, ly2),
                'distance': abs(y - (ly1 + ly2) / 2),
                'type': 'horizontal_line'
            })

    if horizontal_lines:
        print(f"      Found {len(horizontal_lines)} horizontal line candidates")

    # Search for vertical lines and arrows
    has_up_arrow = False
    has_down_arrow = False

    if v_top_x2 > v_top_x1 and v_top_y2 > v_top_y1:
        v_top_roi = image[v_top_y1:v_top_y2, v_top_x1:v_top_x2]
        top_v_lines = detect_lines_in_roi(v_top_roi, v_top_x1, v_top_y1)
        has_up_arrow = detect_arrow_in_roi(v_top_roi, 'up')
        print(f"      Top V-ROI: shape={v_top_roi.shape}, found {len(top_v_lines)} lines, up-arrow={has_up_arrow}")
    else:
        top_v_lines = []
        print(f"      Top V-ROI: Invalid bounds")

    if v_bottom_x2 > v_bottom_x1 and v_bottom_y2 > v_bottom_y1:
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
    for line in top_v_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
        # Debug angle detection
        if DEBUG_MODE and measurement.get('text') == '5 1/2':
            x_dist = abs(x - (lx1 + lx2) / 2)
            print(f"        Top line: angle={angle:.1f}°, x_dist={x_dist:.1f}, coords=({lx1},{ly1})-({lx2},{ly2})")
        # More tolerant: 55-125° for "generally vertical"
        if 55 < angle < 125:
            x_distance = abs(x - (lx1 + lx2) / 2)
            # Keep the distance check but make it less strict
            if x_distance > 2:  # Reduced from 5 to 2
                vertical_lines.append({
                    'coords': (lx1, ly1, lx2, ly2),
                    'distance': x_distance,
                    'type': 'vertical_line',
                    'source': 'top'  # Track that this came from top ROI
                })

    for line in bottom_v_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
        # More tolerant: 55-125° for "generally vertical"
        if 55 < angle < 125:
            x_distance = abs(x - (lx1 + lx2) / 2)
            # Keep the distance check but make it less strict
            if x_distance > 2:  # Reduced from 5 to 2
                vertical_lines.append({
                    'coords': (lx1, ly1, lx2, ly2),
                    'distance': x_distance,
                    'type': 'vertical_line',
                    'source': 'bottom'  # Track that this came from bottom ROI
                })

    if vertical_lines:
        print(f"      Found {len(vertical_lines)} vertical line candidates")

    # NEW LOGIC: Sequential check requiring HORIZONTAL lines on BOTH sides for WIDTH,
    # or VERTICAL lines on BOTH sides for HEIGHT

    # Step 1: Check for WIDTH - must have HORIZONTAL lines on BOTH left AND right
    # horizontal_lines already contains only horizontal lines (angle filtered)
    # Check which side of the text center each horizontal line is on
    left_h_lines = [l for l in horizontal_lines if l['coords'][0] < x and l['coords'][2] < x]  # Both endpoints left of center
    right_h_lines = [l for l in horizontal_lines if l['coords'][0] > x and l['coords'][2] > x]  # Both endpoints right of center

    has_left_horizontal = len(left_h_lines) > 0
    has_right_horizontal = len(right_h_lines) > 0

    if has_left_horizontal and has_right_horizontal:
        # Found HORIZONTAL lines on BOTH left and right - classify as WIDTH and stop
        best_h = min(horizontal_lines, key=lambda l: l['distance'])
        print(f"      Found HORIZONTAL lines on BOTH left and right → WIDTH")
        return {
            'line': best_h['coords'],
            'orientation': 'horizontal_line',
            'distance': best_h['distance']
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
        return {
            'line': best_v['coords'],
            'orientation': 'vertical_line',
            'distance': best_v['distance']
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

    # Step 3: Neither condition met - UNCLASSIFIED
    print(f"      No lines on both sides (L-horiz:{has_left_horizontal} R-horiz:{has_right_horizontal} T-vert:{has_top_vertical} B-vert:{has_bottom_vertical}) → UNCLASSIFIED")
    return None

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

        if line_info:
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
            print(f"    No dimension lines found nearby")
            classified['unclassified'].append(meas['text'])
            measurement_categories.append('unclassified')
            print(f"    → Classified as UNCLASSIFIED")

    return classified, measurement_categories

def pair_measurements_by_proximity(classified_measurements, all_measurements):
    """
    Pair width and height measurements based on proximity to form cabinet openings
    Returns list of paired openings
    """
    print("\n=== PAIRING MEASUREMENTS INTO OPENINGS ===")

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
                    'bounds': meas.get('bounds', None)
                })
            elif category == 'height':
                heights.append({
                    'text': meas['text'],
                    'x': meas['position'][0],
                    'y': meas['position'][1],
                    'bounds': meas.get('bounds', None)
                })

    if not widths or not heights:
        print("  Cannot pair - need both widths and heights")
        return []

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
                openings.append({
                    'width': best_width['text'],
                    'height': height['text'],
                    'width_pos': (best_width['x'], best_width['y']),
                    'height_pos': (height['x'], height['y']),
                    'distance': best_distance
                })
                # Track by position, not text
                height_id = (height['x'], height['y'])
                used_heights.add(height_id)

    else:
        # Side-by-side: each width finds closest height above it
        print("  → Each width will find closest height above it")

        for width in widths:
            best_height = None
            best_distance = float('inf')

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

            if best_height:
                print(f"\n  Width '{width['text']}' pairs with height '{best_height['text']}'")
                print(f"    Distance: {best_distance:.0f}px")
                openings.append({
                    'width': width['text'],
                    'height': best_height['text'],
                    'width_pos': (width['x'], width['y']),
                    'height_pos': (best_height['x'], best_height['y']),
                    'distance': best_distance
                })
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
                x_tolerance = height_text_width / 2 + 20  # Add some tolerance

                # First try: Find width directly below within X-range
                for width in widths:
                    # Check if width is below this height
                    if width['y'] <= height['y']:
                        continue

                    # Check if width is within X-range (directly below)
                    x_diff = abs(width['x'] - height_x)
                    if x_diff <= x_tolerance:
                        distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)
                        if distance < best_distance:
                            best_distance = distance
                            best_width = width

                # Second try: If no width directly below in X-range, try ANY width below
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

                # Third try: If still no width below, find closest width above
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
                    openings.append({
                        'width': best_width['text'],
                        'height': height['text'],
                        'width_pos': (best_width['x'], best_width['y']),
                        'height_pos': (height['x'], height['y']),
                        'distance': best_distance
                    })

    print(f"\n  Total openings paired: {len(openings)}")

    # Sort openings by Y position (top to bottom) for consistent numbering
    # For side-by-side layouts, use height Y as primary sort key
    # For stacked layouts, use the average Y position
    if len(openings) > 0:
        # Check if this is a side-by-side layout (single width, multiple heights)
        unique_widths = len(set(o['width_pos'] for o in openings))
        if unique_widths == 1:
            # Side-by-side: sort primarily by height Y position
            openings.sort(key=lambda o: o['height_pos'][1])
            print("  Sorted by height Y position (side-by-side layout)")
        else:
            # Stacked or mixed: use average Y position
            openings.sort(key=lambda o: (o['height_pos'][1] + o['width_pos'][1]) / 2)
            print("  Sorted by average Y position (stacked/mixed layout)")

    return openings

def find_clear_position_for_marker(intersection_x, intersection_y, width_pos, height_pos, measurements_list, opening, image, existing_markers=None, overlay_info=None):
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
            # Add circular marker region
            marker_radius = 35  # Exact radius, no padding
            occupied_regions.append({
                'left': marker_x - marker_radius,
                'right': marker_x + marker_radius,
                'top': marker_y - marker_radius,
                'bottom': marker_y + marker_radius
            })

            # Add dimension text below marker (add overlay if present)
            ol_text = f" + {overlay_info}" if overlay_info else ""
            dim_text = f"{marker['opening']['width']}{ol_text} W x {marker['opening']['height']}{ol_text} H"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.7, 2)
            occupied_regions.append({
                'left': marker_x - text_width//2 - 1,  # Minimal 1px padding
                'right': marker_x + text_width//2 + 1,
                'top': marker_y + 48,  # Text starts at +50, minimal padding
                'bottom': marker_y + 50 + text_height
            })

    # Calculate dimension text size for the marker (add overlay if present)
    ol_text = f" + {overlay_info}" if overlay_info else ""
    dim_text = f"{opening['width']}{ol_text} W x {opening['height']}{ol_text} H"
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
    max_radius = 120

    # First, try moving up pixel by pixel (prefer upward movement)
    for offset in range(1, 50):  # Check every single pixel upward
        test_positions.append((intersection_x, intersection_y - offset))

    # Then try other directions pixel by pixel
    for offset in range(1, 50):
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
        if test_x - 35 < 0 or test_x + 35 >= image.shape[1]:
            continue
        if test_y - 35 < 0 or test_y + 70 >= image.shape[0]:
            continue

        # Calculate marker bounds (including text below)
        marker_left = test_x - max(35, text_width // 2)
        marker_right = test_x + max(35, text_width // 2)
        marker_top = test_y - 35
        marker_bottom = test_y + 50 + text_height  # Account for text below

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

def create_visualization(image_path, groups, measurement_texts, measurement_logic=None, save_viz=True, opencv_regions=None, measurement_categories=None, measurements_list=None, show_rois=False, paired_openings=None, show_groups=False, show_opencv=False, show_line_rois=True, show_panel=True, show_pairing=True, show_classification=True, room_name=None, overlay_info=None):
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
            else:
                # Estimate
                text_height = 30
                text_width = len(meas.get('text', '')) * 15

            # Calculate ROIs same as in find_lines_near_measurement
            # Horizontal ROIs (for WIDTH detection)
            # Use 0.75x text width but ensure minimum of 40 pixels for short text
            h_strip_extension = max(40, int(text_width * 0.75))

            h_left_x1 = max(0, int(x - text_width//2 - h_strip_extension))
            h_left_x2 = int(x - text_width//2 - 5)
            h_left_y1 = int(y - text_height//2)
            h_left_y2 = int(y + text_height//2)

            h_right_x1 = int(x + text_width//2 + 5)
            h_right_x2 = min(w, int(x + text_width//2 + h_strip_extension))
            h_right_y1 = h_left_y1
            h_right_y2 = h_left_y2

            # Vertical ROIs (for HEIGHT detection)
            v_strip_extension = int(text_height * 4)  # Using the taller setting

            # Use full text width for vertical ROIs
            v_top_x1 = int(x - text_width//2)
            v_top_x2 = int(x + text_width//2)
            v_top_y1 = max(0, int(y - text_height//2 - v_strip_extension))
            v_top_y2 = int(y - text_height//2 - 5)

            v_bottom_x1 = v_top_x1
            v_bottom_x2 = v_top_x2
            v_bottom_y1 = int(y + text_height//2 + 5)
            v_bottom_y2 = min(h, int(y + text_height//2 + v_strip_extension))

            # Draw horizontal ROIs with dotted rectangles (for WIDTH detection) - RED
            # Left horizontal ROI
            if h_left_x2 > h_left_x1 and h_left_y2 > h_left_y1:
                # Draw dotted rectangle manually
                for px in range(h_left_x1, h_left_x2, 8):  # Dotted line on top and bottom
                    cv2.line(vis_image, (px, h_left_y1), (min(px+4, h_left_x2), h_left_y1), (0, 0, 200), 1)
                    cv2.line(vis_image, (px, h_left_y2), (min(px+4, h_left_x2), h_left_y2), (0, 0, 200), 1)
                for py in range(h_left_y1, h_left_y2, 8):  # Dotted line on sides
                    cv2.line(vis_image, (h_left_x1, py), (h_left_x1, min(py+4, h_left_y2)), (0, 0, 200), 1)
                    cv2.line(vis_image, (h_left_x2, py), (h_left_x2, min(py+4, h_left_y2)), (0, 0, 200), 1)
                # Label
                cv2.putText(vis_image, "W", (h_left_x1+2, h_left_y1-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)

            # Right horizontal ROI
            if h_right_x2 > h_right_x1 and h_right_y2 > h_right_y1:
                for px in range(h_right_x1, h_right_x2, 8):
                    cv2.line(vis_image, (px, h_right_y1), (min(px+4, h_right_x2), h_right_y1), (0, 0, 200), 1)
                    cv2.line(vis_image, (px, h_right_y2), (min(px+4, h_right_x2), h_right_y2), (0, 0, 200), 1)
                for py in range(h_right_y1, h_right_y2, 8):
                    cv2.line(vis_image, (h_right_x1, py), (h_right_x1, min(py+4, h_right_y2)), (0, 0, 200), 1)
                    cv2.line(vis_image, (h_right_x2, py), (h_right_x2, min(py+4, h_right_y2)), (0, 0, 200), 1)
                cv2.putText(vis_image, "W", (h_right_x2-10, h_right_y1-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)

            # Draw vertical ROIs with dotted rectangles (for HEIGHT detection) - BLUE
            # Top vertical ROI
            if v_top_x2 > v_top_x1 and v_top_y2 > v_top_y1:
                for px in range(v_top_x1, v_top_x2, 8):
                    cv2.line(vis_image, (px, v_top_y1), (min(px+4, v_top_x2), v_top_y1), (200, 0, 0), 1)
                    cv2.line(vis_image, (px, v_top_y2), (min(px+4, v_top_x2), v_top_y2), (200, 0, 0), 1)
                for py in range(v_top_y1, v_top_y2, 8):
                    cv2.line(vis_image, (v_top_x1, py), (v_top_x1, min(py+4, v_top_y2)), (200, 0, 0), 1)
                    cv2.line(vis_image, (v_top_x2, py), (v_top_x2, min(py+4, v_top_y2)), (200, 0, 0), 1)
                cv2.putText(vis_image, "H", (v_top_x1+2, v_top_y1+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 1)

            # Bottom vertical ROI
            if v_bottom_x2 > v_bottom_x1 and v_bottom_y2 > v_bottom_y1:
                for px in range(v_bottom_x1, v_bottom_x2, 8):
                    cv2.line(vis_image, (px, v_bottom_y1), (min(px+4, v_bottom_x2), v_bottom_y1), (200, 0, 0), 1)
                    cv2.line(vis_image, (px, v_bottom_y2), (min(px+4, v_bottom_x2), v_bottom_y2), (200, 0, 0), 1)
                for py in range(v_bottom_y1, v_bottom_y2, 8):
                    cv2.line(vis_image, (v_bottom_x1, py), (v_bottom_x1, min(py+4, v_bottom_y2)), (200, 0, 0), 1)
                    cv2.line(vis_image, (v_bottom_x2, py), (v_bottom_x2, min(py+4, v_bottom_y2)), (200, 0, 0), 1)
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

            # Draw a small marker at measurement position
            cv2.circle(vis_image, (x, y), 8, color, 2)

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

            # Calculate intersection point (width's X, height's Y)
            intersection_x = int(width_x)
            intersection_y = int(height_y)

            # Draw small cross at the intersection point
            cross_size = 10
            cv2.line(vis_image, (intersection_x - cross_size, intersection_y),
                    (intersection_x + cross_size, intersection_y), color, 2)
            cv2.line(vis_image, (intersection_x, intersection_y - cross_size),
                    (intersection_x, intersection_y + cross_size), color, 2)

            # Draw lines from intersection to measurements (thin lines)
            cv2.line(vis_image, (intersection_x, intersection_y),
                    (int(width_x), int(width_y)), color, 1, cv2.LINE_AA)
            cv2.line(vis_image, (intersection_x, intersection_y),
                    (int(height_x), int(height_y)), color, 1, cv2.LINE_AA)

            # Mark measurement positions with small circles
            cv2.circle(vis_image, (int(width_x), int(width_y)), 5, color, 2)
            cv2.circle(vis_image, (int(height_x), int(height_y)), 5, color, 2)

            # Find clear position for marker that avoids overlapping text and other markers
            marker_x, marker_y = find_clear_position_for_marker(
                intersection_x, intersection_y,
                (width_x, width_y), (height_x, height_y),
                measurements_list if measurements_list else [],
                opening, vis_image,
                placed_markers,  # Pass existing markers to avoid
                overlay_info  # Pass overlay info for dimension text
            )

            # Draw white filled circle with colored outline for opening number
            cv2.circle(vis_image, (marker_x, marker_y), 35, (255, 255, 255), -1)  # White fill
            cv2.circle(vis_image, (marker_x, marker_y), 35, color, 3)  # Colored outline

            # Draw opening number (bigger text with #)
            text = f"#{idx+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2

            # Get text size for proper centering
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

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

            # Add dimension label below marker (add overlay if present)
            ol_text = f" + {overlay_info}" if overlay_info else ""
            dim_text = f"{opening['width']}{ol_text} W x {opening['height']}{ol_text} H"
            label_y = marker_y + 50

            # Background for dimension text
            (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.7, 2)
            cv2.rectangle(vis_image,
                         (marker_x - text_width//2 - 3, label_y - text_height - 3),
                         (marker_x + text_width//2 + 3, label_y + 3),
                         (255, 255, 255), -1)

            # Draw dimension text
            cv2.putText(vis_image, dim_text,
                       (marker_x - text_width//2, label_y),
                       font, 0.7, color, 2)

            # Add this marker to the placed markers list for collision avoidance
            placed_markers.append({
                'position': (marker_x, marker_y),
                'opening': opening
            })

        # Add opening list panel/legend
        if room_name or overlay_info:
            legend_height = 60 + (len(paired_openings) * 35)
            legend_width = 350
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
            title_text = f"{room_name} - Finish Sizes"
            if overlay_info:
                title_text = f"{room_name} - Finish Sizes (including {overlay_info})"

            title_y = legend_y + 30
            cv2.putText(vis_image, title_text,
                       (legend_x + 10, title_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Opening list
            for i, opening in enumerate(paired_openings):
                opening_y = title_y + 35 + (i * 35)
                color = opening_colors[i % len(opening_colors)]

                # Draw small circle with number (increased size for better fit)
                circle_center_x = legend_x + 25
                circle_center_y = opening_y - 5
                cv2.circle(vis_image, (circle_center_x, circle_center_y), 15, (255, 255, 255), -1)
                cv2.circle(vis_image, (circle_center_x, circle_center_y), 15, color, 2)

                # Center text in circle
                legend_text = f"#{i+1}"
                legend_font = cv2.FONT_HERSHEY_SIMPLEX
                legend_font_scale = 0.5
                legend_thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(legend_text, legend_font, legend_font_scale, legend_thickness)
                text_x = circle_center_x - text_w // 2
                text_y = circle_center_y + text_h // 2 - baseline // 2

                cv2.putText(vis_image, legend_text,
                           (text_x, text_y),
                           legend_font, legend_font_scale, color, legend_thickness)

                # Opening specification (add overlay if present)
                ol_text = f" + {overlay_info}" if overlay_info else ""
                spec_text = f"{opening['width']}{ol_text} W x {opening['height']}{ol_text} H"
                cv2.putText(vis_image, spec_text,
                           (legend_x + 50, opening_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


    # No need to combine images since panel is overlaid
    combined = vis_image

    # Save visualization
    if save_viz:
        output_path = image_path.replace('.png', '_test_viz.png')
        cv2.imwrite(output_path, combined)
        # Convert to Windows path for display if needed
        display_path = output_path.replace('/', '\\') if output_path.startswith('//') else output_path
        print(f"\n[SAVED] Visualization: {display_path}")

    return combined

def main():
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
        sys.exit(1)

    image_path = sys.argv[1]
    # Always create visualization by default (use --no-viz to disable)
    create_viz = '--no-viz' not in sys.argv
    save_debug = '--debug' in sys.argv or '--save-debug' in sys.argv

    # Visualization flags
    show_groups = '--show-groups' in sys.argv
    show_opencv = '--show-opencv' in sys.argv
    show_line_rois = '--no-rois' not in sys.argv  # ROIs enabled by default
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

        # Handle both old format (string) and new format (tuple)
        if isinstance(result, tuple):
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

            measurements.append({
                'text': measurement,
                'position': center,
                'bounds': updated_bounds
            })
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
    if measurement_categories and measurements:
        print("\n=== PHASE 4: Pairing Measurements into Cabinet Openings ===")
        paired_openings = pair_measurements_by_proximity(measurement_categories, measurements)

        if paired_openings:
            print("\nCABINET OPENING SPECIFICATIONS:")
            print("-" * 60)
            for i, opening in enumerate(paired_openings, 1):
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
                           overlay_info=overlay_info)

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
                    'specification': f"{opening['width']} W × {opening['height']} H",
                    'width_position': opening['width_pos'],
                    'height_position': opening['height_pos'],
                    'pairing_distance': opening['distance']
                }
                for i, opening in enumerate(paired_openings, 1)
            ]
        }

        output_json = os.path.join(output_dir, f"{base_name}_cabinet_openings.json")
        with open(output_json, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n[SAVED] Cabinet openings data: {output_json}")

if __name__ == "__main__":
    main()