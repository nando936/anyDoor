#!/usr/bin/env python3
"""
Text detection and OCR functions for cabinet measurement extraction.
Handles Vision API calls, room/overlay extraction, and text grouping.
"""

import cv2
import numpy as np
import base64
import requests
import re
from measurement_config import (
    ROOM_PATTERNS, EXCLUDE_PATTERNS, OVERLAY_PATTERN, QUANTITY_NOTATION_PATTERN, GROUPING_CONFIG
)
from image_preprocessing import apply_hsv_preprocessing, find_opencv_supplemental_regions


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

            # First, find "OL" text position
            ol_position = None
            for ann in annotations[1:]:
                text = ann['description']
                if 'OL' in text.upper():
                    vertices = ann['boundingPoly']['vertices']
                    ol_x = sum(v.get('x', 0) for v in vertices) / 4
                    ol_y = sum(v.get('y', 0) for v in vertices) / 4
                    ol_position = (ol_x, ol_y)
                    exclude_items.append({'text': text, 'x': ol_x, 'y': ol_y})
                    break

            # Now find fractions that are close to "OL" position
            if ol_position:
                proximity_threshold = 150  # pixels
                for ann in annotations[1:]:
                    text = ann['description']
                    # Check if it's a fraction (contains /)
                    if '/' in text and 'OL' not in text.upper():
                        vertices = ann['boundingPoly']['vertices']
                        x = sum(v.get('x', 0) for v in vertices) / 4
                        y = sum(v.get('y', 0) for v in vertices) / 4
                        # Calculate distance to OL
                        distance = ((x - ol_position[0])**2 + (y - ol_position[1])**2)**0.5
                        if distance < proximity_threshold:
                            # This fraction is close to OL, exclude it
                            exclude_items.append({'text': text, 'x': x, 'y': y})
                            print(f"  Excluding overlay fraction '{text}' near OL (distance: {distance:.1f}px)")

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
                    text_upper = text.upper()
                    # Exclude if exact match OR if text contains room words
                    # This handles cases like "#3" being part of "Bath #3"
                    should_exclude = False
                    if text_upper in [w.upper() for w in room_words]:
                        should_exclude = True
                    else:
                        # Check if this text is part of the room name (e.g., "#3" in "Bath #3")
                        for word in room_words:
                            if word.upper() in text_upper or text_upper in word.upper():
                                should_exclude = True
                                break

                    if should_exclude:
                        vertices = ann['boundingPoly']['vertices']
                        x = sum(v.get('x', 0) for v in vertices) / 4
                        y = sum(v.get('y', 0) for v in vertices) / 4
                        exclude_items.append({'text': text, 'x': x, 'y': y})
                break

        # Check for customer quantity notation (e.g., "1 or 2")
        quantity_matches = re.finditer(QUANTITY_NOTATION_PATTERN, full_text, re.IGNORECASE)
        for match in quantity_matches:
            notation_text = match.group(0)
            print(f"Found quantity notation: {notation_text}")

            # Find all parts of the notation in annotations and exclude them
            parts = notation_text.split()  # e.g., ["1", "or", "2"]
            for part in parts:
                for ann in annotations[1:]:
                    if ann['description'].lower() == part.lower():
                        vertices = ann['boundingPoly']['vertices']
                        x = sum(v.get('x', 0) for v in vertices) / 4
                        y = sum(v.get('y', 0) for v in vertices) / 4
                        exclude_items.append({'text': ann['description'], 'x': x, 'y': y})

        # Also exclude common non-measurement text
        for ann in annotations[1:]:
            text = ann['description'].upper()
            # Check if the entire text matches OR if any word in the text matches
            words = text.split()
            should_exclude = text in EXCLUDE_PATTERNS or any(word in EXCLUDE_PATTERNS for word in words)
            if should_exclude:
                vertices = ann['boundingPoly']['vertices']
                x = sum(v.get('x', 0) for v in vertices) / 4
                y = sum(v.get('y', 0) for v in vertices) / 4
                exclude_items.append({'text': ann['description'], 'x': x, 'y': y})

    return room_name, overlay_info, exclude_items


def extract_measurements_from_text(text):
    """Extract measurement patterns from text"""
    # Pattern for measurements: "24 3/4", "11 1/4", "5/8", "20", "6", etc.
    # Matches: whole + fraction | fraction | decimal | any whole number
    pattern = r'\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\b\d+\b'
    matches = re.findall(pattern, text)
    return matches


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


def is_valid_measurement(texts):
    """
    Check if a set of texts forms a valid measurement.
    Valid patterns:
    - Single number: "18", "22"
    - Number with fraction: "6 3/8", "22 1/16"
    - Just a fraction: "1/2", "3/8"
    Invalid: Multiple complete measurements like "6 3/8" and "18"
    """
    import re

    # Join texts and check pattern
    combined = ' '.join(texts)

    # Count whole numbers and fractions
    whole_numbers = re.findall(r'\b\d+\b(?!\s*/)', combined)  # Numbers not followed by /
    fractions = re.findall(r'\b\d+/\d+', combined)  # Fractions like 3/8

    # Valid if we have:
    # - 0 or 1 whole number AND 0 or 1 fraction
    has_valid_count = len(whole_numbers) <= 1 and len(fractions) <= 1

    # Invalid if we have multiple whole numbers (like "6" and "18")
    if len(whole_numbers) > 1:
        return False

    return has_valid_count


def merge_close_centers(interest_areas, threshold=None):
    """
    Merge centers that are too close together.
    IMPORTANT: Only merge if the result forms a VALID measurement.
    Don't merge two complete measurements (like "6 3/8" + "18").
    """
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
            # Y should be tight (max 45px) since text on same line should have similar Y
            # X can be more relaxed (use full threshold)
            if x_dist < threshold and y_dist < 45:
                # Check if merging would create a valid measurement
                proposed_texts = merged_area['all_texts'] + other['texts']
                if is_valid_measurement(proposed_texts):
                    merged_area['centers'].append(other['center'])
                    merged_area['all_bounds'].append(other['bounds'])
                    merged_area['all_texts'].extend(other['texts'])
                    used.add(j)
                    print(f"  Merging areas at {area['center']} and {other['center']} (x_dist={x_dist:.1f}, y_dist={y_dist:.1f})")
                else:
                    print(f"  SKIP merging {area['center']} and {other['center']} - would create invalid measurement: {proposed_texts}")

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
