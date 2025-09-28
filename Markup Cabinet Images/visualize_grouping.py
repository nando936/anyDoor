#!/usr/bin/env python3
"""
Visualize text grouping bounds for debugging measurement detection
"""

import cv2
import numpy as np
import sys
import os
import base64
import requests
import json

# Configuration
GROUPING_CONFIG = {
    'x_distance': 100,   # Max horizontal distance for grouping (increased to 100)
    'y_distance': 40,   # Max vertical distance for grouping
    'merge_threshold': 70  # Distance threshold for merging centers
}

# Colors for visualization (BGR format)
COLORS = [
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (128, 0, 255),    # Orange
    (255, 0, 128),    # Pink
    (0, 128, 255),    # Light orange
    (128, 255, 0),    # Light green
]

def apply_hsv_preprocessing(image):
    """Apply HSV preprocessing to enhance green text"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Convert mask to 3-channel
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Invert to get black text on white background
    preprocessed = cv2.bitwise_not(mask_3channel)

    return preprocessed

def find_room_and_overlay(annotations):
    """Extract room name and overlay info"""
    import re
    room_name = ""
    overlay_info = ""
    exclude_items = []

    if not annotations:
        return room_name, overlay_info, exclude_items

    # Get full text to search for overlay pattern
    full_text = annotations[0].get('description', '') if annotations else ''

    # Search for overlay pattern in full text (e.g., "5/8 OL")
    overlay_pattern = r'(\d+/\d+\s+OL)'
    overlay_match = re.search(overlay_pattern, full_text, re.IGNORECASE)

    if overlay_match:
        overlay_info = overlay_match.group(1).strip()

        # Now find and exclude individual annotations that are part of the overlay
        for ann in annotations[1:]:  # Skip full text
            text = ann['description']
            vertices = ann['boundingPoly']['vertices']
            x = sum(v.get('x', 0) for v in vertices) / 4
            y = sum(v.get('y', 0) for v in vertices) / 4

            # Exclude if it contains OL or is part of the overlay text
            if 'OL' in text.upper() or text in overlay_info:
                exclude_items.append({'text': text, 'x': x, 'y': y})

    return room_name, overlay_info, exclude_items

def visualize_grouping(image_path):
    """Create visualization of text grouping"""

    # Get API key
    api_key = os.environ.get('GOOGLE_VISION_API_KEY')
    if not api_key:
        # Try loading from .env file
        from dotenv import load_dotenv
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            api_key = os.environ.get('GOOGLE_VISION_API_KEY')

    if not api_key:
        print("[ERROR] GOOGLE_VISION_API_KEY not set")
        return

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    # Create copy for visualization
    vis_image = image.copy()

    # Apply preprocessing
    hsv_image = apply_hsv_preprocessing(image)

    # Encode for Vision API
    _, buffer = cv2.imencode('.png', hsv_image)
    content = base64.b64encode(buffer).decode('utf-8')

    # Run OCR
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
        return

    result = response.json()
    annotations = result['responses'][0].get('textAnnotations', [])

    # Extract room and overlay info
    room_name, overlay_info, exclude_items = find_room_and_overlay(annotations)

    print(f"Overlay: {overlay_info}")
    print(f"Excluding {len(exclude_items)} items")

    # Get all text items with numbers (excluding overlay items)
    text_items = []
    for ann in annotations[1:]:  # Skip full text
        text = ann['description']
        original_text = text
        cleaned_text = text.strip('-')

        if cleaned_text and any(char.isdigit() for char in cleaned_text):
            vertices = ann['boundingPoly']['vertices']
            x = sum(v.get('x', 0) for v in vertices) / 4
            y = sum(v.get('y', 0) for v in vertices) / 4

            # Check if should exclude
            should_exclude = False
            for exc in exclude_items:
                if (exc['text'] == original_text and
                    abs(exc['x'] - x) < 10 and
                    abs(exc['y'] - y) < 10):
                    should_exclude = True
                    break

            if not should_exclude:
                text_items.append({
                    'text': cleaned_text,
                    'x': x,
                    'y': y,
                    'vertices': vertices
                })

    print(f"\nFound {len(text_items)} text items:")
    for item in text_items:
        print(f"  '{item['text']}' at ({item['x']:.0f}, {item['y']:.0f})")

    # Draw individual text items as small circles
    for item in text_items:
        cv2.circle(vis_image, (int(item['x']), int(item['y'])), 5, (0, 0, 0), -1)
        # Label each text item
        cv2.putText(vis_image, item['text'],
                   (int(item['x'] + 10), int(item['y'] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Group nearby items
    groups = []
    used = set()

    for i, item in enumerate(text_items):
        if i in used:
            continue

        group = [item]
        used.add(i)

        # Find nearby items
        for j, other in enumerate(text_items):
            if j in used:
                continue

            x_dist = abs(other['x'] - item['x'])
            y_dist = abs(other['y'] - item['y'])

            print(f"  Checking distance from '{item['text']}' to '{other['text']}': x={x_dist:.0f}, y={y_dist:.0f}")

            if x_dist < GROUPING_CONFIG['x_distance'] and y_dist < GROUPING_CONFIG['y_distance']:
                print(f"    -> GROUPED (within {GROUPING_CONFIG['x_distance']}x{GROUPING_CONFIG['y_distance']})")
                group.append(other)
                used.add(j)
            else:
                reasons = []
                if x_dist >= GROUPING_CONFIG['x_distance']:
                    reasons.append(f"x too far: {x_dist:.0f} >= {GROUPING_CONFIG['x_distance']}")
                if y_dist >= GROUPING_CONFIG['y_distance']:
                    reasons.append(f"y too far: {y_dist:.0f} >= {GROUPING_CONFIG['y_distance']}")
                print(f"    -> NOT GROUPED ({', '.join(reasons)})")

        groups.append(group)

    print(f"\nFormed {len(groups)} groups:")

    # Draw groups with different colors
    for i, group in enumerate(groups):
        color = COLORS[i % len(COLORS)]

        # Calculate group bounds
        all_x = []
        all_y = []
        texts = []

        for item in group:
            vertices = item['vertices']
            for v in vertices:
                all_x.append(v.get('x', item['x']))
                all_y.append(v.get('y', item['y']))
            texts.append(item['text'])

        left = int(min(all_x))
        right = int(max(all_x))
        top = int(min(all_y))
        bottom = int(max(all_y))

        # Draw rectangle for group
        cv2.rectangle(vis_image, (left - 5, top - 5), (right + 5, bottom + 5), color, 3)

        # Draw group number
        group_label = f"G{i+1}"
        cv2.putText(vis_image, group_label, (left - 30, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Print group info
        print(f"  Group {i+1}: {' + '.join(texts)}")
        print(f"    Bounds: left={left}, right={right}, top={top}, bottom={bottom}")
        print(f"    Width={right-left}, Height={bottom-top}")

    # Add legend
    legend_y = 50
    cv2.putText(vis_image, f"Grouping Config: x_dist < {GROUPING_CONFIG['x_distance']}px, y_dist < {GROUPING_CONFIG['y_distance']}px",
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    legend_y += 30
    cv2.putText(vis_image, f"Total Groups: {len(groups)}",
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save visualization
    output_path = image_path.replace('.png', '_grouping_viz.png')
    cv2.imwrite(output_path, vis_image)
    print(f"\nSaved visualization to: {output_path}")

    # Also test what happens if we increase x_distance
    print(f"\n=== Testing with increased x_distance ===")
    test_thresholds = [90, 100, 110, 120]

    for test_x in test_thresholds:
        groups_test = []
        used_test = set()

        for i, item in enumerate(text_items):
            if i in used_test:
                continue

            group = [item]
            used_test.add(i)

            for j, other in enumerate(text_items):
                if j in used_test:
                    continue

                x_dist = abs(other['x'] - item['x'])
                y_dist = abs(other['y'] - item['y'])

                if x_dist < test_x and y_dist < GROUPING_CONFIG['y_distance']:
                    group.append(other)
                    used_test.add(j)

            groups_test.append(group)

        print(f"  x_distance={test_x}px -> {len(groups_test)} groups")
        if len(groups_test) == 4:
            print(f"    SUCCESS: Got expected 4 groups!")
            for g_idx, g in enumerate(groups_test):
                print(f"      Group {g_idx+1}: {' + '.join([item['text'] for item in g])}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_grouping.py <image_path>")
        sys.exit(1)

    visualize_grouping(sys.argv[1])