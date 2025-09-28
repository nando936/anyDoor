"""
Debug script to see exactly how the grouping and reconstruction logic
handles the two "20 1/16" measurements on page 3.
"""
import cv2
import numpy as np
import requests
import base64
import json
import re
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("="*80)
    print("PAGE 3 GROUPING DEBUG - Why second '20 1/16' isn't detected")
    print("="*80)

    # Get API key
    API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')
    if not API_KEY:
        print("[ERROR] Please set GOOGLE_VISION_API_KEY")
        return

    # Path to page 3 image
    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3.png"

    # Run initial OCR
    print("\n[STEP 1] Running initial OCR...")
    with open(image_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode('utf-8')

    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
    request = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, json=request)
    all_text_items_list = []

    if response.status_code == 200:
        result = response.json()
        if 'responses' in result and result['responses']:
            annotations = result['responses'][0].get('textAnnotations', [])

            if annotations:
                # Skip first (full text) and collect individual items
                for ann in annotations[1:]:
                    text = ann['description']
                    vertices = ann['boundingPoly']['vertices']
                    x = sum(v.get('x', 0) for v in vertices) / 4
                    y = sum(v.get('y', 0) for v in vertices) / 4

                    all_text_items_list.append({
                        'text': text,
                        'x': x,
                        'y': y,
                        'vertices': vertices
                    })

    # Find all "20" and fraction items
    print("\n[STEP 2] Finding all '20' and fraction components...")
    twenty_items = []
    fraction_items = []

    for item in all_text_items_list:
        if item['text'] == '20':
            twenty_items.append(item)
            print(f"  Found '20' at ({item['x']:.0f}, {item['y']:.0f})")

        if '1/16' in item['text'] or item['text'] == '1' or item['text'] == '16':
            fraction_items.append(item)
            print(f"  Found fraction '{item['text']}' at ({item['x']:.0f}, {item['y']:.0f})")

    print(f"\nTotal: {len(twenty_items)} '20' items, {len(fraction_items)} fraction items")

    # Simulate the grouping logic from measurement_based_detector.py
    print("\n[STEP 3] Simulating grouping logic (120px horizontal, 30px vertical)...")
    groups = []
    used_indices = set()

    for i, item in enumerate(all_text_items_list):
        if i in used_indices:
            continue

        # Only process items with numbers
        if not any(char.isdigit() for char in item['text']):
            continue

        # Start a new group
        group = [item]
        used_indices.add(i)

        # Find nearby items
        for j, other in enumerate(all_text_items_list):
            if j in used_indices:
                continue

            # Check proximity
            x_dist = abs(other['x'] - item['x'])
            y_dist = abs(other['y'] - item['y'])

            if x_dist < 120 and y_dist < 30:  # Same criteria as detector
                group.append(other)
                used_indices.add(j)

        # Only show groups containing "20"
        if any('20' in g['text'] for g in group):
            group_text = ' '.join([g['text'] for g in group])
            avg_x = sum(g['x'] for g in group) / len(group)
            avg_y = sum(g['y'] for g in group) / len(group)
            print(f"\n  Group at ({avg_x:.0f}, {avg_y:.0f}):")
            print(f"    Items: {[g['text'] for g in group]}")
            print(f"    Combined: '{group_text}'")

    # Check specific pairing distances
    print("\n[STEP 4] Checking specific pairing possibilities...")
    for twenty in twenty_items:
        print(f"\n  '20' at ({twenty['x']:.0f}, {twenty['y']:.0f}):")

        # Find nearest fraction
        nearest_fraction = None
        min_dist = float('inf')

        for frac in fraction_items:
            dist = ((twenty['x'] - frac['x'])**2 + (twenty['y'] - frac['y'])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_fraction = frac

        if nearest_fraction:
            x_dist = abs(twenty['x'] - nearest_fraction['x'])
            y_dist = abs(twenty['y'] - nearest_fraction['y'])
            print(f"    Nearest fraction: '{nearest_fraction['text']}' at ({nearest_fraction['x']:.0f}, {nearest_fraction['y']:.0f})")
            print(f"    Distance: {min_dist:.1f}px (x_dist={x_dist:.1f}, y_dist={y_dist:.1f})")

            if x_dist < 120 and y_dist < 30:
                print(f"    [WOULD GROUP] Within threshold - should form '20 {nearest_fraction['text']}'")
            else:
                print(f"    [NO GROUP] Outside threshold (need x<120, y<30)")

    # Check the reconstruction logic for whole + fraction
    print("\n[STEP 5] Testing reconstruction logic...")
    reconstructed = []
    used = set()

    for i, item in enumerate(all_text_items_list):
        if i in used or item['text'] != '20':
            continue

        # Look for fraction nearby
        for j, other in enumerate(all_text_items_list):
            if j != i and j not in used:
                if re.match(r'^\d+/\d+$', other['text']):
                    x_dist = abs(other['x'] - item['x'])
                    y_dist = abs(other['y'] - item['y'])

                    if x_dist < 100 and y_dist < 30:  # Reconstruction threshold
                        combined = f"{item['text']} {other['text']}"
                        avg_x = (item['x'] + other['x']) / 2
                        avg_y = (item['y'] + other['y']) / 2
                        reconstructed.append({
                            'text': combined,
                            'x': avg_x,
                            'y': avg_y
                        })
                        used.add(i)
                        used.add(j)
                        print(f"  Reconstructed: '{combined}' at ({avg_x:.0f}, {avg_y:.0f})")
                        break

    print(f"\nTotal reconstructed: {len(reconstructed)} measurements")

    # Visual debug - create image showing the positions
    image = cv2.imread(image_path)
    debug_img = image.copy()

    # Draw all "20" positions in red
    for twenty in twenty_items:
        x, y = int(twenty['x']), int(twenty['y'])
        cv2.circle(debug_img, (x, y), 20, (0, 0, 255), 3)
        cv2.putText(debug_img, "20", (x-15, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw all fraction positions in blue
    for frac in fraction_items:
        x, y = int(frac['x']), int(frac['y'])
        cv2.circle(debug_img, (x, y), 20, (255, 0, 0), 3)
        cv2.putText(debug_img, frac['text'], (x-30, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw grouping ranges (120x30 boxes) around each "20"
    for twenty in twenty_items:
        x, y = int(twenty['x']), int(twenty['y'])
        # Draw the grouping threshold box
        cv2.rectangle(debug_img,
                     (x - 120, y - 30),
                     (x + 120, y + 30),
                     (0, 255, 0), 2)
        cv2.putText(debug_img, "Group Zone", (x - 120, y - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite("debug_page3_grouping_visualization.png", debug_img)
    print(f"\nSaved visualization to: debug_page3_grouping_visualization.png")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("Check the visualization to see why the second '20 1/16' isn't grouping")

if __name__ == "__main__":
    main()