#!/usr/bin/env python3
"""
Analyze down arrow angles in a specific image to determine correct angle ranges.
Excludes vertical lines and focuses on diagonal arrow legs.
"""

import cv2
import numpy as np
import sys
import os

def analyze_arrow_angles(image_path):
    """Analyze line angles in the image to identify down arrow patterns."""

    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return

    print(f"Analyzing image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)

    if lines is None or len(lines) == 0:
        print("No lines detected!")
        return

    print(f"\nTotal lines detected: {len(lines)}")

    # Analyze angles
    angle_stats = {
        'down_right': [],  # 20-88°
        'down_left': [],   # 285-360°
        'vertical_down': [],  # 85-95°
        'vertical_up': [],    # 265-275°
        'horizontal': [],     # 0-10° and 350-360°
        'other': []
    }

    all_angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize to 0-360
        if angle < 0:
            angle += 360

        all_angles.append(angle)

        # Categorize
        if 20 <= angle <= 88:
            angle_stats['down_right'].append((angle, x1, y1, x2, y2))
        elif 285 <= angle <= 360:
            angle_stats['down_left'].append((angle, x1, y1, x2, y2))
        elif 85 <= angle <= 95:
            angle_stats['vertical_down'].append((angle, x1, y1, x2, y2))
        elif 265 <= angle <= 275:
            angle_stats['vertical_up'].append((angle, x1, y1, x2, y2))
        elif (0 <= angle <= 10) or (350 <= angle <= 360):
            angle_stats['horizontal'].append((angle, x1, y1, x2, y2))
        else:
            angle_stats['other'].append((angle, x1, y1, x2, y2))

    # Print statistics
    print("\n=== ANGLE STATISTICS ===")
    print(f"Down-right arrow legs (20-88°): {len(angle_stats['down_right'])} lines")
    print(f"Down-left arrow legs (285-360°): {len(angle_stats['down_left'])} lines")
    print(f"Vertical DOWN (85-95°): {len(angle_stats['vertical_down'])} lines")
    print(f"Vertical UP (265-275°): {len(angle_stats['vertical_up'])} lines")
    print(f"Horizontal (0-10°, 350-360°): {len(angle_stats['horizontal'])} lines")
    print(f"Other angles: {len(angle_stats['other'])} lines")

    # Print all angles sorted
    print(f"\nAll angles detected (sorted): {sorted([f'{a:.1f}' for a in all_angles])}")

    # Print down arrow angles in detail
    if angle_stats['down_right']:
        print("\nDown-right arrow leg angles:")
        for angle, x1, y1, x2, y2 in sorted(angle_stats['down_right']):
            print(f"  {angle:.1f}° - line from ({x1},{y1}) to ({x2},{y2})")

    if angle_stats['down_left']:
        print("\nDown-left arrow leg angles:")
        for angle, x1, y1, x2, y2 in sorted(angle_stats['down_left']):
            print(f"  {angle:.1f}° - line from ({x1},{y1}) to ({x2},{y2})")

    if angle_stats['vertical_down']:
        print("\nVertical DOWN lines (should be excluded):")
        for angle, x1, y1, x2, y2 in sorted(angle_stats['vertical_down']):
            print(f"  {angle:.1f}° - line from ({x1},{y1}) to ({x2},{y2})")

    if angle_stats['vertical_up']:
        print("\nVertical UP lines (should be excluded):")
        for angle, x1, y1, x2, y2 in sorted(angle_stats['vertical_up']):
            print(f"  {angle:.1f}° - line from ({x1},{y1}) to ({x2},{y2})")

    # Create annotated image
    annotated = image.copy()

    # Draw lines with different colors
    # Green: Down arrow lines
    for angle, x1, y1, x2, y2 in angle_stats['down_right']:
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

    for angle, x1, y1, x2, y2 in angle_stats['down_left']:
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

    # Red: Vertical lines (should be excluded)
    for angle, x1, y1, x2, y2 in angle_stats['vertical_down']:
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red

    for angle, x1, y1, x2, y2 in angle_stats['vertical_up']:
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red

    # Gray: Other lines
    for angle, x1, y1, x2, y2 in angle_stats['horizontal']:
        cv2.line(annotated, (x1, y1), (x2, y2), (128, 128, 128), 1)  # Gray

    for angle, x1, y1, x2, y2 in angle_stats['other']:
        cv2.line(annotated, (x1, y1), (x2, y2), (128, 128, 128), 1)  # Gray

    # Save annotated image
    output_path = image_path.replace('.png', '_analyzed.png')
    cv2.imwrite(output_path, annotated)
    print(f"\nAnnotated image saved: {output_path}")
    print("  Green lines: Down arrow legs (20-88° and 285-360°)")
    print("  Red lines: Vertical lines (should be excluded)")
    print("  Gray lines: Other angles")

    # Recommendations
    print("\n=== RECOMMENDATIONS ===")

    has_down_arrow = len(angle_stats['down_right']) > 0 and len(angle_stats['down_left']) > 0
    has_vertical = len(angle_stats['vertical_down']) > 0 or len(angle_stats['vertical_up']) > 0

    if has_down_arrow:
        print("✓ Down arrow detected (both legs present)")
    else:
        print("✗ No down arrow detected (missing one or both legs)")

    if has_vertical:
        print("⚠ Vertical lines detected - current ranges correctly exclude them")
    else:
        print("✓ No vertical lines in current detection ranges")

    print("\nCurrent angle ranges:")
    print("  Right leg: 20-88° (down-right)")
    print("  Left leg: 285-360° (down-left)")
    print("  Excludes: 88-285° (includes vertical 90° and 270°)")


if __name__ == "__main__":
    # Try to find the image
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Possible image locations
    image_name = "page-3_M8_101-4_bottom_vroi_class.png"
    possible_paths = [
        os.path.join(script_dir, image_name),
        os.path.join(script_dir, "..", "..", "..", "..", "..", "onedrive", "customers", "raised-panel", "Measures-2025-10-15(12-09)", "all_pages", image_name),
        os.path.join("/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages", image_name)
    ]

    # Use command line argument if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find the image
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break

        if image_path is None:
            print(f"Error: Could not find {image_name}")
            print("Tried:")
            for path in possible_paths:
                print(f"  {path}")
            sys.exit(1)

    analyze_arrow_angles(image_path)
