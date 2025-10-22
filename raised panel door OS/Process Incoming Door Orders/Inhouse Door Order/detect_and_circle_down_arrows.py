#!/usr/bin/env python3
"""
Detect down arrows in an image and circle them precisely at their apex.
Analyzes angles to determine correct detection ranges.
"""

import cv2
import numpy as np
import sys
import os

def analyze_and_detect_arrows(image_path, output_path=None):
    """Analyze down arrow angles and detect them with precise circle placement."""

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

    # Create copy for visualization
    vis_image = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try green channel isolation first (arrows are often green)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Use green mask for edge detection
    edges = cv2.Canny(green_mask, 20, 60)

    # Save debug images
    debug_dir = os.path.dirname(image_path)
    cv2.imwrite(os.path.join(debug_dir, 'debug_green_mask.png'), green_mask)
    cv2.imwrite(os.path.join(debug_dir, 'debug_edges.png'), edges)

    # Detect lines with more lenient parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=5, minLineLength=5, maxLineGap=5)

    if lines is None or len(lines) == 0:
        print("No lines detected!")
        print("Trying grayscale edge detection...")

        # Fallback to grayscale
        edges_gray = cv2.Canny(gray, 20, 60)
        lines = cv2.HoughLinesP(edges_gray, 1, np.pi/180, threshold=5, minLineLength=5, maxLineGap=5)

        if lines is None or len(lines) == 0:
            print("Still no lines detected!")
            return

    print(f"\nTotal lines detected: {len(lines)}")

    # Analyze angles
    all_angles = []
    all_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize to 0-360
        if angle < 0:
            angle += 360

        all_angles.append(angle)
        all_lines.append((x1, y1, x2, y2, angle))

    # Print all angles
    print(f"\nAll angles detected (sorted): {sorted([f'{a:.1f}' for a in all_angles])}")

    # Categorize lines for down arrow detection
    down_right_lines = []  # 10-88° (widened from 20-88)
    down_left_lines = []   # 285-360°

    for x1, y1, x2, y2, angle in all_lines:
        if 10 <= angle <= 88:
            down_right_lines.append((x1, y1, x2, y2, angle))
        elif 285 <= angle <= 360:
            down_left_lines.append((x1, y1, x2, y2, angle))

    print(f"\nDown-right lines (10-88°): {len(down_right_lines)}")
    for x1, y1, x2, y2, angle in down_right_lines:
        print(f"  {angle:.1f}° - line from ({x1},{y1}) to ({x2},{y2})")

    print(f"\nDown-left lines (285-360°): {len(down_left_lines)}")
    for x1, y1, x2, y2, angle in down_left_lines:
        print(f"  {angle:.1f}° - line from ({x1},{y1}) to ({x2},{y2})")

    # Check if we have a down arrow
    has_down_arrow = len(down_right_lines) > 0 and len(down_left_lines) > 0

    if has_down_arrow:
        print("\n✓ DOWN ARROW DETECTED!")

        # Calculate visual center of the arrow (middle of entire arrow)
        all_y_coords = []
        all_x_coords = []

        for x1, y1, x2, y2, angle in down_right_lines:
            all_y_coords.extend([y1, y2])
            all_x_coords.extend([x1, x2])

        for x1, y1, x2, y2, angle in down_left_lines:
            all_y_coords.extend([y1, y2])
            all_x_coords.extend([x1, x2])

        # Visual center: midpoint between top and bottom of arrow
        min_y = min(all_y_coords)
        max_y = max(all_y_coords)
        center_y = int((min_y + max_y) / 2)
        center_x = int(sum(all_x_coords) / len(all_x_coords))

        print(f"\nArrow visual center calculated at: ({center_x}, {center_y})")
        print(f"  Arrow extends from Y={min_y} to Y={max_y}")

        # Draw circle centered on visual center
        circle_radius = 30
        cv2.circle(vis_image, (center_x, center_y), circle_radius, (0, 255, 255), 2)  # Yellow circle
        cv2.circle(vis_image, (center_x, center_y), 2, (0, 0, 255), -1)  # Red dot at center

        # Draw the detected arrow lines
        for x1, y1, x2, y2, angle in down_right_lines:
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

        for x1, y1, x2, y2, angle in down_left_lines:
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

        # Add label
        cv2.putText(vis_image, "DOWN ARROW", (center_x - 40, center_y - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    else:
        print("\n✗ No down arrow detected (missing one or both legs)")

    # Save output
    if output_path is None:
        output_path = image_path.replace('.png', '_detected.png')

    cv2.imwrite(output_path, vis_image)
    print(f"\nOutput saved: {output_path}")
    print("  Yellow circle: Arrow visual center (middle of entire arrow)")
    print("  Green lines: Detected arrow legs")
    print("  Red dot: Exact center point")

    # Print recommendations
    print("\n=== ANGLE RANGE ANALYSIS ===")

    # Find actual angle ranges in the data
    if down_right_lines:
        dr_angles = [angle for _, _, _, _, angle in down_right_lines]
        print(f"Right leg angles found: {min(dr_angles):.1f}° to {max(dr_angles):.1f}°")

    if down_left_lines:
        dl_angles = [angle for _, _, _, _, angle in down_left_lines]
        print(f"Left leg angles found: {min(dl_angles):.1f}° to {max(dl_angles):.1f}°")

    print("\nRecommended angle ranges:")
    if down_right_lines and min([a for _, _, _, _, a in down_right_lines]) < 20:
        print("  Right leg: 10-88° (arrows found below 20°)")
    else:
        print("  Right leg: 20-88° (current range is good)")

    print("  Left leg: 285-360° (current range)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 detect_and_circle_down_arrows.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    analyze_and_detect_arrows(image_path, output_path)
