#!/usr/bin/env python3
"""
Test script to detect and visualize the left arrow in M11 extent ROI
"""

import cv2
import numpy as np
from measurement_config import HSV_CONFIG

# Path to the M11 left extent ROI image
roi_image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-09-09(14-18)/all_pages/page_8_M11_181-16_extent_left_roi.png"

# Read the ROI image
roi_image = cv2.imread(roi_image_path)
if roi_image is None:
    print(f"ERROR: Could not read image at {roi_image_path}")
    exit(1)

print(f"ROI image shape: {roi_image.shape}")

# Apply HSV filter to detect only green dimension arrows
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

if lines is None:
    print("ERROR: No lines detected in ROI")
    exit(1)

print(f"\nFound {len(lines)} lines in ROI")

# Create a debug image to draw on
debug_image = roi_image.copy()

# Draw all detected lines
for i, line in enumerate(lines):
    x1, y1, x2, y2 = line[0]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    # Draw line in blue
    cv2.line(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Put line number and angle
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.putText(debug_image, f"L{i}: {angle:.1f}°", (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    print(f"  Line {i}: ({x1},{y1}) to ({x2},{y2}), angle={angle:.1f}°")

# Now check for converging line pairs (arrows)
print("\n=== Checking for LEFT arrow patterns ===")

arrow_pairs = []

for i in range(len(lines)):
    x1a, y1a, x2a, y2a = lines[i][0]
    angle1 = np.degrees(np.arctan2(y2a - y1a, x2a - x1a))

    for j in range(i + 1, len(lines)):
        x1b, y1b, x2b, y2b = lines[j][0]
        angle2 = np.degrees(np.arctan2(y2b - y1b, x2b - x1b))

        # Check if angles are opposite (forming V pattern)
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Show ALL pairs, not just those in 90-170° range
        print(f"\n  Pair {i}+{j}: angles {angle1:.1f}° and {angle2:.1f}°, diff={angle_diff:.1f}°")

        # Lines should be converging (15-170 degrees apart)
        if not (15 < angle_diff < 170):
            print(f"    ⚠️  SKIPPED: angle diff {angle_diff:.1f}° not in 15-170° range")
            continue

        # Calculate deviations
        abs_angle1 = abs(angle1)
        abs_angle2 = abs(angle2)

        # Deviation from horizontal (0° or 180°)
        dev_horiz_1 = min(abs_angle1, abs(180 - abs_angle1))
        dev_horiz_2 = min(abs_angle2, abs(180 - abs_angle2))

        # Deviation from vertical (90° or -90°)
        dev_vert_1 = abs(abs_angle1 - 90)
        dev_vert_2 = abs(abs_angle2 - 90)

        print(f"    Line {i}: horiz_dev={dev_horiz_1:.1f}°, vert_dev={dev_vert_1:.1f}°")
        print(f"    Line {j}: horiz_dev={dev_horiz_2:.1f}°, vert_dev={dev_vert_2:.1f}°")

        # Check if EITHER line is very close to vertical (±90°)
        line1_near_vertical = dev_vert_1 < 20
        line2_near_vertical = dev_vert_2 < 20

        if line1_near_vertical or line2_near_vertical:
            print(f"    ❌ REJECTED: near-vertical (line {i}: {dev_vert_1:.1f}°, line {j}: {dev_vert_2:.1f}°)")
        else:
            print(f"    ✓ ACCEPTED: horizontal arrow")
            arrow_pairs.append((i, j))

            # Highlight this pair in green
            cv2.line(debug_image, (x1a, y1a), (x2a, y2a), (0, 255, 0), 3)
            cv2.line(debug_image, (x1b, y1b), (x2b, y2b), (0, 255, 0), 3)

            # Mark arrow tip
            all_x = [x1a, x2a, x1b, x2b]
            arrow_x = min(all_x)
            arrow_y = (y1a + y2a + y1b + y2b) // 4
            cv2.circle(debug_image, (arrow_x, arrow_y), 5, (0, 0, 255), -1)
            cv2.putText(debug_image, "ARROW", (arrow_x + 10, arrow_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

print(f"\n=== Found {len(arrow_pairs)} valid arrow pairs ===")

# Save debug image to same folder as input
import os
output_dir = os.path.dirname(roi_image_path)
output_path = os.path.join(output_dir, "test_m11_arrow_debug.png")
cv2.imwrite(output_path, debug_image)
print(f"\n✓ Saved debug image: {output_path}")

# Also save the green mask and edges for inspection
mask_path = os.path.join(output_dir, "test_m11_arrow_mask.png")
edges_path = os.path.join(output_dir, "test_m11_arrow_edges.png")
cv2.imwrite(mask_path, green_mask)
cv2.imwrite(edges_path, edges)
print(f"✓ Saved green mask: {mask_path}")
print(f"✓ Saved edges: {edges_path}")
