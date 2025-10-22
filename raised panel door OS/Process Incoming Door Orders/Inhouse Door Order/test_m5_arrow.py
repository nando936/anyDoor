#!/usr/bin/env python3
"""
Test arrow detection on page 7 M5 down arrow scan image.
Uses edge-based detection instead of HSV filtering.
"""

import cv2
import numpy as np

# Image path
image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page7_M5_227-8_down_arrow_scan.png"

print(f"Loading image: {image_path}")
image = cv2.imread(image_path)

if image is None:
    print(f"ERROR: Could not load image from {image_path}")
    exit(1)

print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

# Convert to grayscale
print("\nConverting to grayscale...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
print("Detecting edges with Canny (threshold 50-150)...")
edges = cv2.Canny(gray, 50, 150)

# Find lines using HoughLinesP
print("Finding lines with HoughLinesP...")
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)

if lines is None:
    print("\nNO LINES FOUND")
    exit(0)

print(f"\nFound {len(lines)} lines total")

# Collect all angles
all_angles = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    # Normalize angle to 0-360 range
    if angle < 0:
        angle += 360
    all_angles.append(angle)

print(f"All line angles (sorted): {[f'{a:.1f}°' for a in sorted(all_angles)]}")

# Filter for down arrow legs
# Right leg: 20-88° (going down-right)
# Left leg: 285-360° (going down-left)
down_right_lines = []
down_left_lines = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    # Normalize angle to 0-360 range
    if angle < 0:
        angle += 360

    # Check if line is right leg (20-88°)
    if 20 <= angle <= 88:
        down_right_lines.append((x1, y1, x2, y2, angle))
        print(f"  Down-right: ({x1},{y1})-({x2},{y2}) angle={angle:.1f}°")
    # Check if line is left leg (285-360°)
    elif 285 <= angle <= 360:
        down_left_lines.append((x1, y1, x2, y2, angle))
        print(f"  Down-left:  ({x1},{y1})-({x2},{y2}) angle={angle:.1f}°")

print(f"\n=== RESULTS ===")
print(f"Down-right lines (20-88°): {len(down_right_lines)}")
print(f"Down-left lines (285-360°): {len(down_left_lines)}")

# Check if arrow detected
has_arrow = (len(down_right_lines) > 0 and len(down_left_lines) > 0)
print(f"Arrow detected: {has_arrow}")

if has_arrow:
    print("\n  ARROW FOUND: Both right and left legs detected!")
else:
    print("\n  NO ARROW: Missing one or both legs")
    if len(down_right_lines) == 0:
        print("    - Missing down-right leg (20-88°)")
    if len(down_left_lines) == 0:
        print("    - Missing down-left leg (285-360°)")
