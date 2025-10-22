#!/usr/bin/env python3
"""
Test script to analyze ROI images and detect arrows using different methods.
"""

import cv2
import numpy as np
import sys
import os

def detect_lines_simple(image):
    """Detect lines using HoughLinesP"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=10)
    return lines if lines is not None else []

def check_for_arrow_pattern(image, direction='left'):
    """
    Check for arrow pattern by looking for converging lines.
    direction: 'left' or 'right'
    """
    lines = detect_lines_simple(image)

    print(f"\n  Found {len(lines)} lines total")

    if len(lines) < 2:
        print(f"  Need at least 2 lines to form arrow")
        return False

    # Print all line angles
    print(f"\n  All line angles:")
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        print(f"    Line {i+1}: ({x1},{y1})->({x2},{y2}), angle={angle:.1f}°, length={length:.1f}px")

    # Check for converging line pairs
    arrow_found = False
    for i, line1 in enumerate(lines):
        x1a, y1a, x2a, y2a = line1[0]
        angle1 = np.degrees(np.arctan2(y2a - y1a, x2a - x1a))

        for j, line2 in enumerate(lines[i+1:], start=i+1):
            x1b, y1b, x2b, y2b = line2[0]
            angle2 = np.degrees(np.arctan2(y2b - y1b, x2b - x1b))

            # Check if angles are opposite (forming V pattern)
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # Lines should be roughly opposite (90-170 degrees apart)
            if 90 < angle_diff < 170:
                print(f"\n  POTENTIAL ARROW PATTERN:")
                print(f"    Line {i+1}: angle={angle1:.1f}°")
                print(f"    Line {j+1}: angle={angle2:.1f}°")
                print(f"    Angle difference: {angle_diff:.1f}°")
                arrow_found = True

    return arrow_found

def analyze_green_arrows(image):
    """Filter for green color and detect arrows"""
    # HSV filter for green dimension lines
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations
    kernel = np.ones((2,2), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Count green pixels
    green_pixels = np.sum(green_mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100

    print(f"\n  Green pixel analysis:")
    print(f"    Green pixels: {green_pixels}/{total_pixels} ({green_percentage:.2f}%)")

    # Detect edges on green-filtered image
    edges = cv2.Canny(green_mask, 30, 100)

    # Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)

    if lines is not None:
        print(f"    Found {len(lines)} lines in green-filtered image")

        # Analyze line angles
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Normalize to 0-360
            if angle < 0:
                angle += 360
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            print(f"      Line {i+1}: angle={angle:.1f}°, length={length:.1f}px")

        return lines
    else:
        print(f"    No lines found in green-filtered image")
        return []

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_arrow_detection.py <roi_image_path>")
        print("\nExample:")
        print("  python test_arrow_detection.py /tmp/Munknown_18_extent_left_roi.png")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"ARROW DETECTION TEST")
    print(f"{'='*60}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Size: {os.path.getsize(image_path)} bytes")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image")
        sys.exit(1)

    print(f"Dimensions: {image.shape[1]}x{image.shape[0]} pixels")

    # Determine direction from filename
    direction = 'left' if 'left' in image_path.lower() else 'right'
    print(f"Expected arrow direction: {direction}")

    # Method 1: Standard line detection
    print(f"\n{'='*60}")
    print("METHOD 1: Standard HoughLinesP detection")
    print(f"{'='*60}")
    has_arrow_1 = check_for_arrow_pattern(image, direction)
    print(f"\n  Result: {'ARROW FOUND' if has_arrow_1 else 'NO ARROW'}")

    # Method 2: Green-filtered detection
    print(f"\n{'='*60}")
    print("METHOD 2: Green-filtered detection")
    print(f"{'='*60}")
    green_lines = analyze_green_arrows(image)
    has_arrow_2 = len(green_lines) >= 2
    print(f"\n  Result: {'ARROW FOUND' if has_arrow_2 else 'NO ARROW'}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Method 1 (Standard):      {'✓ FOUND' if has_arrow_1 else '✗ NOT FOUND'}")
    print(f"Method 2 (Green-filter):  {'✓ FOUND' if has_arrow_2 else '✗ NOT FOUND'}")

    if has_arrow_1 or has_arrow_2:
        print(f"\n✓ At least one method detected arrows in this ROI")
    else:
        print(f"\n✗ No arrows detected by any method")
        print(f"  This suggests the dimension line + arrow is one continuous segment")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
