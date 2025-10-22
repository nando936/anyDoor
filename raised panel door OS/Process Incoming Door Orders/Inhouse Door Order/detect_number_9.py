#!/usr/bin/env python3
"""
Detect number 9 in images using hole detection.

This matches the exact logic from image_preprocessing.py Phase 1 Pass 3 (lines 307-408).

Logic:
1. Load COLOR image (original, not HSV processed)
2. Convert to grayscale
3. OTSU threshold (THRESH_BINARY + THRESH_OTSU) - NOT inverted
4. Find contours with RETR_TREE to get hierarchy
5. Filter by size: width 10-50px, height 10-50px, area >= 100px²
6. Count children (holes) using hierarchy
7. For each hole, check if hole area < 50px²
8. If hole_count >= 1, it's a 9
9. Circle all 9s in green, number them, save image
"""

import cv2
import numpy as np
from pathlib import Path
import sys


def detect_digit_9(image_path, output_path):
    """
    Detect all instances of digit '9' in an image using hole detection.
    Matches production Phase 1 Pass 3 logic exactly.

    Args:
        image_path: Path to input image
        output_path: Path to save annotated image

    Returns:
        Number of 9s detected
    """
    print(f"Loading image: {image_path}")

    # Step 1: Load image as COLOR (original, not HSV processed)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image")
        return 0

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale")

    # Step 3: OTSU threshold (THRESH_BINARY + THRESH_OTSU) - NOT inverted
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Applied OTSU threshold (mean={np.mean(binary):.1f})")

    # Production code inverts if needed (white digit on black background)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
        print("Inverted binary image (white digits on black background)")

    # Save the threshold image for inspection
    threshold_output = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_threshold.png"
    cv2.imwrite(threshold_output, binary)
    print(f"Saved threshold image: {threshold_output}")

    # Step 4: Find contours with RETR_TREE and CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} total contours")

    nines = []
    all_detections = []  # Track all detections for analysis
    filtered_detections = []  # Track detections that pass all filters

    # Step 5-8: Check each contour
    if hierarchy is not None:
        detection_num = 0
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Skip tiny contours (area >= 100px²)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Size filters: width 10-50px, height 10-50px
            if w < 10 or h < 10 or w > 50 or h > 50:
                continue

            # Count children (holes) using hierarchy[0][idx][2]
            has_hole = False
            hole_count = 0
            total_hole_area = 0

            child_idx = hierarchy[0][idx][2]

            while child_idx != -1:
                hole_contour = contours[child_idx]
                hole_area = cv2.contourArea(hole_contour)

                # Check if hole area < 50px²
                if hole_area < 50:
                    hole_count += 1
                    total_hole_area += hole_area

                # Move to next sibling hole
                child_idx = hierarchy[0][child_idx][0]

            # If hole_count >= 1, track this as a potential detection
            if hole_count >= 1:
                detection_num += 1
                avg_hole_area = total_hole_area / hole_count if hole_count > 0 else 0

                detection_data = {
                    'id': detection_num,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'hole_count': hole_count,
                    'total_hole_area': total_hole_area,
                    'avg_hole_area': avg_hole_area,
                    'filters_passed': []
                }
                all_detections.append(detection_data)

                # FILTER 1: REFINED SIZE FILTER
                # Real digits are width 20-25, height 30-35
                # This excludes the large false positives (35-36px wide) and tiny ones
                if w < 20 or w > 25 or h < 30 or h > 35:
                    detection_data['filters_passed'].append('FAIL: Size (W=20-25, H=30-35)')
                    continue
                detection_data['filters_passed'].append('PASS: Size')

                # FILTER 2: HOLE AREA FILTER
                # Real digits have avg hole area between 10-45
                # This excludes noise with tiny holes (0.5, 2.5, etc.)
                if avg_hole_area < 10 or avg_hole_area > 45:
                    detection_data['filters_passed'].append('FAIL: Avg hole area (10-45)')
                    continue
                detection_data['filters_passed'].append('PASS: Avg hole area')

                # FILTER 3: HOLE COUNT FILTER
                # Real 9s have 1-3 holes, not 6
                if hole_count > 3:
                    detection_data['filters_passed'].append('FAIL: Hole count (1-3)')
                    continue
                detection_data['filters_passed'].append('PASS: Hole count')

                # FILTER 4: AREA FILTER
                # Real digits have area >= 490
                # This excludes smaller noise (#6=411, #13=380)
                if area < 490:
                    detection_data['filters_passed'].append('FAIL: Area (>= 490)')
                    continue
                detection_data['filters_passed'].append('PASS: Area')

                # All filters passed!
                detection_data['filters_passed'].append('*** ALL FILTERS PASSED ***')
                filtered_detections.append(detection_data)
                nines.append((x, y, w, h))

    # Print detailed information for ALL detections
    print(f"\n{'='*80}")
    print(f"DETAILED DETECTION ANALYSIS - ALL {len(all_detections)} DETECTIONS")
    print(f"{'='*80}")
    print(f"{'ID':>3} | {'X':>5} {'Y':>5} | {'W':>3} {'H':>3} | {'Area':>6} | {'Holes':>5} | {'Avg Hole':>8} | Filter Result")
    print(f"{'-'*80}")

    for det in all_detections:
        filter_status = det['filters_passed'][-1] if det['filters_passed'] else 'NO HOLE'
        print(f"#{det['id']:2d} | {det['x']:5d} {det['y']:5d} | "
              f"{det['width']:3d} {det['height']:3d} | "
              f"{det['area']:6.0f} | "
              f"{det['hole_count']:5d} | "
              f"{det['avg_hole_area']:8.1f} | "
              f"{filter_status}")

    print(f"{'-'*80}")
    print(f"\nREAL DIGITS (user confirmed):")
    print(f"  #10 = real 9")
    print(f"  #4  = real 9")
    print(f"  #5  = real 6")
    print(f"\nFALSE POSITIVES: All others")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print(f"FILTER RESULTS: {len(filtered_detections)} detections passed all filters")
    print(f"{'='*80}")
    if len(filtered_detections) > 0:
        print(f"{'ID':>3} | {'X':>5} {'Y':>5} | {'W':>3} {'H':>3} | {'Area':>6} | {'Holes':>5} | {'Avg Hole':>8}")
        print(f"{'-'*80}")
        for det in filtered_detections:
            print(f"#{det['id']:2d} | {det['x']:5d} {det['y']:5d} | "
                  f"{det['width']:3d} {det['height']:3d} | "
                  f"{det['area']:6.0f} | "
                  f"{det['hole_count']:5d} | "
                  f"{det['avg_hole_area']:8.1f}")
    print(f"{'='*80}\n")

    print(f"\nDetected {len(nines)} instances of '9'")

    # Step 6-7: Circle all 9s in green, number them
    if len(nines) > 0:
        vis_image = image.copy()

        for i, (x, y, w, h) in enumerate(nines):
            center_x = x + w // 2
            center_y = y + h // 2
            radius = int(np.sqrt(w**2 + h**2) / 2) + 5

            # Circle and number
            cv2.circle(vis_image, (center_x, center_y), radius, (0, 255, 0), 2)
            label = str(i + 1)
            label_x = center_x + radius + 10
            label_y = center_y
            cv2.putText(vis_image, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            print(f"  9 #{i+1}: center=({center_x},{center_y}), bbox=({x},{y},{w},{h})")

        # Step 8: Save to output path
        cv2.imwrite(str(output_path), vis_image)
        print(f"\nSaved result: {output_path}")

    return len(nines)


def main():
    # Hard-coded paths
    image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_phase1_preprocessed.png"
    output_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_phase1_preprocessed_detected_9s.png"

    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    count = detect_digit_9(image_path, output_path)

    if count == 0:
        print("\nNo '9' digits detected.")


if __name__ == "__main__":
    main()
