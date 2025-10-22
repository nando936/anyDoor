#!/usr/bin/env python3
"""
Simplified detector: Only detect contours with holes (inner contours).
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def detect_holes(image_path, output_path=None):
    """
    Detect all contours that have holes inside them.
    """
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image")
        return []

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if needed
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} total contours")

    detections = []

    # Check each contour for holes
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Filter by area - only small to medium shapes (individual digits)
        if area < 100 or area > 2000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Filter by dimensions - only digit-sized shapes
        if w < 10 or h < 15 or w > 80 or h > 100:
            continue

        # Check for holes (child contours)
        has_hole = False
        hole_count = 0

        if hierarchy is not None:
            child_idx = hierarchy[0][idx][2]  # First child
            while child_idx != -1:
                hole_count += 1
                child_idx = hierarchy[0][child_idx][0]  # Next sibling

            if hole_count > 0:
                has_hole = True

        # Check if shape looks like a 9
        # 9 characteristics: taller than wide, has 1 hole, rounded top
        aspect_ratio = w / h if h > 0 else 0
        is_tall = 0.4 < aspect_ratio < 0.9
        has_one_hole = hole_count == 1

        # Check if top is rounded (not straight like "1")
        # Extract top third of the contour
        roi = binary[y:y+h, x:x+w]
        top_third_height = h // 3
        top_roi = roi[:top_third_height, :]

        # Count white pixels in top third rows
        top_widths = []
        for row in top_roi:
            white_pixels = np.sum(row > 0)
            top_widths.append(white_pixels)

        # Rounded top has varying width (curves in/out)
        # Straight top (like "1") has consistent width
        has_rounded_top = False
        if len(top_widths) > 5:
            width_variance = np.std(top_widths)
            width_mean = np.mean(top_widths)
            coefficient_of_variation = width_variance / width_mean if width_mean > 0 else 0
            # Rounded shape has higher variation in width
            has_rounded_top = coefficient_of_variation > 0.15

        # Accept if: has hole AND tall aspect ratio AND rounded top
        if has_hole and is_tall and has_rounded_top:
            detections.append({
                'bbox': (x, y, w, h),
                'hole_count': hole_count,
                'area': area,
                'aspect': aspect_ratio,
                'roundness': coefficient_of_variation,
                'match_type': '9-like (rounded)' if has_one_hole else 'digit (rounded)'
            })
        elif has_hole and is_tall:
            # Keep straight tops for comparison
            detections.append({
                'bbox': (x, y, w, h),
                'hole_count': hole_count,
                'area': area,
                'aspect': aspect_ratio,
                'roundness': coefficient_of_variation if 'coefficient_of_variation' in locals() else 0,
                'match_type': 'straight top'
            })

    print(f"\nDetected {len(detections)} regions with holes")

    # Visualize
    if len(detections) > 0:
        vis_image = image.copy()

        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']

            # Calculate center and radius
            center_x = x + w // 2
            center_y = y + h // 2
            radius = int(np.sqrt(w**2 + h**2) / 2) + 5

            # Draw circle
            cv2.circle(vis_image, (center_x, center_y), radius, (0, 255, 0), 2)

            print(f"  #{i+1}: bbox=({x},{y},{w},{h}), holes={det['hole_count']}, aspect={det['aspect']:.2f}, round={det['roundness']:.2f}, type={det['match_type']}")

        # Save
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
            print(f"\nSaved result: {output_path}")
        else:
            input_path = Path(image_path)
            auto_output = input_path.parent / f"{input_path.stem}_holes_detected.png"
            cv2.imwrite(str(auto_output), vis_image)
            print(f"\nSaved result: {auto_output}")

    return detections


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_holes_only.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    detections = detect_holes(image_path, output_path)

    if len(detections) == 0:
        print("\nNo holes detected.")


if __name__ == "__main__":
    main()
