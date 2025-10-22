#!/usr/bin/env python3
"""
Test script: Load page 5 phase1 preprocessed image and draw circles
around all detected text and numbers using OpenCV.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

def main():
    # Input: phase1 preprocessed image (binary: white text on black background)
    preprocessed_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_phase1_preprocessed.png"

    print(f"Loading preprocessed image: {preprocessed_path}")

    # Load the preprocessed image
    preprocessed = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)
    if preprocessed is None:
        print(f"ERROR: Could not load image from {preprocessed_path}")
        sys.exit(1)

    print(f"Image loaded: {preprocessed.shape[1]}x{preprocessed.shape[0]} pixels")

    # Use connected components to find all text regions
    print("\nDetecting text regions using connected components...")

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(preprocessed, connectivity=8)

    print(f"Found {num_labels - 1} components (excluding background)")

    # Filter components by area
    min_area = 20  # Lower threshold to catch smaller text
    max_area = 100000  # Higher threshold
    text_regions = []

    # Skip label 0 (background)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:
            text_regions.append((x, y, w, h, area))

    print(f"Filtered to {len(text_regions)} text regions (area {min_area}-{max_area})")

    # Create color version for visualization
    vis_image = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)

    # Draw circles around each text region
    for x, y, w, h, area in text_regions:
        # Calculate center
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate radius (use diagonal distance + padding)
        radius = int(np.sqrt(w**2 + h**2) / 2) + 5

        # Draw green circle
        cv2.circle(vis_image, (center_x, center_y), radius, (0, 255, 0), 2)

    # Save result to same directory as input
    input_dir = Path(preprocessed_path).parent
    output_path = input_dir / "page-5_opencv_circles.png"
    cv2.imwrite(str(output_path), vis_image)
    print(f"\nSaved result: {output_path}")

    # Print statistics
    if text_regions:
        areas = [area for x, y, w, h, area in text_regions]
        print(f"\nStatistics:")
        print(f"  Total regions: {len(text_regions)}")
        print(f"  Min area: {min(areas)} px")
        print(f"  Max area: {max(areas)} px")
        print(f"  Avg area: {sum(areas) / len(areas):.1f} px")


if __name__ == "__main__":
    main()
