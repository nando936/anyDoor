#!/usr/bin/env python3
"""
Test script: Detect all text and numbers on page 5 using OpenCV
after phase 1 preprocessing and draw circles around them.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Import the preprocessing function
from image_preprocessing import apply_hsv_preprocessing

def detect_text_regions(binary_image, min_area=50, max_area=50000):
    """
    Detect text regions using contour detection on binary image.

    Args:
        binary_image: Binary image (white text on black background)
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider

    Returns:
        List of (x, y, w, h) bounding boxes for detected text regions
    """
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            text_regions.append((x, y, w, h))

    return text_regions


def main():
    # Image path
    image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page_5.png"

    print(f"Loading image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        sys.exit(1)

    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # Apply phase 1 preprocessing
    print("\nApplying phase 1 HSV preprocessing...")
    preprocessed = apply_hsv_preprocessing(image)

    # Save preprocessed image
    preproc_path = "page_5_phase1_preprocessed.png"
    cv2.imwrite(preproc_path, preprocessed)
    print(f"Saved preprocessed image: {preproc_path}")

    # Detect text regions
    print("\nDetecting text regions...")
    text_regions = detect_text_regions(preprocessed, min_area=50, max_area=50000)
    print(f"Found {len(text_regions)} text regions")

    # Create visualization image
    vis_image = image.copy()

    # Draw circles around each detected text region
    for i, (x, y, w, h) in enumerate(text_regions):
        # Calculate center and radius
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2 + 5  # Add padding

        # Draw circle
        cv2.circle(vis_image, (center_x, center_y), radius, (0, 255, 0), 2)

        # Optionally draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Save result
    output_path = "page_5_text_circles.png"
    cv2.imwrite(output_path, vis_image)
    print(f"\nSaved result: {output_path}")

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total regions detected: {len(text_regions)}")
    if text_regions:
        areas = [w * h for x, y, w, h in text_regions]
        print(f"  Min area: {min(areas)} pixels")
        print(f"  Max area: {max(areas)} pixels")
        print(f"  Avg area: {sum(areas) / len(areas):.1f} pixels")

    # Also create a version showing preprocessed + circles
    preprocessed_color = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(text_regions):
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2 + 5
        cv2.circle(preprocessed_color, (center_x, center_y), radius, (0, 255, 0), 2)

    preproc_circles_path = "page_5_preprocessed_with_circles.png"
    cv2.imwrite(preproc_circles_path, preprocessed_color)
    print(f"  Saved preprocessed with circles: {preproc_circles_path}")


if __name__ == "__main__":
    main()
