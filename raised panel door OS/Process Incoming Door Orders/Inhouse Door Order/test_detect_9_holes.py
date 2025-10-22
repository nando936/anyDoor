#!/usr/bin/env python3
"""
Test script to detect holes in the number 9 using OpenCV contour detection.
Looks for circular holes that are characteristic of the number 9.
"""

import cv2
import numpy as np

def detect_9_holes(image_path, output_path):
    """
    Detect holes in number 9s using OpenCV contour detection.

    Args:
        image_path: Path to input image
        output_path: Path to save output image with detected holes circled
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    print(f"Image shape: {img.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold - invert so holes are white on black background
    # Using OTSU for automatic threshold determination
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    print(f"Applied binary threshold (inverted)")

    # Apply morphological operations to clean up the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    print(f"Applied morphological closing (3x3 ellipse, 2 iterations)")

    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} total contours")

    # Draw ALL contours - no filters
    detected_holes = []

    for i, contour in enumerate(contours):
        # Get contour properties
        area = cv2.contourArea(contour)

        # Get center point
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Calculate average radius
            radius = int(np.sqrt(area / np.pi)) if area > 0 else 5

            detected_holes.append({
                'center': (cx, cy),
                'radius': radius,
                'area': area
            })

            print(f"Contour {len(detected_holes)}: center=({cx}, {cy}), radius={radius}, area={area:.1f}")

    print(f"\nDetected {len(detected_holes)} potential 9 holes")

    # Load original page-5.png image for drawing
    original_img = cv2.imread("/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5.png")
    if original_img is None:
        print("Error: Could not load original page-5.png")
        return

    # Draw results on original image
    output_img = original_img.copy()

    for idx, hole in enumerate(detected_holes, 1):
        cx, cy = hole['center']
        radius = hole['radius']

        # Draw circle around hole (green)
        cv2.circle(output_img, (cx, cy), radius + 5, (0, 255, 0), 2)

        # Draw center point (red)
        cv2.circle(output_img, (cx, cy), 3, (0, 0, 255), -1)

        # Draw number label (blue text)
        label = str(idx)
        cv2.putText(output_img, label, (cx - 15, cy - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Save output image to same folder as input
    output_folder = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages"
    output_path = f"{output_folder}/page-5_detected_9_holes.png"
    cv2.imwrite(output_path, output_img)
    print(f"\nSaved output image to: {output_path}")

    return detected_holes

if __name__ == "__main__":
    input_image = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5.png"
    output_image = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_detected_9_holes.png"

    print("=" * 60)
    print("OpenCV 9 Hole Detection Test")
    print("=" * 60)
    print(f"Input:  {input_image}")
    print(f"Output: {output_image}")
    print("=" * 60)

    holes = detect_9_holes(input_image, output_image)

    print("=" * 60)
    print("Detection complete!")
    print("=" * 60)
