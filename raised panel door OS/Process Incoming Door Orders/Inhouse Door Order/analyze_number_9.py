#!/usr/bin/env python3
"""
Analyze the number 9 screenshot to extract characteristics
for OpenCV detection in other images.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def analyze_digit_9(image_path):
    """
    Comprehensive analysis of digit 9 characteristics.
    """
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return None

    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try different thresholding methods
    print("\n=== THRESHOLDING ===")

    # Method 1: Simple threshold
    _, binary1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    print("Binary threshold (127)")

    # Method 2: Otsu's threshold
    _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu's threshold")

    # Method 3: Adaptive threshold
    binary3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    print("Adaptive threshold")

    # Use Otsu for analysis (usually best)
    binary = binary2

    # Invert if needed (we want white digit on black background)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
        print("Inverted binary (white digit on black background)")

    # Find contours
    print("\n=== CONTOUR ANALYSIS ===")
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")

    # Find the main contour (largest area)
    if len(contours) == 0:
        print("ERROR: No contours found!")
        return None

    main_contour = max(contours, key=cv2.contourArea)
    main_idx = contours.index(main_contour)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(main_contour)
    print(f"\nMain contour bounding box: x={x}, y={y}, w={w}, h={h}")

    # Extract ROI
    roi = binary[y:y+h, x:x+w]

    # === CHARACTERISTIC 1: Aspect Ratio ===
    aspect_ratio = w / h if h > 0 else 0
    print(f"\n1. ASPECT RATIO: {aspect_ratio:.3f}")
    print(f"   Width: {w}px, Height: {h}px")
    print(f"   Expected for '9': 0.4-0.8 (taller than wide)")

    # === CHARACTERISTIC 2: Area ===
    area = cv2.contourArea(main_contour)
    bbox_area = w * h
    extent = area / bbox_area if bbox_area > 0 else 0
    print(f"\n2. AREA METRICS:")
    print(f"   Contour area: {area:.0f}px²")
    print(f"   Bounding box area: {bbox_area}px²")
    print(f"   Extent (fill ratio): {extent:.3f}")
    print(f"   Expected for '9': 0.4-0.7 (moderate fill)")

    # === CHARACTERISTIC 3: Holes (Inner Contours) ===
    print(f"\n3. INNER CONTOURS (HOLES):")
    if hierarchy is not None:
        # Check for children of main contour
        inner_contours = []
        child_idx = hierarchy[0][main_idx][2]  # First child
        while child_idx != -1:
            inner_contours.append(contours[child_idx])
            child_idx = hierarchy[0][child_idx][0]  # Next sibling

        print(f"   Number of holes: {len(inner_contours)}")
        if len(inner_contours) > 0:
            hole_contour = inner_contours[0]
            hole_area = cv2.contourArea(hole_contour)
            hole_ratio = hole_area / area if area > 0 else 0
            hx, hy, hw, hh = cv2.boundingRect(hole_contour)
            hole_center_y = hy + hh/2
            hole_position = hole_center_y / h if h > 0 else 0

            print(f"   Hole area: {hole_area:.0f}px²")
            print(f"   Hole/digit ratio: {hole_ratio:.3f}")
            print(f"   Hole vertical position: {hole_position:.3f} (0=top, 1=bottom)")
            print(f"   Expected for '9': 1 hole, position ~0.3-0.5 (top half)")
    else:
        print(f"   No hierarchy information")

    # === CHARACTERISTIC 4: Moments & Center of Mass ===
    print(f"\n4. MOMENTS & BALANCE:")
    moments = cv2.moments(main_contour)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']

        # Normalize to ROI coordinates
        cx_norm = cx / w if w > 0 else 0
        cy_norm = cy / h if h > 0 else 0

        print(f"   Center of mass: ({cx:.1f}, {cy:.1f})")
        print(f"   Normalized (0-1): ({cx_norm:.3f}, {cy_norm:.3f})")
        print(f"   Expected for '9': cy_norm ~0.35-0.5 (top-heavy)")

        # Check if top-heavy
        top_heavy = cy_norm < 0.5
        print(f"   Top-heavy: {top_heavy}")

    # === CHARACTERISTIC 5: Perimeter & Solidity ===
    print(f"\n5. SHAPE COMPLEXITY:")
    perimeter = cv2.arcLength(main_contour, True)
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    print(f"   Perimeter: {perimeter:.1f}px")
    print(f"   Convex hull area: {hull_area:.0f}px²")
    print(f"   Solidity: {solidity:.3f}")
    print(f"   Expected for '9': 0.7-0.9 (somewhat convex)")

    # === CHARACTERISTIC 6: Hu Moments (Shape Signature) ===
    print(f"\n6. HU MOMENTS (Shape Signature):")
    hu_moments = cv2.HuMoments(moments).flatten()
    for i, hu in enumerate(hu_moments):
        print(f"   Hu[{i}]: {hu:.6e}")

    # === CHARACTERISTIC 7: Skeleton Analysis ===
    print(f"\n7. SKELETON ANALYSIS:")
    skeleton = cv2.ximgproc.thinning(roi)

    # Count endpoints and junctions
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton, cv2.CV_8U, kernel)

    endpoints = np.sum(filtered == 11)  # 1 neighbor + 10 center
    junctions = np.sum(filtered >= 13)   # 3+ neighbors + 10 center

    print(f"   Endpoints: {endpoints}")
    print(f"   Junctions: {junctions}")
    print(f"   Expected for '9': 2-4 endpoints, 1-3 junctions")

    # === CHARACTERISTIC 8: Vertical Profile ===
    print(f"\n8. VERTICAL DENSITY PROFILE:")
    vertical_sum = np.sum(roi, axis=1) / 255  # Pixels per row
    h_third = h // 3

    top_density = np.mean(vertical_sum[:h_third]) if h_third > 0 else 0
    mid_density = np.mean(vertical_sum[h_third:2*h_third]) if h_third > 0 else 0
    bot_density = np.mean(vertical_sum[2*h_third:]) if h_third > 0 else 0

    print(f"   Top third density: {top_density:.1f}px/row")
    print(f"   Middle third density: {mid_density:.1f}px/row")
    print(f"   Bottom third density: {bot_density:.1f}px/row")
    print(f"   Expected for '9': top > mid > bot (decreasing)")

    # === VISUALIZATION ===
    print("\n=== GENERATING VISUALIZATIONS ===")

    # Create composite visualization
    vis_height = max(image.shape[0], 800)
    vis_width = image.shape[1] * 3
    vis_image = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

    # Column 1: Original + contour
    vis1 = image.copy()
    cv2.drawContours(vis1, [main_contour], -1, (0, 255, 0), 2)
    cv2.rectangle(vis1, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if moments['m00'] > 0:
        cv2.circle(vis1, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    vis_image[:image.shape[0], :image.shape[1]] = vis1

    # Column 2: Binary + skeleton
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # Overlay skeleton in red
    skeleton_full = np.zeros_like(binary)
    skeleton_full[y:y+h, x:x+w] = skeleton
    binary_color[skeleton_full > 0] = [0, 0, 255]
    vis_image[:image.shape[0], image.shape[1]:image.shape[1]*2] = binary_color

    # Column 3: ROI enlarged
    roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    roi_enlarged = cv2.resize(roi_color, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    vis_image[:image.shape[0], image.shape[1]*2:] = roi_enlarged

    # Save visualizations
    output_dir = Path(image_path).parent
    vis_path = output_dir / "number_9_analysis.png"
    cv2.imwrite(str(vis_path), vis_image)
    print(f"Saved visualization: {vis_path}")

    # Save just the extracted ROI
    roi_path = output_dir / "number_9_template.png"
    cv2.imwrite(str(roi_path), roi)
    print(f"Saved template: {roi_path}")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY: Digit '9' Detection Criteria")
    print("="*60)
    print(f"✓ Aspect ratio: 0.4-0.8 (taller)")
    print(f"✓ Has 1 inner hole in top half")
    print(f"✓ Top-heavy (center of mass y < 0.5)")
    print(f"✓ Extent: 0.4-0.7")
    print(f"✓ Solidity: 0.7-0.9")
    print(f"✓ Vertical density: decreasing (top > mid > bot)")
    print("="*60)

    return {
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'solidity': solidity,
        'has_hole': len(inner_contours) > 0 if hierarchy is not None else False,
        'top_heavy': cy_norm < 0.5 if moments['m00'] > 0 else False,
        'hu_moments': hu_moments,
        'roi': roi
    }


def main():
    # Find the screenshot in Pictures folder
    image_path = Path.home() / "Pictures" / "Screenshot_2025-10-17_07-07-19.png"

    if not image_path.exists():
        print(f"ERROR: Could not find {image_path}")
        sys.exit(1)

    # Analyze the digit
    results = analyze_digit_9(image_path)

    if results is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
