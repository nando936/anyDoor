#!/usr/bin/env python3
"""
Test script to check if OpenCV can detect the missing "9" in the preprocessed image
"""
import cv2
import numpy as np

# Load the preprocessed image
image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"ERROR: Could not load image from {image_path}")
    exit(1)

print(f"Loaded image: {image.shape}")
print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

# Apply threshold to get binary image (white text on black background)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\nFound {len(contours)} total contours")

# Same size thresholds as OpenCV supplemental detection
MIN_WIDTH = 10
MAX_WIDTH = 120
MIN_HEIGHT = 10
MAX_HEIGHT = 45
MIN_AREA = 100

# M7 position from debug output: (1115, 666)
# The "9" should be below M7, roughly at (1115, 700-850)
TARGET_X_MIN = 1000
TARGET_X_MAX = 1200
TARGET_Y_MIN = 670
TARGET_Y_MAX = 900

print(f"\nLooking for regions near M7 (x={TARGET_X_MIN}-{TARGET_X_MAX}, y={TARGET_Y_MIN}-{TARGET_Y_MAX})")
print(f"Size filters: width={MIN_WIDTH}-{MAX_WIDTH}px, height={MIN_HEIGHT}-{MAX_HEIGHT}px, area>={MIN_AREA}px²")
print("\n" + "="*80)

valid_regions = []
regions_near_9 = []

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    center_x = x + w/2
    center_y = y + h/2

    # Check size filters
    size_ok = (MIN_WIDTH <= w <= MAX_WIDTH and
               MIN_HEIGHT <= h <= MAX_HEIGHT and
               area >= MIN_AREA)

    # Check if near M7/9 location
    near_target = (TARGET_X_MIN <= center_x <= TARGET_X_MAX and
                   TARGET_Y_MIN <= center_y <= TARGET_Y_MAX)

    if size_ok:
        valid_regions.append({
            'id': i,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'center': (int(center_x), int(center_y)),
            'area': int(area),
            'near_target': near_target
        })

        if near_target:
            regions_near_9.append(valid_regions[-1])

print(f"\nRegions that pass size filters: {len(valid_regions)}")
print(f"Regions near the '9' location: {len(regions_near_9)}")

if regions_near_9:
    print("\n" + "="*80)
    print("REGIONS NEAR THE '9' LOCATION:")
    print("="*80)
    for region in regions_near_9:
        print(f"Region #{region['id']}:")
        print(f"  Position: ({region['x']}, {region['y']})")
        print(f"  Size: {region['w']}x{region['h']} pixels")
        print(f"  Center: {region['center']}")
        print(f"  Area: {region['area']}px²")
        print(f"  ← CANDIDATE FOR THE '9'")
        print()
else:
    print("\nNO regions found near the '9' location")
    print("\nClosest valid regions to target area:")
    # Sort by distance to center of target area
    target_center_x = (TARGET_X_MIN + TARGET_X_MAX) / 2
    target_center_y = (TARGET_Y_MIN + TARGET_Y_MAX) / 2

    for region in sorted(valid_regions,
                        key=lambda r: ((r['center'][0]-target_center_x)**2 +
                                      (r['center'][1]-target_center_y)**2)**0.5)[:5]:
        dist = ((region['center'][0]-target_center_x)**2 +
                (region['center'][1]-target_center_y)**2)**0.5
        print(f"Region #{region['id']}: center={region['center']}, size={region['w']}x{region['h']}, distance={dist:.0f}px")

# Create visualization
vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw all valid regions in blue
for region in valid_regions:
    cv2.rectangle(vis_image,
                 (region['x'], region['y']),
                 (region['x']+region['w'], region['y']+region['h']),
                 (255, 0, 0), 2)

# Draw regions near '9' in green
for region in regions_near_9:
    cv2.rectangle(vis_image,
                 (region['x'], region['y']),
                 (region['x']+region['w'], region['y']+region['h']),
                 (0, 255, 0), 3)
    cv2.putText(vis_image, f"#{region['id']}",
               (region['x'], region['y']-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Draw target search area
cv2.rectangle(vis_image,
             (TARGET_X_MIN, TARGET_Y_MIN),
             (TARGET_X_MAX, TARGET_Y_MAX),
             (0, 0, 255), 2)
cv2.putText(vis_image, "Search area for '9'",
           (TARGET_X_MIN, TARGET_Y_MIN-10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Save visualization
output_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/test_opencv_9_detection.png"
cv2.imwrite(output_path, vis_image)
print(f"\n[SAVED] Visualization: {output_path}")
print("  Blue boxes = valid regions")
print("  Green boxes = regions near '9' location")
print("  Red box = search area")
