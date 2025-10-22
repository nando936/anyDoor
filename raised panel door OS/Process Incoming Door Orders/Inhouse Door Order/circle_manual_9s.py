#!/usr/bin/env python3
"""
Circle the 9s that were visually identified in the image.
"""

import cv2

# Load image
image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_phase1_preprocessed.png"
image = cv2.imread(image_path)

# Manually identified 9 locations (x, y, radius)
detected_9s = [
    (228, 585, 25, "9 with arrow"),
    (145, 898, 25, "9 in left bottom"),
]

# Draw circles
for x, y, radius, label in detected_9s:
    cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
    print(f"Circled: {label} at ({x}, {y})")

# Save
output_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_manual_9s.png"
cv2.imwrite(output_path, image)
print(f"\nSaved: {output_path}")
