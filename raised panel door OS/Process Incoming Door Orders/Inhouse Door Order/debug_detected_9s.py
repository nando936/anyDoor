#!/usr/bin/env python3
"""
Debug script to show what the 9 detector is actually finding.
"""

import cv2
import numpy as np

# Load the detected result
result_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_phase1_preprocessed_detected_9s.png"
result = cv2.imread(result_path)

# Show the detection coordinates from the last run
detections = [
    (274, 1157, 22, 32, 1.000),
    (609, 1092, 22, 31, 0.970),
    (947, 499, 22, 31, 0.970),
    (1037, 499, 22, 31, 0.925),
    (150, 380, 14, 16, 0.835)
]

# Create visualization with rectangles AND labels
vis = result.copy()

for i, (x, y, w, h, conf) in enumerate(detections):
    # Draw rectangle
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw center point
    center_x = x + w // 2
    center_y = y + h // 2
    cv2.circle(vis, (center_x, center_y), 3, (255, 0, 0), -1)

    # Add label
    label = f"#{i+1} ({x},{y})"
    cv2.putText(vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Save
output = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_debug_detections.png"
cv2.imwrite(output, vis)
print(f"Saved: {output}")

print("\nDetected regions:")
for i, (x, y, w, h, conf) in enumerate(detections):
    center_x = x + w // 2
    center_y = y + h // 2
    print(f"  #{i+1}: bbox=({x},{y}) size={w}x{h} center=({center_x},{center_y}) conf={conf:.3f}")
