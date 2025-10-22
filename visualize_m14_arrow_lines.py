#!/usr/bin/env python3
"""
Visualize M14's two arrow lines to see if they're the same line or a real arrow.

Line 1 (BLUE left leg): coords=(130,84)-(143,79), angle=339.0°
Line 2 (GREEN right leg): coords=(130,85)-(142,91), angle=26.6°
"""

import cv2
import numpy as np

# Create a blank white image (ROI size was 144x99 pixels)
# Make it larger for visibility
scale = 5
roi_width = 144
roi_height = 99
img_width = roi_width * scale
img_height = roi_height * scale

# Create white background
img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

# Line coordinates (from bottom V-ROI)
# Line 1 (BLUE - down-left leg, 285-360°)
x1_line1, y1_line1 = 130, 84
x2_line1, y2_line1 = 143, 79

# Line 2 (GREEN - down-right leg, 20-88°)
x1_line2, y1_line2 = 130, 85
x2_line2, y2_line2 = 142, 91

# Scale coordinates
x1_line1_s, y1_line1_s = x1_line1 * scale, y1_line1 * scale
x2_line1_s, y2_line1_s = x2_line1 * scale, y2_line1 * scale
x1_line2_s, y1_line2_s = x1_line2 * scale, y1_line2 * scale
x2_line2_s, y2_line2_s = x2_line2 * scale, y2_line2 * scale

# Draw Line 1 in BLUE (down-left leg)
cv2.line(img, (x1_line1_s, y1_line1_s), (x2_line1_s, y2_line1_s), (255, 0, 0), 3)

# Draw Line 2 in GREEN (down-right leg)
cv2.line(img, (x1_line2_s, y1_line2_s), (x2_line2_s, y2_line2_s), (0, 255, 0), 3)

# Mark start points with circles
cv2.circle(img, (x1_line1_s, y1_line1_s), 8, (255, 0, 0), -1)  # BLUE circle
cv2.circle(img, (x1_line2_s, y1_line2_s), 8, (0, 255, 0), -1)  # GREEN circle

# Mark end points with circles
cv2.circle(img, (x2_line1_s, y2_line1_s), 8, (255, 0, 0), 2)  # BLUE circle (hollow)
cv2.circle(img, (x2_line2_s, y2_line2_s), 8, (0, 255, 0), 2)  # GREEN circle (hollow)

# Add text labels
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 2

# Label Line 1 (BLUE - LEFT leg)
label1 = "LEFT leg (339.0°)"
cv2.putText(img, label1, (10, 30), font, font_scale, (255, 0, 0), thickness)
cv2.putText(img, f"Start: ({x1_line1},{y1_line1})", (10, 60), font, font_scale, (255, 0, 0), thickness)
cv2.putText(img, f"End: ({x2_line1},{y2_line1})", (10, 90), font, font_scale, (255, 0, 0), thickness)
cv2.putText(img, "Length: 13.9px", (10, 120), font, font_scale, (255, 0, 0), thickness)

# Label Line 2 (GREEN - RIGHT leg)
label2 = "RIGHT leg (26.6°)"
cv2.putText(img, label2, (10, 160), font, font_scale, (0, 255, 0), thickness)
cv2.putText(img, f"Start: ({x1_line2},{y1_line2})", (10, 190), font, font_scale, (0, 255, 0), thickness)
cv2.putText(img, f"End: ({x2_line2},{y2_line2})", (10, 220), font, font_scale, (0, 255, 0), thickness)
cv2.putText(img, "Length: 13.4px", (10, 250), font, font_scale, (0, 255, 0), thickness)

# Add distance info
cv2.putText(img, "Distance: 6.5px", (10, 300), font, font_scale, (0, 0, 0), thickness)

# Add title
title = "M14 Arrow Lines - Are They The Same Line?"
cv2.putText(img, title, (img_width // 2 - 300, img_height - 20),
            font, 0.8, (0, 0, 0), 2)

# Save the image
output_path = "/home/nando/projects/anyDoor/m14_arrow_visualization.png"
cv2.imwrite(output_path, img)

print(f"Visualization saved to: {output_path}")
print()
print("Line 1 (BLUE - LEFT leg):")
print(f"  Start: ({x1_line1}, {y1_line1})")
print(f"  End:   ({x2_line1}, {y2_line1})")
print(f"  Angle: 339.0°")
print(f"  Length: 13.9px")
print()
print("Line 2 (GREEN - RIGHT leg):")
print(f"  Start: ({x1_line2}, {y1_line2})")
print(f"  End:   ({x2_line2}, {y2_line2})")
print(f"  Angle: 26.6°")
print(f"  Length: 13.4px")
print()
print("Start points are only 1 pixel apart vertically: (130,84) vs (130,85)")
print("End points are only 1 pixel apart: (143,79) vs (142,91)")
print()
print("These look like the SAME LINE detected twice, not two arrow legs!")
