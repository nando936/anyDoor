"""
Debug what OCR is seeing on page 1
"""
import cv2
import os

image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_1.png"

# Load and check the area around coordinates (910, 1568) where "20.34" was detected
img = cv2.imread(image_path)

# Draw where OCR thinks it found "20.34"
x, y = 910, 1568
cv2.circle(img, (x, y), 10, (0, 0, 255), 2)  # Red circle
cv2.putText(img, "20.34?", (x+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Draw where OCR found "20"
x2, y2 = 866, 1576
cv2.circle(img, (x2, y2), 10, (0, 255, 0), 2)  # Green circle
cv2.putText(img, "20", (x2+15, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Draw where OCR found "3/16"
x3, y3 = 972, 1574
cv2.circle(img, (x3, y3), 10, (255, 0, 0), 2)  # Blue circle
cv2.putText(img, "3/16", (x3+15, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Crop the region to focus on this area
crop_margin = 200
x_min = max(0, min(x, x2, x3) - crop_margin)
x_max = min(img.shape[1], max(x, x2, x3) + crop_margin)
y_min = max(0, min(y, y2, y3) - crop_margin)
y_max = min(img.shape[0], max(y, y2, y3) + crop_margin)

cropped = img[y_min:y_max, x_min:x_max]
cv2.imwrite("debug_page1_ocr_locations.png", cropped)
print(f"Saved debug image showing OCR detection locations")
print(f"Red circle: '20.34' at ({x}, {y})")
print(f"Green circle: '20' at ({x2}, {y2})")
print(f"Blue circle: '3/16' at ({x3}, {y3})")