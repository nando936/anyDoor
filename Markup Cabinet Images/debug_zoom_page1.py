"""
Debug zoom verification for page 1 "20 3/16" issue
"""
import os
import sys
sys.path.append("C:/Users/nando/Projects/anyDoor/Markup Cabinet Images/")
from measurement_based_detector import verify_measurement_with_zoom

# Test the zoom verification at the coordinates where "20.34 20 3/16" was found
image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_1.png"
api_key = os.getenv('GOOGLE_CLOUD_API_KEY')

# Coordinates from the debug output
x = 916  # avg_x for the group
y = 1573  # avg_y for the group
original_text = "20.34 20 3/16"

print(f"Testing zoom verification at ({x}, {y})")
print(f"Original grouped text: '{original_text}'")
print("-" * 60)

result = verify_measurement_with_zoom(image_path, x, y, original_text, api_key)
print(f"\nZoom verification result: '{result}'")