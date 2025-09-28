"""
Debug the zoom area where "20.34" is being detected
"""
import cv2
import base64
import requests
import os
import sys

# Image and coordinates from the problem
image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_1.png"
x = 916  # Center of group
y = 1573  # Center of group

print(f"Debugging zoom area at ({x}, {y})")
print("=" * 60)

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("[ERROR] Cannot load image")
    sys.exit(1)

h, w = image.shape[:2]
print(f"Image dimensions: {w}x{h}")

# Create a WIDER crop area to capture the complete measurement
# Original was padding = 100, let's make it wider
padding_x = 200  # Double the horizontal padding to capture full text
padding_y = 50   # Keep vertical padding same
x1 = max(0, int(x - padding_x))
y1 = max(0, int(y - padding_y))
x2 = min(w, int(x + padding_x))
y2 = min(h, int(y + padding_y))

print(f"Crop area: ({x1}, {y1}) to ({x2}, {y2})")
print(f"Crop size: {x2-x1}x{y2-y1} pixels")

# First save the ORIGINAL narrow crop (like the function uses)
padding_original = 100
x1_orig = max(0, int(x - padding_original))
y1_orig = max(0, int(y - padding_original/2))
x2_orig = min(w, int(x + padding_original))
y2_orig = min(h, int(y + padding_original/2))
cropped_original = image[y1_orig:y2_orig, x1_orig:x2_orig]
cv2.imwrite("debug_cropped_narrow.png", cropped_original)
print(f"\n[SAVED] Original narrow crop as: debug_cropped_narrow.png")
print(f"  Narrow crop size: {x2_orig-x1_orig}x{y2_orig-y1_orig} pixels")

# Now crop the WIDER region
cropped = image[y1:y2, x1:x2]

# Save the wider cropped image
cv2.imwrite("debug_cropped_wide.png", cropped)
print(f"[SAVED] Wider crop area as: debug_cropped_wide.png")
print(f"  Wide crop size: {x2-x1}x{y2-y1} pixels")

# Zoom in 3x (same as the function)
zoom_factor = 3
zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
print(f"Zoomed size: {zoomed.shape[1]}x{zoomed.shape[0]} pixels (3x zoom)")

# Save the zoomed image
cv2.imwrite("debug_zoomed_area.png", zoomed)
print(f"[SAVED] Zoomed area as: debug_zoomed_area.png")

# Now try to OCR the zoomed image
api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
if not api_key:
    print("\n[WARNING] No API key found, trying without it...")
    # Try to read from a file if it exists
    key_file = "C:/Users/nando/Projects/anyDoor/vision_api_key.txt"
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
            print("[OK] API key loaded from file")
    else:
        print("[ERROR] No API key available for OCR test")
        print("\nBut you can still examine the saved images:")
        print("  - debug_cropped_area.png (original crop)")
        print("  - debug_zoomed_area.png (3x zoomed)")
        sys.exit(0)

# Encode the zoomed image for Vision API
_, buffer = cv2.imencode('.png', zoomed)
zoomed_content = base64.b64encode(buffer).decode('utf-8')

print("\n" + "=" * 60)
print("Running OCR on zoomed area...")
print("=" * 60)

# Run OCR on the zoomed region
url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
request = {
    "requests": [{
        "image": {"content": zoomed_content},
        "features": [{"type": "TEXT_DETECTION"}]
    }]
}

response = requests.post(url, json=request)

if response.status_code == 200:
    result = response.json()
    if 'responses' in result and result['responses']:
        annotations = result['responses'][0].get('textAnnotations', [])
        if annotations:
            # Get the full text from the zoomed region
            full_text = annotations[0]['description'].strip()
            print(f"\nFULL TEXT FROM ZOOMED OCR:")
            print(f"'{full_text}'")

            print(f"\nINDIVIDUAL TEXT ITEMS:")
            for ann in annotations[1:]:
                text = ann['description']
                vertices = ann['boundingPoly']['vertices']
                x_avg = sum(v.get('x', 0) for v in vertices) / 4
                y_avg = sum(v.get('y', 0) for v in vertices) / 4
                print(f"  '{text}' at ({x_avg:.0f}, {y_avg:.0f})")
        else:
            print("No text found in zoomed area")
    else:
        print("No response from Vision API")
else:
    print(f"Error {response.status_code} from Vision API")
    error_msg = response.json().get('error', {}).get('message', 'Unknown error')
    print(f"Error message: {error_msg}")

print("\n" + "=" * 60)
print("Images saved for inspection:")
print("  - debug_cropped_area.png (original crop)")
print("  - debug_zoomed_area.png (3x zoomed)")
print("=" * 60)