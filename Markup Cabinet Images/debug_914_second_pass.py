"""
Debug script to see exactly what the zoom verification (2nd pass) sees for "914"
This simulates what verify_measurement_with_zoom does
"""
import cv2
import base64
import requests
import os
from dotenv import load_dotenv
import sys

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

def simulate_verify_measurement_zoom(image_path, x, y, text, api_key):
    """Simulate exactly what verify_measurement_with_zoom does for "914" """
    print(f"Simulating verify_measurement_with_zoom for '{text}' at ({x}, {y})")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Cannot load image")
        return

    h, w = image.shape[:2]

    # Use the SAME padding as the actual function (no group bounds for single item)
    padding_x = 200  # Wider horizontal padding
    padding_y = 50   # Vertical padding
    x1 = max(0, int(x - padding_x))
    y1 = max(0, int(y - padding_y))
    x2 = min(w, int(x + padding_x))
    y2 = min(h, int(y + padding_y))

    print(f"Crop area: ({x1}, {y1}) to ({x2}, {y2})")

    # Crop the region
    cropped = image[y1:y2, x1:x2]
    print(f"Cropped size: {cropped.shape}")

    # Zoom in 3x (same as actual function)
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
    print(f"Zoomed size: {zoomed.shape}")

    # Save the EXACT zoomed image that the 2nd pass sees
    output_filename = "debug_914_EXACT_2nd_pass_image.png"
    cv2.imwrite(output_filename, zoomed)
    print(f"\nSAVED: {output_filename} - This is EXACTLY what the 2nd pass OCR sees")

    # Now run OCR on this exact image
    _, buffer = cv2.imencode('.png', zoomed)
    content = base64.b64encode(buffer).decode('utf-8')

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, json=request)

    if response.status_code == 200:
        result = response.json()
        if 'responses' in result and result['responses']:
            annotations = result['responses'][0].get('textAnnotations', [])
            if annotations:
                full_text = annotations[0]['description']
                print(f"\n2ND PASS OCR RESULT:")
                print(f"Full text: '{full_text.replace(chr(10), ' | ')}'")

                # Calculate center point (like the actual function does)
                zoomed_height, zoomed_width = zoomed.shape[:2]
                center_x = zoomed_width / 2
                center_y = zoomed_height / 2
                print(f"\nCenter of zoomed image: ({center_x:.0f}, {center_y:.0f})")

                # Find what's at the center
                print(f"\nText items and their positions:")
                for ann in annotations[1:]:
                    text_item = ann['description']
                    vertices = ann.get('boundingPoly', {}).get('vertices', [])
                    if vertices:
                        x_coords = [v.get('x', 0) for v in vertices]
                        y_coords = [v.get('y', 0) for v in vertices]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)

                        # Check if center is within this text
                        margin = 10
                        if (min_x - margin <= center_x <= max_x + margin and
                            min_y - margin <= center_y <= max_y + margin):
                            print(f"  '{text_item}' - AT CENTER - bounds: ({min_x},{min_y})-({max_x},{max_y})")
                        else:
                            dist_from_center = ((((min_x + max_x)/2) - center_x)**2 +
                                               (((min_y + max_y)/2) - center_y)**2)**0.5
                            print(f"  '{text_item}' - {dist_from_center:.0f}px from center")

def main():
    API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')
    if not API_KEY:
        print("Set GOOGLE_VISION_API_KEY")
        return

    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3.png"

    print("="*70)
    print("EXACT SIMULATION OF 2ND PASS (ZOOM VERIFICATION) FOR '914'")
    print("="*70)
    print()

    # Use the exact position from the debug output
    x = 701
    y = 1264
    text = "914"

    simulate_verify_measurement_zoom(image_path, x, y, text, API_KEY)

    print("\n" + "="*70)
    print("The saved image 'debug_914_EXACT_2nd_pass_image.png' shows")
    print("EXACTLY what the zoom verification sees in the 2nd pass")

if __name__ == "__main__":
    main()