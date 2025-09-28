"""
Debug script to investigate why the second "9 1/4" is being read as "914"
"""
import cv2
import base64
import requests
import os
from dotenv import load_dotenv
import sys

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

def zoom_and_ocr(image_path, x, y, api_key, label):
    """Zoom in on a specific location and run OCR"""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Create zoomed region
    padding = 100
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(w, int(x + padding))
    y2 = min(h, int(y + padding))

    cropped = image[y1:y2, x1:x2]
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Save for inspection
    cv2.imwrite(f"debug_{label}.png", zoomed)
    print(f"Saved: debug_{label}.png")

    # Run OCR
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
                print(f"\nFull text in zoomed region: '{full_text.replace(chr(10), ' | ')}'")

                # Show individual items
                print("\nIndividual items detected:")
                for ann in annotations[1:]:
                    text = ann['description']
                    vertices = ann.get('boundingPoly', {}).get('vertices', [])
                    if vertices:
                        x_avg = sum(v.get('x', 0) for v in vertices) / 4
                        y_avg = sum(v.get('y', 0) for v in vertices) / 4
                        print(f"  '{text}' at ({x_avg:.0f}, {y_avg:.0f})")

def main():
    API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')
    if not API_KEY:
        print("Set GOOGLE_VISION_API_KEY")
        return

    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3.png"

    print("="*60)
    print("DEBUGGING SECOND '9 1/4' (detected as '914')")
    print("="*60)

    # Position where "914" was detected
    print("\n[1] Checking position (701, 1264) where '914' was detected:")
    zoom_and_ocr(image_path, 701, 1264, API_KEY, "914_position")

    # Position of the first "9 1/4" for comparison
    print("\n[2] Checking first '9 1/4' at (688, 1412) for comparison:")
    zoom_and_ocr(image_path, 715, 1413, API_KEY, "first_9_1_4")

    # Try slightly different position for the second one
    print("\n[3] Trying adjusted position around the second '9 1/4':")
    zoom_and_ocr(image_path, 701, 1250, API_KEY, "adjusted_position")

if __name__ == "__main__":
    main()