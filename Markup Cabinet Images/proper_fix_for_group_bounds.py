"""
The CORRECT fix: Pass group bounds to verify_measurement_with_zoom
and extract text from the GROUP'S BOUNDING BOX, not just center point.
"""

def verify_measurement_with_zoom_PROPER_FIX(image_path, x, y, text, api_key, group_bounds=None):
    """
    Fixed version that extracts text from the group's bounding box area.

    Args:
        image_path: Path to the image
        x, y: Center coordinates of the group (for backwards compatibility)
        text: Original group text (e.g., "20 ラ 17 116")
        api_key: Google Vision API key
        group_bounds: Dict with 'left', 'right', 'top', 'bottom' of the group in original image
    """
    import cv2
    import base64
    import requests
    import re

    print(f"    Zoom verifying: '{text}' at center ({x:.0f}, {y:.0f})")

    # If we have group bounds, use them to determine crop area
    if group_bounds:
        print(f"      Group bounds: left={group_bounds['left']:.0f}, right={group_bounds['right']:.0f}, "
              f"top={group_bounds['top']:.0f}, bottom={group_bounds['bottom']:.0f}")

        # Use group bounds to determine what to crop
        # Add some padding around the group
        padding = 50
        x1 = int(group_bounds['left'] - padding)
        x2 = int(group_bounds['right'] + padding)
        y1 = int(group_bounds['top'] - padding)
        y2 = int(group_bounds['bottom'] + padding)
    else:
        # Fallback to center-based crop
        padding_x = 200
        padding_y = 50
        x1 = int(x - padding_x)
        x2 = int(x + padding_x)
        y1 = int(y - padding_y)
        y2 = int(y + padding_y)

    # Load image and apply bounds
    image = cv2.imread(image_path)
    if image is None:
        return text

    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Crop to the group's region
    cropped = image[y1:y2, x1:x2]

    # Zoom in 3x for better OCR
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Run OCR on zoomed region
    _, buffer = cv2.imencode('.png', zoomed)
    zoomed_content = base64.b64encode(buffer).decode('utf-8')

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
                # The FIRST annotation contains ALL text in the image
                full_text = annotations[0]['description']
                print(f"      Full text in group region: '{full_text.replace(chr(10), ' ')}'")

                # If we zoomed specifically on the group bounds,
                # the full_text should be what's IN the group
                if group_bounds:
                    # Clean up the text and look for measurement patterns
                    full_text_clean = full_text.replace('\n', ' ').strip()

                    # Look for measurement patterns
                    patterns = [
                        r'^(\d+\s+\d+/\d+)',      # "20 1/16" at start
                        r'^(\d+-\d+/\d+)',         # "20-1/16" at start
                        r'^(\d+)\s+(\d+/\d+)',     # "20" "1/16" at start
                    ]

                    for pattern in patterns:
                        match = re.match(pattern, full_text_clean)
                        if match:
                            if match.lastindex == 2:
                                measurement = f"{match.group(1)} {match.group(2)}"
                            else:
                                measurement = match.group(1).replace('-', ' ')

                            # Sanity check
                            first_num = int(re.match(r'^(\d+)', measurement).group(1))
                            if 2 <= first_num <= 100:
                                print(f"      VERIFIED: Group contains measurement '{measurement}'")
                                return measurement

                    # If no pattern matched but we have numbers, return cleaned text
                    if re.search(r'\d+', full_text_clean):
                        # Extract just the measurement-like parts
                        nums_and_fractions = re.findall(r'\d+(?:\s+\d+/\d+)?|\d+/\d+', full_text_clean)
                        if nums_and_fractions:
                            combined = ' '.join(nums_and_fractions)
                            print(f"      Extracted numbers from group: '{combined}'")
                            return combined

                # Fallback: look for measurement anywhere in the text
                measurement_match = re.search(r'\b(\d+\s+\d+/\d+)\b', full_text)
                if measurement_match:
                    measurement = measurement_match.group(1)
                    first_num = int(re.match(r'^(\d+)', measurement).group(1))
                    if 2 <= first_num <= 100:
                        print(f"      Found measurement in text: '{measurement}'")
                        return measurement

    print(f"      No clear measurement found, keeping original: '{text}'")
    return text


# Show what needs to change in measurement_based_detector.py
def show_required_changes():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("="*80)
    print("REQUIRED CHANGES TO measurement_based_detector.py")
    print("="*80)
    print("""
1. Modify the call to verify_measurement_with_zoom (around line 490) to pass group bounds:

CHANGE FROM:
    verified_text = verify_measurement_with_zoom(image_path, center_x, center_y, group_text, api_key)

CHANGE TO:
    # Pass the actual group bounds so verification can extract text from that specific area
    group_bounds = {
        'left': min_x,
        'right': max_x,
        'top': min_y,
        'bottom': max_y
    }
    verified_text = verify_measurement_with_zoom(image_path, center_x, center_y, group_text, api_key, group_bounds)

2. Update verify_measurement_with_zoom function signature (around line 92) to accept group_bounds:

CHANGE FROM:
    def verify_measurement_with_zoom(image_path, x, y, text, api_key):

CHANGE TO:
    def verify_measurement_with_zoom(image_path, x, y, text, api_key, group_bounds=None):

3. In verify_measurement_with_zoom, use group_bounds to crop to the exact group area:

    - If group_bounds is provided, crop to that specific region (not just around center)
    - Run OCR on that cropped region
    - The full_text from OCR will be what's IN the group bounds
    - Extract and return the measurement from that text

This way, when we verify a group like ['20', '[garbled]', '17', '116']:
- We crop to the exact bounds of all four items
- OCR on that region returns "20 1/16"
- We return the complete measurement, not just part of it
""")

if __name__ == "__main__":
    show_required_changes()