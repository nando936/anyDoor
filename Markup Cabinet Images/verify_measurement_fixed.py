def verify_measurement_with_zoom(image_path, x, y, text, api_key, group_bounds=None):
    """Verify a measurement by zooming in on its location

    Args:
        image_path: Path to the image
        x, y: Center coordinates of the measurement/group
        text: Original text to verify
        api_key: Google Vision API key
        group_bounds: Optional dict with 'left', 'right', 'top', 'bottom' of the group
    """
    import cv2
    import re
    import base64
    import requests
    import numpy as np

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return text  # Return original if can't load

    h, w = image.shape[:2]

    # Determine crop area based on whether we have group bounds
    if group_bounds:
        # Use the exact group bounds with some padding
        padding = 30  # Small padding around group
        x1 = max(0, int(group_bounds['left'] - padding))
        y1 = max(0, int(group_bounds['top'] - padding))
        x2 = min(w, int(group_bounds['right'] + padding))
        y2 = min(h, int(group_bounds['bottom'] + padding))
    else:
        # Fallback to center-based crop (original behavior)
        padding_x = 200  # Wider horizontal padding to capture complete measurements
        padding_y = 50   # Vertical padding
        x1 = max(0, int(x - padding_x))
        y1 = max(0, int(y - padding_y))
        x2 = min(w, int(x + padding_x))
        y2 = min(h, int(y + padding_y))

    # Crop the region
    cropped = image[y1:y2, x1:x2]

    # Zoom in 3x
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Helper function to run OCR
    def run_ocr(img, is_hsv=False):
        _, buffer = cv2.imencode('.png', img)
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
                    return annotations
        return None

    # Helper to extract measurement from annotations
    def extract_measurement(annotations, prefix_msg=""):
        full_text = annotations[0]['description']

        if prefix_msg:
            print(f"    {prefix_msg}: '{full_text.replace(chr(10), ' ')}'")

        # Look for measurement patterns
        patterns = [
            r'\b(\d+\s+\d+/\d+)\b',      # "9 1/4" or "20 1/16"
            r'\b(\d+)\s+(\d+/\d+)\b',    # "9" "1/4" on different lines
        ]

        for pattern in patterns:
            match = re.search(pattern, full_text)
            if match:
                if match.lastindex == 2:
                    measurement = f"{match.group(1)} {match.group(2)}"
                else:
                    measurement = match.group(1)

                # Sanity check
                first_num = int(re.match(r'^(\d+)', measurement).group(1))
                if 2 <= first_num <= 100:
                    return measurement

        # Check center-aligned text
        zoomed_height, zoomed_width = zoomed.shape[:2]
        center_x = zoomed_width / 2
        center_y = zoomed_height / 2

        for ann in annotations[1:]:  # Skip full text
            vertices = ann.get('boundingPoly', {}).get('vertices', [])
            if len(vertices) >= 4:
                x_coords = [v.get('x', 0) for v in vertices]
                y_coords = [v.get('y', 0) for v in vertices]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                margin = 10
                if (min_x - margin <= center_x <= max_x + margin and
                    min_y - margin <= center_y <= max_y + margin):
                    text_item = ann['description'].strip()

                    # Check if it's a valid measurement or number
                    if re.match(r'^\d+$', text_item):
                        num = int(text_item)
                        if num > 100:  # Unrealistic
                            return None  # Signal to retry with HSV
                        return text_item
                    elif re.match(r'^\d+\s+\d+/\d+$', text_item):
                        return text_item

        return None

    # First attempt with normal OCR
    annotations = run_ocr(zoomed)
    if annotations:
        result = extract_measurement(annotations, "Normal OCR result")

        # Check if result is unrealistic
        if result and re.match(r'^\d+$', result):
            num = int(result)
            if num > 100:
                print(f"    Result '{result}' is unrealistic, trying HSV preprocessing...")
                result = None  # Force HSV retry

        # If we got a good result, return it
        if result:
            if result != text:
                print(f"    Verification: '{text}' -> '{result}'")
            return result

    # If normal OCR failed or returned unrealistic result, try HSV
    print(f"    Trying HSV preprocessing for better green text detection...")

    # Apply HSV preprocessing
    hsv = cv2.cvtColor(zoomed, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # Run OCR on HSV preprocessed image
    annotations = run_ocr(mask_inv, is_hsv=True)
    if annotations:
        result = extract_measurement(annotations, "HSV preprocessed OCR result")
        if result:
            print(f"    Verification: HSV found '{result}'")
            return result

    # If all else fails, return original
    print(f"    Verification: No clear measurement found, keeping original '{text}'")
    return text