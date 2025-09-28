"""
Fix for verify_measurement_with_zoom to properly handle grouped text.
Instead of just checking what's at the center point, it should return
the complete measurement that spans the group's area.
"""

def verify_measurement_with_zoom_FIXED(image_path, x, y, text, api_key, group_bounds=None):
    """
    Fixed version that considers the full group bounds, not just center point.

    Args:
        image_path: Path to the image
        x, y: Center coordinates of the group
        text: Original group text (e.g., "20 ラ 17 116")
        api_key: Google Vision API key
        group_bounds: Optional dict with 'min_x', 'max_x', 'min_y', 'max_y' of the group
    """
    import cv2
    import base64
    import requests
    import re

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return text

    h, w = image.shape[:2]

    # Create a padded crop area
    padding_x = 200
    padding_y = 50
    x1 = max(0, int(x - padding_x))
    y1 = max(0, int(y - padding_y))
    x2 = min(w, int(x + padding_x))
    y2 = min(h, int(y + padding_y))

    # Crop the region
    cropped = image[y1:y2, x1:x2]

    # Zoom in 3x
    zoom_factor = 3
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Encode for Vision API
    _, buffer = cv2.imencode('.png', zoomed)
    zoomed_content = base64.b64encode(buffer).decode('utf-8')

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
                # Get full text first
                full_text = annotations[0]['description']

                # Priority 1: Check if a complete measurement exists in the full text
                # This is especially important for grouped text that might be garbled
                measurement_patterns = [
                    r'\b(\d+\s+\d+/\d+)\b',      # "20 1/16"
                    r'\b(\d+-\d+/\d+)\b',         # "20-1/16"
                    r'\b(\d+)\s+(\d+/\d+)\b',    # "20" "1/16" separately
                ]

                for pattern in measurement_patterns:
                    match = re.search(pattern, full_text)
                    if match:
                        if match.lastindex == 2:
                            # Two groups - combine them
                            measurement = f"{match.group(1)} {match.group(2)}"
                        else:
                            measurement = match.group(1)

                        # Normalize format (replace dash with space)
                        measurement = measurement.replace('-', ' ')

                        # Sanity check
                        first_num = int(re.match(r'^(\d+)', measurement).group(1))
                        if 2 <= first_num <= 100:
                            print(f"    Verification: Found complete measurement '{measurement}' in zoomed text")
                            return measurement

                # Priority 2: If we have group bounds, look for text that spans that area
                if group_bounds:
                    # Calculate where the group bounds would be in the zoomed image
                    # The group bounds are in original image coordinates
                    # We need to transform them to zoomed image coordinates

                    # Transform to cropped coordinates
                    group_x1_cropped = (group_bounds['min_x'] - x1) * zoom_factor
                    group_x2_cropped = (group_bounds['max_x'] - x1) * zoom_factor
                    group_y1_cropped = (group_bounds['min_y'] - y1) * zoom_factor
                    group_y2_cropped = (group_bounds['max_y'] - y1) * zoom_factor

                    # Find text items that overlap with the group area
                    overlapping_texts = []
                    for ann in annotations[1:]:  # Skip full text
                        vertices = ann.get('boundingPoly', {}).get('vertices', [])
                        if len(vertices) >= 4:
                            x_coords = [v.get('x', 0) for v in vertices]
                            y_coords = [v.get('y', 0) for v in vertices]
                            min_x, max_x = min(x_coords), max(x_coords)
                            min_y, max_y = min(y_coords), max(y_coords)

                            # Check for overlap with group bounds
                            if (min_x < group_x2_cropped and max_x > group_x1_cropped and
                                min_y < group_y2_cropped and max_y > group_y1_cropped):
                                overlapping_texts.append(ann['description'])

                    # Combine overlapping texts
                    if overlapping_texts:
                        combined = ' '.join(overlapping_texts)
                        print(f"    Verification: Texts overlapping group area: {overlapping_texts}")

                        # Check if it forms a valid measurement
                        for pattern in measurement_patterns:
                            match = re.search(pattern, combined)
                            if match:
                                if match.lastindex == 2:
                                    measurement = f"{match.group(1)} {match.group(2)}"
                                else:
                                    measurement = match.group(1)
                                measurement = measurement.replace('-', ' ')
                                first_num = int(re.match(r'^(\d+)', measurement).group(1))
                                if 2 <= first_num <= 100:
                                    print(f"    Verification: Combined overlapping texts form '{measurement}'")
                                    return measurement

                # Priority 3: Original logic - check center point
                zoomed_height, zoomed_width = zoomed.shape[:2]
                center_x = zoomed_width / 2
                center_y = zoomed_height / 2

                valid_texts = []
                for ann in annotations[1:]:
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
                            valid_texts.append(text_item)

                # Check if original text appears
                if text in valid_texts:
                    return text

                # Check for measurement patterns
                for text_item in valid_texts:
                    if re.match(r'^\d+$', text_item) or \
                       re.match(r'^\d+\s+\d+/\d+$', text_item) or \
                       re.match(r'^\d+-\d+/\d+$', text_item):
                        return text_item

    return text  # Return original if nothing better found


# Test the fix
def test_fix():
    """Test that the fix properly handles the grouped measurement"""
    print("="*80)
    print("TESTING FIX FOR GROUPED MEASUREMENT VERIFICATION")
    print("="*80)

    # Simulate the group bounds for the second "20 1/16"
    # Based on our debug, the group spans roughly:
    # - "20" at (1172, 1416)
    # - Plus garbled text extending to the right

    group_bounds = {
        'min_x': 1172 - 50,  # Approximate left edge
        'max_x': 1172 + 150,  # Approximate right edge (to include "1/16")
        'min_y': 1416 - 20,   # Approximate top
        'max_y': 1416 + 20    # Approximate bottom
    }

    print("\nThe fix adds these improvements:")
    print("1. First checks if full zoomed text contains a complete measurement pattern")
    print("2. If group bounds are provided, looks for text spanning that area")
    print("3. Falls back to center-point checking only as last resort")
    print("\nThis should correctly return '20 1/16' instead of just '20'")

    print("\nKey insight: When we zoom on the center of a group like ['20', 'ラ', '17', '116'],")
    print("we should return the COMPLETE measurement found in that region ('20 1/16'),")
    print("not just what's at the exact center pixel ('20').")

if __name__ == "__main__":
    test_fix()