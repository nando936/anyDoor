"""
Debug script to show what text regions the marker is trying to avoid
"""
import json
import cv2
import numpy as np
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

def visualize_text_avoidance(image_path, opening_number=1):
    """
    Visualize what text regions the marker for a specific opening is trying to avoid
    """
    # Get base name and directory
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(os.path.abspath(image_path))

    # Load data
    with open(os.path.join(image_dir, f'{base_name}_measurements_data.json'), 'r') as f:
        measurement_data = json.load(f)

    with open(os.path.join(image_dir, f'{base_name}_openings_data.json'), 'r') as f:
        pairing_data = json.load(f)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        return

    debug_image = image.copy()

    # Find the specific opening
    target_opening = None
    for opening in pairing_data['openings']:
        if opening['number'] == opening_number:
            target_opening = opening
            break

    if not target_opening:
        print(f"[ERROR] Opening #{opening_number} not found")
        return

    print("=" * 80)
    print(f"DEBUG: Text Avoidance for Opening #{opening_number}")
    print("=" * 80)
    print(f"Opening: {target_opening['specification']}")
    print(f"Width: {target_opening['width']}")
    print(f"Height: {target_opening['height']}")
    print()

    # Get positions
    width_pos = target_opening.get('width_pos')
    height_pos = target_opening.get('height_pos')

    if not width_pos or not height_pos:
        print("[ERROR] Positions not found in opening data")
        return

    # Calculate intersection point
    intersection_x = int(width_pos[0])
    intersection_y = int(height_pos[1])

    print(f"Intersection point: ({intersection_x}, {intersection_y})")
    print()

    # Draw intersection point
    cv2.circle(debug_image, (intersection_x, intersection_y), 5, (0, 255, 0), -1)
    cv2.putText(debug_image, "INTERSECTION", (intersection_x + 10, intersection_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Collect all text regions to avoid
    avoid_regions = []
    all_measurements = measurement_data['measurements']

    print("Text regions to avoid:")
    print("-" * 60)

    # Helper to add measurement bounds
    def add_measurement_bounds(meas_text, pos, color=(255, 0, 0)):
        """Add bounds for a measurement and draw them"""
        if not pos:
            return

        # Find the measurement to get its bounds
        for m in all_measurements:
            if m['text'] == meas_text:
                # Use full_bounds if available (complete bounding box from OCR)
                if 'full_bounds' in m and m['full_bounds']:
                    left = m['full_bounds']['left']
                    right = m['full_bounds']['right']
                    top = m['full_bounds']['top']
                    bottom = m['full_bounds']['bottom']

                    # Draw the avoid region
                    cv2.rectangle(debug_image,
                                (int(left), int(top)),
                                (int(right), int(bottom)),
                                color, 2)

                    # Label it
                    cv2.putText(debug_image, meas_text,
                              (int(left), int(top) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    avoid_regions.append({
                        'text': meas_text,
                        'left': left,
                        'right': right,
                        'top': top,
                        'bottom': bottom,
                        'center': pos
                    })

                    print(f"  '{meas_text}' (full bounds): L={left}, R={right}, T={top}, B={bottom}")
                    return
                # Fall back to old bounds if full_bounds not available
                elif 'bounds' in m:
                    left_bound, right_bound = m['bounds']
                    top = pos[1] - 20
                    bottom = pos[1] + 20

                    # Draw the avoid region
                    cv2.rectangle(debug_image,
                                (int(left_bound), int(top)),
                                (int(right_bound), int(bottom)),
                                color, 2)

                    # Label it
                    cv2.putText(debug_image, meas_text,
                              (int(left_bound), int(top) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    avoid_regions.append({
                        'text': meas_text,
                        'left': left_bound,
                        'right': right_bound,
                        'top': top,
                        'bottom': bottom,
                        'center': pos
                    })

                    print(f"  '{meas_text}' (estimated): L={left_bound}, R={right_bound}, T={top}, B={bottom}")
                    return

        # Fallback if bounds not found
        left = pos[0] - 70
        right = pos[0] + 70
        top = pos[1] - 20
        bottom = pos[1] + 20

        cv2.rectangle(debug_image,
                    (int(left), int(top)),
                    (int(right), int(bottom)),
                    color, 2)

        cv2.putText(debug_image, meas_text + " (no bounds)",
                  (int(left), int(top) - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        avoid_regions.append({
            'text': meas_text + " (estimated)",
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
            'center': pos
        })

        print(f"  '{meas_text}' (estimated): L={left}, R={right}, T={top}, B={bottom}")

    # Add the width and height of current opening (in blue)
    print("\nCurrent opening measurements (BLUE):")
    add_measurement_bounds(target_opening['width'], width_pos, color=(255, 0, 0))
    add_measurement_bounds(target_opening['height'], height_pos, color=(255, 0, 0))

    # Add all other measurements (in red)
    print("\nOther measurements (RED):")
    for m in all_measurements:
        if 'text' in m and 'position' in m:
            # Skip the width and height of current opening
            if m['text'] != target_opening['width'] and m['text'] != target_opening['height']:
                add_measurement_bounds(m['text'], m['position'], color=(0, 0, 255))

    print()
    print("Testing marker positions:")
    print("-" * 60)

    # Test different marker positions
    offset_distance = 80
    test_positions = [
        ("Left", intersection_x - offset_distance, intersection_y),
        ("Right", intersection_x + offset_distance, intersection_y),
        ("Upper-left", intersection_x - offset_distance, intersection_y - offset_distance),
        ("Upper-right", intersection_x + offset_distance, intersection_y - offset_distance),
        ("Lower-left", intersection_x - offset_distance, intersection_y + offset_distance),
        ("Lower-right", intersection_x + offset_distance, intersection_y + offset_distance),
        ("Above", intersection_x, intersection_y - offset_distance),
        ("Below", intersection_x, intersection_y + offset_distance),
    ]

    radius = 45

    for name, test_x, test_y in test_positions:
        # Check for overlaps
        marker_left = test_x - radius
        marker_right = test_x + radius
        marker_top = test_y - radius
        marker_bottom = test_y + radius

        overlaps = []
        for region in avoid_regions:
            if (marker_left < region['right'] and marker_right > region['left'] and
                marker_top < region['bottom'] and marker_bottom > region['top']):
                overlaps.append(region['text'])

        # Draw test position
        if overlaps:
            # Red circle for positions with overlap
            cv2.circle(debug_image, (test_x, test_y), radius, (0, 0, 255), 1)
            cv2.putText(debug_image, f"{name} (OVERLAP)", (test_x - 30, test_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            print(f"  {name}: OVERLAPS with: {', '.join(overlaps)}")
        else:
            # Green circle for clear positions
            cv2.circle(debug_image, (test_x, test_y), radius, (0, 255, 0), 2)
            cv2.putText(debug_image, f"{name} (CLEAR)", (test_x - 30, test_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            print(f"  {name}: CLEAR")

    # Save debug image
    output_path = os.path.join(image_dir, f"{base_name}_opening{opening_number}_debug.png")
    cv2.imwrite(output_path, debug_image)
    print(f"\n[OK] Debug image saved: {output_path}")
    print("\nLegend:")
    print("  GREEN dot = Intersection point")
    print("  BLUE boxes = Current opening's measurements to avoid")
    print("  RED boxes = Other measurements to avoid")
    print("  GREEN circles = Clear marker positions")
    print("  RED circles = Positions that would overlap")

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to page 2
        image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/page_2/page_2.png"

    # Get opening number if specified
    opening_number = 1  # Default to opening 1
    if len(sys.argv) > 2:
        opening_number = int(sys.argv[2])

    visualize_text_avoidance(image_path, opening_number)

if __name__ == "__main__":
    main()