"""
Proximity-Based Pairing Detector
Pairs widths with their closest heights when dimensions are stacked
"""
import json
import cv2
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

def load_measurement_results(results_file):
    """Load the results from measurement_based_detector.py"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    measurements = {
        'heights': [],
        'widths': []
    }

    for meas in data['measurements']:
        text = meas['text']
        x, y = meas['position']

        if text in data['vertical']:
            measurements['heights'].append({
                'text': text,
                'x': x,
                'y': y
            })
        elif text in data['horizontal']:
            measurements['widths'].append({
                'text': text,
                'x': x,
                'y': y
            })

    # Sort by Y position
    measurements['heights'].sort(key=lambda h: h['y'])
    measurements['widths'].sort(key=lambda w: w['y'])

    # Get room name if available
    room_name = data.get('room_name', '')

    return measurements, room_name

def check_if_stacked(measurements, dimension_type, tolerance=100):
    """
    Check if dimensions are vertically stacked (approximately same X position)
    tolerance: pixels of variance allowed for hand-written measurements
    """
    items = measurements[dimension_type]
    if len(items) < 2:
        return False, 0

    x_positions = [item['x'] for item in items]
    x_variance = np.std(x_positions)
    x_range = max(x_positions) - min(x_positions)

    # Check if all X positions are within tolerance of each other
    # Using range instead of variance for clearer threshold
    is_stacked = x_range < tolerance

    print(f"\n  Checking if {dimension_type} are stacked:")
    print(f"    X positions: {[f'{x:.0f}' for x in x_positions]}")
    print(f"    X range (max-min): {x_range:.0f} pixels")
    print(f"    Tolerance: ±{tolerance} pixels")
    print(f"    Result: {'STACKED' if is_stacked else 'NOT STACKED'}")

    return is_stacked, x_range

def find_closest_width_for_height(height, widths):
    """Find the closest width to a given height"""
    min_distance = float('inf')
    closest_width = None

    for width in widths:
        # Calculate Euclidean distance
        distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)

        if distance < min_distance:
            min_distance = distance
            closest_width = width

    return closest_width, min_distance

def pair_by_proximity(measurements):
    """
    Pair widths with heights based on proximity
    If widths are stacked, each width pairs with its closest height(s)
    """
    heights = measurements['heights']
    widths = measurements['widths']

    # Check if dimensions are stacked (with tolerance for hand-written measurements)
    STACKING_TOLERANCE = 100  # pixels tolerance for hand-written measurements

    print("\n" + "=" * 60)
    print("DIMENSION ARRANGEMENT ANALYSIS")
    print("-" * 60)
    print(f"Using ±{STACKING_TOLERANCE} pixels tolerance for hand-written measurements")

    print(f"\nWidths arrangement:")
    for w in widths:
        print(f"  {w['text']} at position ({w['x']:.0f}, {w['y']:.0f})")
    widths_stacked, width_range = check_if_stacked(measurements, 'widths', STACKING_TOLERANCE)

    print(f"\nHeights arrangement:")
    for h in heights:
        print(f"  {h['text']} at position ({h['x']:.0f}, {h['y']:.0f})")
    heights_stacked, height_range = check_if_stacked(measurements, 'heights', STACKING_TOLERANCE)

    # Pair based on proximity
    openings = []
    paired_heights = set()
    paired_widths = set()

    print("\n" + "=" * 60)
    print("PROXIMITY-BASED PAIRING")
    print("-" * 60)

    if widths_stacked:
        print("\n[LOGIC] Widths are stacked → Each width pairs with closest height(s)")
        print()

        # For each height, find its closest width
        for height in heights:
            closest_width, distance = find_closest_width_for_height(height, widths)

            print(f"Height {height['text']} at Y={height['y']:.0f}")
            print(f"  Closest width: {closest_width['text']} at Y={closest_width['y']:.0f}")
            print(f"  Distance: {distance:.0f}")
            print(f"  → PAIRED: {closest_width['text']} W × {height['text']} H")
            print()

            openings.append({
                'width': closest_width['text'],
                'height': height['text'],
                'width_pos': (closest_width['x'], closest_width['y']),
                'height_pos': (height['x'], height['y']),
                'distance': distance
            })

            paired_heights.add(height['text'])
            paired_widths.add(closest_width['text'])


    else:
        print("\n[LOGIC] Widths are side-by-side → Different pairing logic needed")
        # This would need different logic for side-by-side layouts
        # For now, use simple all-combinations
        for width in widths:
            for height in heights:
                openings.append({
                    'width': width['text'],
                    'height': height['text'],
                    'width_pos': (width['x'], width['y']),
                    'height_pos': (height['x'], height['y']),
                    'distance': np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)
                })

    return openings

def main():
    import os
    print("=" * 80)
    print("PROXIMITY-BASED OPENING DETECTOR")
    print("Pairs dimensions based on closest proximity")
    print("=" * 80)

    # Get image path from command line or default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "page_16.png"

    # Get base name and directory for input/output files
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(os.path.abspath(image_path))

    # Load measurements
    results_file = os.path.join(image_dir, f"{base_name}_measurements_data.json")

    try:
        measurements, room_name = load_measurement_results(results_file)
    except FileNotFoundError:
        print(f"\n[ERROR] Results file not found: {results_file}")
        print("Please run measurement_based_detector.py first")
        return

    print(f"\nLoaded measurements:")
    print(f"  Heights: {[h['text'] for h in measurements['heights']]}")
    print(f"  Widths: {[w['text'] for w in measurements['widths']]}")

    # Find opening pairs
    openings = pair_by_proximity(measurements)

    # If no openings found, report it
    if len(openings) == 0:
        print("\n" + "=" * 60)
        print("NO OPENINGS FOUND")
        print("-" * 60)
        print("No valid width-height pairs could be created from the measurements.")

    # Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS - CABINET OPENINGS")
    print("-" * 60)
    print(f"Total openings found: {len(openings)}")

    for i, opening in enumerate(openings, 1):
        print(f"\nOpening {i}:")
        print(f"  {opening['width']} W × {opening['height']} H")
        print(f"  Width at: ({opening['width_pos'][0]:.0f}, {opening['width_pos'][1]:.0f})")
        print(f"  Height at: ({opening['height_pos'][0]:.0f}, {opening['height_pos'][1]:.0f})")


    # Create annotated image
    image = cv2.imread(image_path)
    if image is not None:
        annotated = image.copy()

        # Draw openings with different colors
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red

        for i, opening in enumerate(openings):
            color = colors[i % len(colors)]

            # Draw connection line
            width_pos = (int(opening['width_pos'][0]), int(opening['width_pos'][1]))
            height_pos = (int(opening['height_pos'][0]), int(opening['height_pos'][1]))

            cv2.line(annotated, width_pos, height_pos, color, 2)

            # Draw circles at positions
            cv2.circle(annotated, width_pos, 10, color, -1)
            cv2.circle(annotated, height_pos, 10, color, -1)

            # Label the opening
            mid_x = (width_pos[0] + height_pos[0]) // 2
            mid_y = (width_pos[1] + height_pos[1]) // 2
            cv2.putText(annotated, f"Opening {i+1}", (mid_x - 40, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Add dimensions
            dim_text = f"{opening['width']}W x {opening['height']}H"
            cv2.putText(annotated, dim_text, (mid_x - 60, mid_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Add title
        cv2.rectangle(annotated, (10, 10), (350, 60), (255, 255, 255), -1)
        cv2.putText(annotated, f"Proximity Pairing: {len(openings)} Openings",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        num_openings = len(openings)
        if num_openings > 0:
            opening_range = f"1-{num_openings}" if num_openings > 1 else "1"
        else:
            opening_range = "0"

        # Include room name in filename if available
        if room_name:
            output_filename = f"{base_name}_{room_name}_openings_{opening_range}_paired.png"
        else:
            output_filename = f"{base_name}_openings_{opening_range}_paired.png"

        output_path = os.path.join(image_dir, output_filename)
        cv2.imwrite(output_path, annotated)
        print(f"\n[OK] Annotated image saved: {output_path}")

    # Save results
    results = {
        'total_openings': len(openings),
        'openings': [
            {
                'number': i,
                'width': opening['width'],
                'height': opening['height'],
                'specification': f"{opening['width']} W x {opening['height']} H"
            }
            for i, opening in enumerate(openings, 1)
        ]
    }

    output_json = os.path.join(image_dir, f"{base_name}_openings_data.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved: {output_json}")

if __name__ == "__main__":
    main()