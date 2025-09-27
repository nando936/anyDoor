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

    # Get customer notations if available
    customer_notations = data.get('customer_notations', [])

    return measurements, room_name, customer_notations

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

def associate_notations_with_openings(openings, customer_notations):
    """Associate customer notations with the nearest opening based on proximity"""
    if not customer_notations:
        return openings

    print("\n" + "=" * 60)
    print("ASSOCIATING CUSTOMER NOTATIONS WITH OPENINGS")
    print("-" * 60)

    for notation in customer_notations:
        notation_x = notation['x']
        notation_y = notation['y']
        notation_text = notation['text']

        # Find the closest opening to this notation
        min_distance = float('inf')
        closest_opening_idx = None

        for idx, opening in enumerate(openings):
            # Calculate distance to opening center (average of width and height positions)
            width_x, width_y = opening['width_pos']
            height_x, height_y = opening['height_pos']

            # Opening center is roughly the intersection point
            opening_x = (width_x + height_x) / 2
            opening_y = (width_y + height_y) / 2

            distance = np.sqrt((notation_x - opening_x)**2 + (notation_y - opening_y)**2)

            if distance < min_distance:
                min_distance = distance
                closest_opening_idx = idx

        # Associate notation with closest opening if within reasonable distance
        if closest_opening_idx is not None and min_distance < 200:  # 200 pixels threshold
            if 'notations' not in openings[closest_opening_idx]:
                openings[closest_opening_idx]['notations'] = []
            openings[closest_opening_idx]['notations'].append(notation_text)

            print(f"Notation '{notation_text}' at ({notation_x:.0f}, {notation_y:.0f})")
            print(f"  Associated with opening #{closest_opening_idx + 1}: {openings[closest_opening_idx]['width']} W x {openings[closest_opening_idx]['height']} H")
            print(f"  Distance: {min_distance:.0f} pixels")
            print()

    return openings

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
        print("\n[LOGIC] Widths are side-by-side → Finding closest height above each width")
        print()

        # Track which heights have been used
        used_heights = set()

        # For each width, find the closest height that's above it (smaller Y value)
        for width in widths:
            print(f"Width '{width['text']}' at position ({width['x']:.0f}, {width['y']:.0f})")

            best_height = None
            best_distance = float('inf')

            for height in heights:
                # DON'T skip used heights - they can be shared between widths
                # if id(height) in used_heights:
                #     continue

                # ONLY consider heights ABOVE the width (smaller Y coordinate)
                if height['y'] >= width['y']:
                    continue

                # Calculate distance
                distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)

                # Prefer heights that are reasonably close in X position (same opening area)
                x_distance = abs(height['x'] - width['x'])
                y_distance = abs(height['y'] - width['y'])

                # Weight X distance more heavily for better pairing
                # This helps pair measurements that belong to the same cabinet opening
                weighted_distance = distance + (x_distance * 0.5)

                print(f"  Checking height '{height['text']}' at ({height['x']:.0f}, {height['y']:.0f})")
                print(f"    Distance: {distance:.0f} pixels (weighted: {weighted_distance:.0f})")

                if weighted_distance < best_distance:
                    best_distance = weighted_distance
                    best_height = height

            if best_height:
                print(f"  → PAIRED with '{best_height['text']}' (distance: {best_distance:.0f})")
                openings.append({
                    'width': width['text'],
                    'height': best_height['text'],
                    'width_pos': (width['x'], width['y']),
                    'height_pos': (best_height['x'], best_height['y']),
                    'distance': best_distance
                })
                # Track paired heights (but they can still be reused)
                used_heights.add(id(best_height))
            else:
                print(f"  → No height found above this width")
            print()

        # Second pass: Handle any unpaired heights
        unpaired_heights = [h for h in heights if id(h) not in used_heights]
        if unpaired_heights:
            print("Second pass - handling unpaired heights:")
            print()

            for height in unpaired_heights:
                print(f"Unpaired height '{height['text']}' at position ({height['x']:.0f}, {height['y']:.0f})")

                # Find the closest width to this height
                best_width = None
                best_distance = float('inf')

                for width in widths:
                    distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)
                    print(f"  Checking width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f})")
                    print(f"    Distance: {distance:.0f} pixels")

                    if distance < best_distance:
                        best_distance = distance
                        best_width = width

                if best_width:
                    print(f"  → PAIRED with '{best_width['text']}' (distance: {best_distance:.0f})")
                    openings.append({
                        'width': best_width['text'],
                        'height': height['text'],
                        'width_pos': (best_width['x'], best_width['y']),
                        'height_pos': (height['x'], height['y']),
                        'distance': best_distance
                    })
                else:
                    print(f"  → No width found for this height")
                print()

            # Check for any remaining unpaired dimensions
            print("Final status:")
            print(f"  Total openings found: {len(openings)}")
            if len(openings) < len(heights):
                print(f"  Note: Some dimensions could not be paired")

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

    # Handle both file paths and directory paths
    if os.path.isdir(image_path):
        # If it's a directory, look for page_*.json files
        image_dir = os.path.abspath(image_path)
        # Find the page file in the directory
        import glob
        json_files = glob.glob(os.path.join(image_dir, "page_*_measurements_data.json"))
        if json_files:
            base_name = os.path.basename(json_files[0]).replace("_measurements_data.json", "")
        else:
            print(f"[ERROR] No measurements data file found in {image_dir}")
            return
    else:
        # Get base name and directory for input/output files
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.dirname(os.path.abspath(image_path))

    # Load measurements
    results_file = os.path.join(image_dir, f"{base_name}_measurements_data.json")

    try:
        measurements, room_name, customer_notations = load_measurement_results(results_file)
    except FileNotFoundError:
        print(f"\n[ERROR] Results file not found: {results_file}")
        print("Please run measurement_based_detector.py first")
        return

    print(f"\nLoaded measurements:")
    print(f"  Heights: {[h['text'] for h in measurements['heights']]}")
    print(f"  Widths: {[w['text'] for w in measurements['widths']]}")

    if customer_notations:
        print(f"  Customer notations: {[n['text'] for n in customer_notations]}")

    # Find opening pairs
    openings = pair_by_proximity(measurements)

    # Associate customer notations with openings
    openings = associate_notations_with_openings(openings, customer_notations)

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
        if 'notations' in opening and opening['notations']:
            print(f"  Customer notations: {', '.join(opening['notations'])}")


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
            # Sanitize room name for filename (replace problematic characters)
            filename_safe_room = room_name.replace('/', '-').replace('[', '(').replace(']', ')').replace(' ', '_')
            output_filename = f"{base_name}_{filename_safe_room}_openings_{opening_range}_paired.png"
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
                'width_pos': opening['width_pos'],
                'height_pos': opening['height_pos'],
                'notations': opening.get('notations', []),
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