#!/usr/bin/env python3
"""
Measurement pairing logic.
Pairs width and height measurements to form cabinet openings.
"""

import numpy as np
import cv2


def add_fraction_to_measurement(measurement, fraction_to_add):
    """
    Add a fraction to a measurement string (e.g., "23 15/16" + "5/8" = "24 9/16")
    """
    import re
    from fractions import Fraction

    if not fraction_to_add:
        return measurement

    # Parse the measurement (e.g., "23 15/16" or "23" or "15/16")
    match = re.match(r'(\d+)?\s*(\d+/\d+)?', measurement.strip())
    if not match:
        return measurement

    whole_part = match.group(1)
    fraction_part = match.group(2)

    # Convert to Fraction
    total = Fraction(0)
    if whole_part:
        total += int(whole_part)
    if fraction_part:
        total += Fraction(fraction_part)

    # Parse the overlay fraction (e.g., "5/8 OL" -> "5/8")
    overlay_match = re.match(r'(\d+/\d+)', fraction_to_add.strip())
    if not overlay_match:
        return measurement

    overlay_frac = Fraction(overlay_match.group(1))

    # Add them together
    result = total + overlay_frac

    # Convert back to mixed number format
    whole = result.numerator // result.denominator
    remainder = result.numerator % result.denominator

    if remainder == 0:
        return str(whole)
    elif whole == 0:
        return f"{remainder}/{result.denominator}"
    else:
        return f"{whole} {remainder}/{result.denominator}"


def find_clear_position_for_marker(intersection_x, intersection_y, width_pos, height_pos, measurements_list, opening, image, existing_markers=None, overlay_info=None, marker_radius=35, exclude_items=None):
    """
    Find the best available clear position for the opening number marker.
    Analyzes available space and picks the closest spot to the intersection that fits.
    """
    # Collect all occupied regions
    occupied_regions = []

    # Add regions for all measurement text
    for meas in measurements_list:
        if 'bounds' in meas and meas['bounds']:
            bounds = meas['bounds']
            occupied_regions.append({
                'left': bounds['left'],  # No padding - exact bounds
                'right': bounds['right'],
                'top': bounds['top'],
                'bottom': bounds['bottom']
            })

    # Add regions for room name text (exclude_items)
    if exclude_items:
        for item in exclude_items:
            if 'bounds' in item and item['bounds']:
                bounds = item['bounds']
                occupied_regions.append({
                    'left': bounds['left'],
                    'right': bounds['right'],
                    'top': bounds['top'],
                    'bottom': bounds['bottom']
                })

    # Add regions for existing opening markers
    if existing_markers:
        for marker in existing_markers:
            marker_x, marker_y = marker['position']
            # Add circular marker region (use stored radius if available, otherwise default)
            existing_radius = marker.get('radius', 35)
            occupied_regions.append({
                'left': marker_x - existing_radius,
                'right': marker_x + existing_radius,
                'top': marker_y - existing_radius,
                'bottom': marker_y + existing_radius
            })

            # NOTE: Dimension text below marker is NOT included in collision detection
            # This allows markers to be placed closer together without being pushed away
            # by the dimension text of other markers
            # dim_text = f"{marker['opening']['width']} W x {marker['opening']['height']} H"
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.7, 2)
            # occupied_regions.append({
            #     'left': marker_x - text_width//2 - 1,  # Minimal 1px padding
            #     'right': marker_x + text_width//2 + 1,
            #     'top': marker_y + 48,  # Text starts at +50, minimal padding
            #     'bottom': marker_y + 50 + text_height
            # })

    # Calculate dimension text size for the marker
    dim_text = f"{opening['width']} W x {opening['height']} H"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.7, 2)

    # Start from the center of the opening (intersection point)
    # Then try small offsets within the opening area
    # Only move outside if necessary
    test_positions = [
        (intersection_x, intersection_y),  # Center of opening (preferred)
    ]

    # Scan in expanding circles around the intersection point
    # Start with small movements, prioritizing upward movement
    max_radius = 250  # Increased from 120 to allow more spread

    # First, try moving up pixel by pixel (prefer upward movement)
    for offset in range(1, 100):  # Increased from 50 to 100
        test_positions.append((intersection_x, intersection_y - offset))

    # Then try other directions pixel by pixel
    for offset in range(1, 100):  # Increased from 50 to 100
        test_positions.extend([
            (intersection_x + offset, intersection_y),  # Right
            (intersection_x - offset, intersection_y),  # Left
            (intersection_x, intersection_y + offset),  # Down (last priority)
        ])

    # Now scan in expanding squares for remaining positions
    for radius in range(5, max_radius, 5):
        # Generate points in a square perimeter
        # Top edge (left to right)
        for x in range(intersection_x - radius, intersection_x + radius + 1, 5):
            test_positions.append((x, intersection_y - radius))
        # Right edge (top to bottom)
        for y in range(intersection_y - radius + 5, intersection_y + radius, 5):
            test_positions.append((intersection_x + radius, y))
        # Bottom edge (right to left)
        for x in range(intersection_x + radius, intersection_x - radius - 1, -5):
            test_positions.append((x, intersection_y + radius))
        # Left edge (bottom to top)
        for y in range(intersection_y + radius - 5, intersection_y - radius, -5):
            test_positions.append((intersection_x - radius, y))

    # Find the best clear position closest to intersection
    best_position = None
    best_distance = float('inf')

    for test_x, test_y in test_positions:
        # Check if position is within image bounds
        if test_x - marker_radius < 0 or test_x + marker_radius >= image.shape[1]:
            continue
        if test_y - marker_radius < 0 or test_y + marker_radius + 50 + text_height >= image.shape[0]:
            continue

        # Calculate marker bounds (including text below)
        marker_left = test_x - max(marker_radius, text_width // 2)
        marker_right = test_x + max(marker_radius, text_width // 2)
        marker_top = test_y - marker_radius
        marker_bottom = test_y + marker_radius + 50 + text_height  # Account for circle + text below

        # Calculate overlap with occupied regions
        total_overlap = 0
        for region in occupied_regions:
            # Check for overlap
            if (marker_left < region['right'] and marker_right > region['left'] and
                marker_top < region['bottom'] and marker_bottom > region['top']):
                # Calculate overlap area
                overlap_x = min(marker_right, region['right']) - max(marker_left, region['left'])
                overlap_y = min(marker_bottom, region['bottom']) - max(marker_top, region['top'])
                total_overlap += overlap_x * overlap_y

        # If no overlap, check distance to intersection
        if total_overlap == 0:
            distance = np.sqrt((test_x - intersection_x)**2 + (test_y - intersection_y)**2)
            if distance < best_distance:
                best_distance = distance
                best_position = (test_x, test_y)
            # Use first clear position at this distance
            if distance == 0:  # At intersection
                break

    # If no clear position found, default to offset
    if best_position is None:
        best_position = (intersection_x + 80, intersection_y)

    return int(best_position[0]), int(best_position[1])


def pair_measurements_by_proximity(classified_measurements, all_measurements):
    """
    Pair width and height measurements based on proximity to form cabinet openings
    Returns list of paired openings
    """
    print("\n=== PAIRING MEASUREMENTS INTO OPENINGS ===")

    # Initialize variables
    unpaired_heights = []
    used_heights = set()

    # Extract measurements by type with their positions
    widths = []
    heights = []

    for i, meas in enumerate(all_measurements):
        if i < len(classified_measurements):
            category = classified_measurements[i]
            if category == 'width':
                widths.append({
                    'text': meas['text'],
                    'x': meas['position'][0],
                    'y': meas['position'][1],
                    'bounds': meas.get('bounds', None),
                    'h_line_angle': meas.get('h_line_angle', 0),
                    'is_finished_size': meas.get('is_finished_size', False)
                })
            elif category == 'height':
                height_data = {
                    'text': meas['text'],
                    'x': meas['position'][0],
                    'y': meas['position'][1],
                    'bounds': meas.get('bounds', None),
                    'v_line_angle': meas.get('v_line_angle', 90),
                    'is_finished_size': meas.get('is_finished_size', False)
                }
                if 'notation' in meas:
                    height_data['notation'] = meas['notation']
                heights.append(height_data)

    if not widths or not heights:
        print("  Cannot pair - need both widths and heights")
        return [], []

    # Sort by X position (left to right) first, then Y position (top to bottom)
    widths.sort(key=lambda w: (w['x'], w['y']))
    heights.sort(key=lambda h: (h['x'], h['y']))

    print(f"\n  Widths to pair ({len(widths)}):")
    for w in widths:
        print(f"    {w['text']} at ({w['x']:.0f}, {w['y']:.0f})")

    print(f"\n  Heights to pair ({len(heights)}):")
    for h in heights:
        print(f"    {h['text']} at ({h['x']:.0f}, {h['y']:.0f})")

    # Check if dimensions are stacked (similar X positions)
    STACKING_TOLERANCE = 100  # pixels

    # Check width stacking
    width_x_positions = [w['x'] for w in widths]
    width_x_range = max(width_x_positions) - min(width_x_positions) if widths else 0
    widths_stacked = width_x_range < STACKING_TOLERANCE and len(widths) > 1

    # Check height stacking
    height_x_positions = [h['x'] for h in heights]
    height_x_range = max(height_x_positions) - min(height_x_positions) if heights else 0
    heights_stacked = height_x_range < STACKING_TOLERANCE and len(heights) > 1

    print(f"\n  Arrangement Analysis:")
    print(f"    Widths X-range: {width_x_range:.0f}px - {'STACKED' if widths_stacked else 'SIDE-BY-SIDE'}")
    print(f"    Heights X-range: {height_x_range:.0f}px - {'STACKED' if heights_stacked else 'SIDE-BY-SIDE'}")

    openings = []
    used_heights = set()

    print(f"\n  Pairing Strategy: {'Stacked widths' if widths_stacked else 'Side-by-side widths'}")

    if widths_stacked:
        # Each height finds its closest width
        print("  → Each height will find its closest width")

        for height in heights:
            best_width = None
            best_distance = float('inf')

            for width in widths:
                # Calculate Euclidean distance
                distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)

                if distance < best_distance:
                    best_distance = distance
                    best_width = width

            if best_width:
                print(f"\n  Height '{height['text']}' pairs with width '{best_width['text']}'")
                print(f"    Distance: {best_distance:.0f}px")
                opening_data = {
                    'width': best_width['text'],
                    'height': height['text'],
                    'width_pos': (best_width['x'], best_width['y']),
                    'height_pos': (height['x'], height['y']),
                    'distance': best_distance,
                    'width_angle': best_width.get('h_line_angle', 0),
                    'height_angle': height.get('v_line_angle', 90),
                    'width_is_finished': best_width.get('is_finished_size', False),
                    'height_is_finished': height.get('is_finished_size', False)
                }
                if 'notation' in height:
                    opening_data['notation'] = height['notation']
                openings.append(opening_data)
                height_id = (height['x'], height['y'])
                used_heights.add(height_id)

    else:
        # Side-by-side: each width finds closest height above it
        print("  → Each width will find closest height above it")

        for width in widths:
            best_height = None
            best_distance = float('inf')

            # First try: Look for height above this width
            for height in heights:
                # Only consider heights above this width (at least 20px above)
                if height['y'] >= width['y'] - 20:
                    continue

                # Calculate weighted distance (X distance weighted more)
                x_dist = abs(height['x'] - width['x'])
                y_dist = abs(height['y'] - width['y'])
                distance = np.sqrt(x_dist**2 + y_dist**2)
                weighted_distance = distance + (x_dist * 0.5)

                if weighted_distance < best_distance:
                    best_distance = weighted_distance
                    best_height = height

            # Second try: If no height found above, extend H-lines and check if height is above extended line
            if not best_height:
                print(f"\n  Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): No height found above, extending H-lines...")

                # Get the H-lines from this width measurement
                width_measurement = None
                for i, meas in enumerate(all_measurements):
                    if abs(meas['position'][0] - width['x']) < 5 and abs(meas['position'][1] - width['y']) < 5:
                        width_measurement = meas
                        break

                if width_measurement and 'h_lines' in width_measurement:
                    h_lines = width_measurement['h_lines']
                    print(f"    Found {len(h_lines)} H-lines to extend")

                    # Filter to only use H-lines on the side TOWARD the height
                    for height in heights:
                        if height['x'] < width['x']:
                            relevant_h_lines = [hl for hl in h_lines if (hl['coords'][0] + hl['coords'][2])/2 < width['x']]
                        else:
                            relevant_h_lines = [hl for hl in h_lines if (hl['coords'][0] + hl['coords'][2])/2 > width['x']]

                        if not relevant_h_lines:
                            continue

                        print(f"    Using {len(relevant_h_lines)} H-lines on {'left' if height['x'] < width['x'] else 'right'} side toward height")

                        # For each relevant H-line, calculate where it would be at the height's X position
                        for h_line in relevant_h_lines:
                            lx1, ly1, lx2, ly2 = h_line['coords']
                            angle_deg = h_line.get('angle', np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))

                            if lx2 - lx1 != 0:
                                slope = (ly2 - ly1) / (lx2 - lx1)
                            else:
                                slope = 0

                            if lx1 > lx2:
                                ref_x, ref_y = lx1, ly1
                            else:
                                ref_x, ref_y = lx2, ly2

                            delta_x = height['x'] - ref_x
                            delta_y = slope * delta_x
                            line_y_at_height = ref_y + delta_y

                            tolerance = 200
                            print(f"      H-line angle={angle_deg:.1f}° slope={slope:.3f}: extends to Y={line_y_at_height:.0f}, height at Y={height['y']:.0f}")
                            if height['y'] < line_y_at_height and (line_y_at_height - height['y']) < tolerance:
                                x_dist = abs(height['x'] - width['x'])
                                y_dist = abs(height['y'] - width['y'])
                                distance = np.sqrt(x_dist**2 + y_dist**2)
                                weighted_distance = distance + (x_dist * 0.5)

                                print(f"    Height {line_y_at_height - height['y']:.0f}px above extended line - VALID PAIR!")

                                if weighted_distance < best_distance:
                                    best_distance = weighted_distance
                                    best_height = height
                                    break

                        if best_height:
                            break

            # Third fallback: If still no height, look for height below
            if not best_height:
                print(f"\n  Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): No height via extended lines, trying below...")
                for height in heights:
                    if height['y'] <= width['y'] + 20:
                        continue

                    x_dist = abs(height['x'] - width['x'])
                    y_dist = abs(height['y'] - width['y'])
                    distance = np.sqrt(x_dist**2 + y_dist**2)
                    weighted_distance = distance + (x_dist * 0.5)

                    if weighted_distance < best_distance:
                        best_distance = weighted_distance
                        best_height = height

            if best_height:
                print(f"\n  Width '{width['text']}' pairs with height '{best_height['text']}'")
                print(f"    Distance: {best_distance:.0f}px")
                opening_data = {
                    'width': width['text'],
                    'height': best_height['text'],
                    'width_pos': (width['x'], width['y']),
                    'height_pos': (best_height['x'], best_height['y']),
                    'distance': best_distance,
                    'width_angle': width.get('h_line_angle', 0),
                    'height_angle': best_height.get('v_line_angle', 90),
                    'width_is_finished': width.get('is_finished_size', False),
                    'height_is_finished': best_height.get('is_finished_size', False)
                }
                if 'notation' in best_height:
                    opening_data['notation'] = best_height['notation']
                openings.append(opening_data)
                height_id = (best_height['x'], best_height['y'])
                used_heights.add(height_id)

        # Second pass for unpaired heights (this allows one width to pair with multiple heights!)
        unpaired_heights = [h for h in heights if (h['x'], h['y']) not in used_heights]
        if unpaired_heights:
            print(f"\n  Second pass for {len(unpaired_heights)} unpaired heights:")

            for height in unpaired_heights:
                best_width = None
                best_distance = float('inf')

                # Estimate text width for the height
                if 'bounds' in height and height['bounds']:
                    height_text_width = height['bounds']['right'] - height['bounds']['left']
                else:
                    height_text_width = len(height['text']) * 15

                height_x = height['x']
                x_tolerance = height_text_width / 2 + 50

                # First try: Find width directly below within X-range
                for width in widths:
                    if width['y'] <= height['y']:
                        continue

                    x_diff = abs(width['x'] - height_x)
                    if x_diff <= x_tolerance:
                        distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)
                        print(f"        Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): x_diff={x_diff:.0f}, distance={distance:.0f} - WITHIN X-RANGE")
                        if distance < best_distance:
                            best_distance = distance
                            best_width = width
                            print(f"          -> New best width!")
                    else:
                        print(f"        Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): x_diff={x_diff:.0f} - OUTSIDE X-RANGE")

                # Second try: If no width directly below, try width above within X-range
                if not best_width:
                    print(f"      Second try: Looking for width above within X-range...")
                    for width in widths:
                        if width['y'] >= height['y']:
                            continue

                        x_diff = abs(width['x'] - height_x)
                        print(f"        Width '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}): x_diff={x_diff:.0f}, x_tolerance={x_tolerance:.0f}")
                        if x_diff <= x_tolerance + 1:
                            x_distance = (height['x'] - width['x']) * 3
                            y_distance = height['y'] - width['y']
                            distance = np.sqrt(x_distance**2 + y_distance**2)
                            print(f"          Distance={distance:.0f}, best={best_distance:.0f}")
                            if distance < best_distance:
                                best_distance = distance
                                best_width = width
                                print(f"          -> New best width found!")

                # Third try: If no width in X-range, try ANY width below
                if not best_width:
                    for width in widths:
                        if width['y'] <= height['y']:
                            continue

                        distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)
                        if distance < best_distance:
                            best_distance = distance
                            best_width = width

                # Fourth try: If still no width below, find ANY width above
                if not best_width:
                    for width in widths:
                        if width['y'] >= height['y']:
                            continue

                        distance = np.sqrt((height['x'] - width['x'])**2 + (height['y'] - width['y'])**2)
                        if distance < best_distance:
                            best_distance = distance
                            best_width = width

                if best_width:
                    search_type = "below" if best_width['y'] > height['y'] else "above"
                    print(f"    Height '{height['text']}' pairs with width '{best_width['text']}' {search_type} (distance: {best_distance:.0f}px)")
                    opening_data = {
                        'width': best_width['text'],
                        'height': height['text'],
                        'width_pos': (best_width['x'], best_width['y']),
                        'height_pos': (height['x'], height['y']),
                        'distance': best_distance,
                        'width_angle': best_width.get('h_line_angle', 0),
                        'height_angle': height.get('v_line_angle', 90),
                        'width_is_finished': best_width.get('is_finished_size', False),
                        'height_is_finished': height.get('is_finished_size', False)
                    }
                    if 'notation' in height:
                        opening_data['notation'] = height['notation']
                    openings.append(opening_data)

    print(f"\n  Total openings paired: {len(openings)}")

    # Sort openings by X position (left to right) then Y position (top to bottom)
    if len(openings) > 0:
        openings.sort(key=lambda o: (
            (o['width_pos'][0] + o['height_pos'][0]) / 2,
            (o['width_pos'][1] + o['height_pos'][1]) / 2
        ))
        print("  Sorted by X position (left to right), then Y position (top to bottom)")

    # Collect unpaired heights info
    unpaired_heights_info = []
    for height in heights:
        if (height['x'], height['y']) not in used_heights:
            if 'bounds' in height and height['bounds']:
                height_text_width = height['bounds']['right'] - height['bounds']['left']
            else:
                height_text_width = len(height['text']) * 15

            x_tolerance = height_text_width / 2 + 50
            unpaired_heights_info.append({
                'x': height['x'],
                'y': height['y'],
                'text': height['text'],
                'x_tolerance': x_tolerance
            })

    return openings, unpaired_heights_info
