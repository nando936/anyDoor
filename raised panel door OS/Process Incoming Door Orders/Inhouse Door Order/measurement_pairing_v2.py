#!/usr/bin/env python3
"""
Measurement pairing logic V2.
New improved approach for pairing width and height measurements.
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


def find_clear_position_for_marker(intersection_x, intersection_y, width_pos, height_pos, measurements_list, opening, image, existing_markers=None, overlay_info=None, marker_radius=35):
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

            # Add dimension text below marker
            dim_text = f"{marker['opening']['width']} W x {marker['opening']['height']} H"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.7, 2)
            occupied_regions.append({
                'left': marker_x - text_width//2 - 1,  # Minimal 1px padding
                'right': marker_x + text_width//2 + 1,
                'top': marker_y + 48,  # Text starts at +50, minimal padding
                'bottom': marker_y + 50 + text_height
            })

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


def find_down_arrows_near_heights(image, heights):
    """
    Find down arrows below each height measurement.
    Returns a list of Y-coordinates where down arrows are found.

    Args:
        image: The source image
        heights: List of height measurement dicts with 'x', 'y', and optionally 'height_extent'
    """
    import cv2
    import numpy as np

    if image is None or not heights:
        return []

    down_arrow_positions = []

    # Only look at heights in the bottom half of the page (higher Y = lower on page)
    if heights:
        height_y_positions = [h['y'] for h in heights]
        median_y = sorted(height_y_positions)[len(height_y_positions) // 2]
        bottom_heights = [h for h in heights if h['y'] >= median_y]
        print(f"\n  [DOWN ARROW SCAN] Total heights: {len(heights)}, Median Y: {median_y}, Bottom heights: {len(bottom_heights)}")
    else:
        bottom_heights = []

    # For each bottom height measurement, look for a down arrow below it
    for height in bottom_heights:
        print(f"  [DOWN ARROW SCAN] Checking height '{height['text']}' at ({height['x']:.0f}, {height['y']:.0f})")
        height_x = height['x']
        height_y = height['y']

        # Define search region below the height measurement
        # Start from the measurement text position, not the extent bottom
        search_y_start = int(height_y + 50)

        # Search area: 100px wide centered on height X, from search_y_start down 200px
        roi_x1 = max(0, int(height_x - 50))
        roi_x2 = min(image.shape[1], int(height_x + 50))
        roi_y1 = search_y_start
        roi_y2 = min(image.shape[0], search_y_start + 200)

        print(f"    Search ROI: x=[{roi_x1}, {roi_x2}], y=[{roi_y1}, {roi_y2}]")

        # Extract ROI
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            print(f"    ROI is empty, skipping")
            continue

        # Use same arrow detection method as Phase 3
        # Apply HSV filter to detect only green dimension arrows
        from measurement_config import HSV_CONFIG
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array(HSV_CONFIG['lower_green'])
        upper_green = np.array(HSV_CONFIG['upper_green'])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply morphological operations
        kernel = np.ones((2,2), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Detect edges only on green-filtered image
        edges = cv2.Canny(green_mask, 30, 100)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)

        has_down_arrow = False
        if lines is not None and len(lines) >= 2:
            # Look for converging lines forming down arrow (like v)
            converging_pairs = 0
            for i in range(len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                for j in range(i + 1, len(lines)):
                    x3, y3, x4, y4 = lines[j][0]
                    angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3))

                    # For down arrow, angles should point down from opposite sides
                    angle_diff = abs(angle1 + angle2)
                    if angle_diff > 150:
                        converging_pairs += 1

            has_down_arrow = converging_pairs > 0

        print(f"    Down arrow detected: {has_down_arrow}")

        if has_down_arrow:
            # Found a down arrow - record the Y position (center of search area)
            arrow_y = int(roi_y1 + (roi_y2 - roi_y1) / 2)
            down_arrow_positions.append(arrow_y)
            print(f"    Added down arrow at Y={arrow_y}")

    return down_arrow_positions


def pair_measurements_by_proximity(classified_measurements, all_measurements, image=None):
    """
    V2 - New pairing logic

    Your new implementation goes here.
    Returns list of paired openings and unpaired heights info
    """
    print("\n=== PAIRING MEASUREMENTS V2 ===")

    # Extract measurements by type with their positions
    widths = []
    heights = []

    for i, meas in enumerate(all_measurements):
        if i < len(classified_measurements):
            category = classified_measurements[i]
            if category == 'width':
                width_data = {
                    'text': meas['text'],
                    'x': meas['position'][0],
                    'y': meas['position'][1],
                    'bounds': meas.get('bounds', None),
                    'h_line_angle': meas.get('h_line_angle', 0),
                    'is_finished_size': meas.get('is_finished_size', False)
                }
                # Capture width extent if available
                if 'width_extent' in meas:
                    width_data['width_extent'] = meas['width_extent']
                widths.append(width_data)
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
                # Capture height extent if available
                if 'height_extent' in meas:
                    height_data['height_extent'] = meas['height_extent']
                heights.append(height_data)

    if not widths or not heights:
        print("  Cannot pair - need both widths and heights")
        return [], []

    # Sort by X position (left to right) first, then Y position (top to bottom)
    widths.sort(key=lambda w: (w['x'], w['y']))
    heights.sort(key=lambda h: (h['x'], h['y']))

    # Classify ONLY bottom widths based on down arrow positions
    # All other widths remain unclassified (no position_class field)
    if widths and image is not None:
        import cv2
        import numpy as np

        # Find down arrows below height measurements
        down_arrow_y_positions = find_down_arrows_near_heights(image, heights)

        if down_arrow_y_positions:
            # Get the Y-coordinate of the lowest down arrows (highest Y value)
            max_arrow_y = max(down_arrow_y_positions)

            # Define tolerance range (within 150px of the lowest arrows)
            y_tolerance = 150

            print(f"\n  Down arrow classification:")
            print(f"    Found {len(down_arrow_y_positions)} down arrows")
            print(f"    Lowest arrow Y: {max_arrow_y}")
            print(f"    Bottom width range: {max_arrow_y - y_tolerance} to {max_arrow_y + y_tolerance}")

            # Classify ONLY widths that are within range as 'bottom'
            # All others remain unclassified (no position_class field)
            for width in widths:
                if abs(width['y'] - max_arrow_y) <= y_tolerance:
                    width['position_class'] = 'bottom'
                    print(f"      Classified '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}) as BOTTOM")
        else:
            print("\n  No down arrows found - no bottom widths classified")

    print(f"\n  Widths to pair ({len(widths)}):")
    for w in widths:
        extent_info = ""
        if 'width_extent' in w:
            extent = w['width_extent']
            extent_info = f" | H-lines span: {extent['left']:.0f} to {extent['right']:.0f} (width: {extent['span']:.0f}px)"
        pos_class = ""
        if 'position_class' in w:
            pos_class = f" [{w['position_class']}]"
        print(f"    {w['text']} at ({w['x']:.0f}, {w['y']:.0f}){pos_class}{extent_info}")

    print(f"\n  Heights to pair ({len(heights)}):")
    for h in heights:
        extent_info = ""
        if 'height_extent' in h:
            extent = h['height_extent']
            extent_info = f" | V-lines span: {extent['top']:.0f} to {extent['bottom']:.0f} (height: {extent['span']:.0f}px)"
        print(f"    {h['text']} at ({h['x']:.0f}, {h['y']:.0f}){extent_info}")

    # Analyze drawer configuration ONLY for bottom widths
    print(f"\n=== ANALYZING DRAWER CONFIGURATION (BOTTOM WIDTHS ONLY) ===")
    for i, meas in enumerate(all_measurements):
        if i < len(classified_measurements):
            category = classified_measurements[i]
            if category == 'width' and 'width_extent' in meas:
                # Find corresponding width in widths list
                for width in widths:
                    if width['x'] == meas['position'][0] and width['y'] == meas['position'][1]:
                        # Only analyze if this is a classified bottom width
                        if width.get('position_class') == 'bottom':
                            extent = meas['width_extent']
                            drawer_config, edge_viz_data = analyze_drawer_above_width(width, extent, image, heights)
                            width['drawer_config'] = drawer_config
                            width['edge_viz_data'] = edge_viz_data
                            # Store drawer_config and edge_viz_data back in all_measurements so viz can access it
                            meas['drawer_config'] = drawer_config
                            meas['edge_viz_data'] = edge_viz_data
                            print(f"  {meas['text']} at ({meas['position'][0]:.0f}, {meas['position'][1]:.0f}): {drawer_config}")
                        break

    # Pairing logic
    openings = []
    unpaired_heights_info = []
    paired_heights = set()  # Track which heights have been paired
    width_to_heights = {}  # Track which heights each width paired with

    # For each width, find matching height(s)
    for width in widths:
        if 'width_extent' not in width:
            print(f"  Skipping {width['text']} - no extent available")
            continue

        extent = width['width_extent']
        width_left = extent['left']
        width_right = extent['right']
        drawer_config = width.get('drawer_config', '')

        # Check if this is a bottom width with multiple drawers
        is_bottom_multiple = ('bottom width' in drawer_config and 'multiple drawers' in drawer_config)

        print(f"\n  Pairing {width['text']} at ({width['x']:.0f}, {width['y']:.0f})")
        print(f"    Width extent: {width_left:.0f} to {width_right:.0f}")
        print(f"    Drawer config: {drawer_config}")
        print(f"    Is bottom width with multiple drawers: {is_bottom_multiple}")

        # Calculate scan area X range (before the loop)
        x_padding = 24  # 1/4 inch at 96 DPI (same as visualization horizontal_extension)
        scan_left = width_left - x_padding
        scan_right = width_right + x_padding
        print(f"    Scan area X range: {scan_left:.0f} to {scan_right:.0f} (extent + {x_padding}px padding each side)")

        # Find heights within the scan area X range (extent + horizontal padding)
        candidate_heights = []
        y_padding = 50  # Padding in pixels to account for skew

        for height in heights:
            # Check if ANY part of the height text overlaps with scan area
            # Use bounds if available, otherwise fall back to center position
            if height.get('bounds'):
                # Bounds format: {'left': x, 'right': x, 'top': y, 'bottom': y}
                height_left = height['bounds']['left']
                height_right = height['bounds']['right']
                # Check if height overlaps with scan area horizontally
                overlaps = not (height_right < scan_left or height_left > scan_right)
            else:
                # Fall back to center position check
                height_x = height['x']
                overlaps = scan_left <= height_x <= scan_right

            if overlaps:
                # Only consider heights above the width (with padding for skew)
                if height['y'] < width['y'] + y_padding:
                    candidate_heights.append(height)
                    print(f"      Candidate height: {height['text']} at ({height['x']:.0f}, {height['y']:.0f})")

        print(f"    Found {len(candidate_heights)} candidate heights in X range")

        # Apply pairing rules based on drawer configuration
        if is_bottom_multiple:
            if len(candidate_heights) >= 2:
                print(f"    Multiple heights found ({len(candidate_heights)}) - this width belongs to the drawer bank, will pair with ALL")
            elif len(candidate_heights) < 2:
                # Not enough heights in own extent - discard them and search left width's extent
                print(f"    Insufficient heights ({len(candidate_heights)}) in own extent - discarding and searching left width's extent")

                # Discard heights from own extent
                candidate_heights = []

                # Find the bottom width to the left of current width
                left_width = None
                for w in widths:
                    # Must be a bottom width, to the left (lower X), and closest
                    if (w.get('position_class') == 'bottom' and
                        w['x'] < width['x'] and
                        'width_extent' in w):
                        if left_width is None or w['x'] > left_width['x']:
                            left_width = w

                if left_width:
                    print(f"    Found left width: {left_width['text']} at ({left_width['x']:.0f}, {left_width['y']:.0f})")

                    # Use the same heights that the left width paired with
                    left_width_id = id(left_width)
                    if left_width_id in width_to_heights:
                        candidate_heights = width_to_heights[left_width_id][:]  # Copy the list
                        print(f"    Using {len(candidate_heights)} heights from left width's pairings:")
                        for h in candidate_heights:
                            print(f"      {h['text']} at ({h['x']:.0f}, {h['y']:.0f})")
                    else:
                        print(f"    Left width has not paired yet - cannot use its heights")

                # Re-check if we have enough heights now
                if len(candidate_heights) < 2:
                    print(f"    SKIP: Still insufficient heights ({len(candidate_heights)}) after left search")
                    continue

        # Pairing logic based on width type
        if is_bottom_multiple:
            # Bottom widths with multiple drawers - pair with ALL heights
            if candidate_heights:
                # Store the heights this width paired with
                width_to_heights[id(width)] = candidate_heights[:]

                # Pair with ALL heights in X range
                for height in candidate_heights:
                    opening = {
                        'width': width['text'],
                        'height': height['text'],
                        'width_pos': (width['x'], width['y']),
                        'height_pos': (height['x'], height['y']),
                        'distance': ((width['x'] - height['x'])**2 + (width['y'] - height['y'])**2)**0.5,
                        'width_is_finished': width.get('is_finished_size', False),
                        'height_is_finished': height.get('is_finished_size', False)
                    }

                    if 'notation' in height:
                        opening['notation'] = height['notation']

                    openings.append(opening)
                    paired_heights.add(id(height))
                    print(f"    PAIRED: {width['text']} x {height['text']}")
        else:
            # All other widths (top/unknown) - pair with closest height above and to the left, then right
            print(f"    Looking for single closest height above this width")

            # Find heights above this width
            heights_above = [h for h in heights if h['y'] < width['y']]

            if not heights_above:
                print(f"    No heights found above this width - SKIP")
                continue

            # First: try to find height to the left (X < width X)
            heights_left = [h for h in heights_above if h['x'] < width['x']]
            heights_right = [h for h in heights_above if h['x'] >= width['x']]

            closest_height = None

            if heights_left:
                # Find closest to the left (by distance)
                closest_height = min(heights_left, key=lambda h:
                    ((width['x'] - h['x'])**2 + (width['y'] - h['y'])**2)**0.5)
                print(f"    Found height to the left: {closest_height['text']} at ({closest_height['x']:.0f}, {closest_height['y']:.0f})")
            elif heights_right:
                # No heights to the left, find closest to the right
                closest_height = min(heights_right, key=lambda h:
                    ((width['x'] - h['x'])**2 + (width['y'] - h['y'])**2)**0.5)
                print(f"    No heights to left, found height to the right: {closest_height['text']} at ({closest_height['x']:.0f}, {closest_height['y']:.0f})")

            if closest_height:
                opening = {
                    'width': width['text'],
                    'height': closest_height['text'],
                    'width_pos': (width['x'], width['y']),
                    'height_pos': (closest_height['x'], closest_height['y']),
                    'distance': ((width['x'] - closest_height['x'])**2 + (width['y'] - closest_height['y'])**2)**0.5,
                    'width_is_finished': width.get('is_finished_size', False),
                    'height_is_finished': closest_height.get('is_finished_size', False)
                }

                if 'notation' in closest_height:
                    opening['notation'] = closest_height['notation']

                openings.append(opening)
                paired_heights.add(id(closest_height))
                print(f"    PAIRED: {width['text']} x {closest_height['text']}")
            else:
                print(f"    No suitable height found - SKIP")

    # Pair any unpaired heights with widths above them
    print(f"\n=== PAIRING UNPAIRED HEIGHTS WITH WIDTHS ABOVE ===")
    unpaired_heights = [h for h in heights if id(h) not in paired_heights]

    if unpaired_heights:
        print(f"  Found {len(unpaired_heights)} unpaired heights:")
        for h in unpaired_heights:
            print(f"    {h['text']} at ({h['x']:.0f}, {h['y']:.0f})")

    for height in unpaired_heights:
        print(f"\n  Looking for width above unpaired height: {height['text']} at ({height['x']:.0f}, {height['y']:.0f})")

        # Find all widths above this height (Y < height Y)
        widths_above = [w for w in widths if w['y'] < height['y']]

        if not widths_above:
            print(f"    No widths found above this height - SKIP")
            continue

        # Define tight X range (±50px from height center)
        x_tolerance = 50

        # First: Look for width in tight center range
        widths_center = [w for w in widths_above
                        if abs(w['x'] - height['x']) <= x_tolerance]

        # Second: Look for widths to the left
        widths_left = [w for w in widths_above if w['x'] < height['x'] - x_tolerance]

        # Third: Look for widths to the right
        widths_right = [w for w in widths_above if w['x'] > height['x'] + x_tolerance]

        closest_width = None

        if widths_center:
            # Find closest in center range (by distance)
            closest_width = min(widths_center, key=lambda w:
                ((height['x'] - w['x'])**2 + (height['y'] - w['y'])**2)**0.5)
            print(f"    Found width in center range: {closest_width['text']} at ({closest_width['x']:.0f}, {closest_width['y']:.0f})")
        elif widths_left:
            # No widths in center, find closest to the left
            closest_width = min(widths_left, key=lambda w:
                ((height['x'] - w['x'])**2 + (height['y'] - w['y'])**2)**0.5)
            print(f"    No widths in center, found width to the left: {closest_width['text']} at ({closest_width['x']:.0f}, {closest_width['y']:.0f})")
        elif widths_right:
            # No widths to left, find closest to the right
            closest_width = min(widths_right, key=lambda w:
                ((height['x'] - w['x'])**2 + (height['y'] - w['y'])**2)**0.5)
            print(f"    No widths in center or left, found width to the right: {closest_width['text']} at ({closest_width['x']:.0f}, {closest_width['y']:.0f})")

        if closest_width:
            opening = {
                'width': closest_width['text'],
                'height': height['text'],
                'width_pos': (closest_width['x'], closest_width['y']),
                'height_pos': (height['x'], height['y']),
                'distance': ((closest_width['x'] - height['x'])**2 + (closest_width['y'] - height['y'])**2)**0.5,
                'width_is_finished': closest_width.get('is_finished_size', False),
                'height_is_finished': height.get('is_finished_size', False)
            }

            if 'notation' in height:
                opening['notation'] = height['notation']

            openings.append(opening)
            paired_heights.add(id(height))
            print(f"    PAIRED: {closest_width['text']} x {height['text']}")
        else:
            print(f"    No suitable width found - SKIP")

    return openings, unpaired_heights_info


def analyze_drawer_above_width(width, extent, image, heights=None):
    """
    Analyze the area above a width measurement to determine drawer configuration.
    Scans for H-lines to determine if there's 1 drawer or multiple drawers above.

    Args:
        width: Width measurement dict with 'position_class' field
        extent: Width extent dict with left/right bounds
        image: The source image
        heights: List of height measurements (optional, used to determine scan distance)

    Returns:
        String: drawer configuration with position class label
    """
    from line_detection import detect_lines_in_roi
    import cv2

    if image is None:
        return "unknown"

    width_left = int(extent['left'])
    width_right = int(extent['right'])
    width_y = int(width['y'])

    # Calculate the actual angle from the extent line coordinates
    left_y = extent.get('left_y', width_y)
    right_y = extent.get('right_y', width_y)
    dx = width_right - width_left
    dy = right_y - left_y
    extent_angle = np.degrees(np.arctan2(dy, dx))
    print(f"    Extent angle calculated: {extent_angle:.2f}° (from left_y={left_y}, right_y={right_y})")

    # Determine scan height based on position class
    # Top widths: 5/8" = 60px at 96 DPI
    # Bottom widths: scan to smallest height above, or 4" default
    position_class = width.get('position_class', 'bottom')

    if position_class == 'top':
        scan_distance_px = 60  # 5/8 inch at 96 DPI
        class_label = "top width"
    else:
        # For bottom widths: find the HEIGHT with smallest VALUE above this width
        # Then scan to that height's Y position
        from shared_utils import fraction_to_decimal
        smallest_height_value = None
        smallest_height_y = None
        if heights:
            for height in heights:
                height_y = height['y']
                # Height is above the width if its Y is smaller (higher on page)
                if height_y < width_y:
                    # Parse the height value
                    height_decimal = fraction_to_decimal(height['text'])
                    if height_decimal:
                        if smallest_height_value is None or height_decimal < smallest_height_value:
                            smallest_height_value = height_decimal
                            smallest_height_y = height_y

        # Use distance to smallest height VALUE + 1", or 4" default if no heights found
        if smallest_height_y:
            scan_distance_px = int(width_y - smallest_height_y) + 96  # +1" (96px at 96 DPI)
            print(f"    Scan distance: {scan_distance_px}px (to smallest height value: {smallest_height_value}\" + 1\")")
        else:
            scan_distance_px = 384  # 4 inches at 96 DPI default
            print(f"    Scan distance: {scan_distance_px}px (default 4\")")

        class_label = "bottom width"

    # Define ROI above the width measurement
    # Start 1/8" (12px) higher than the width measurement
    # Horizontally: use the width extent (left to right of dimension line)
    offset_from_width = 12  # 1/8 inch at 96 DPI
    roi_x1 = width_left
    roi_x2 = width_right
    roi_y2 = width_y - offset_from_width  # Start 1/8" above the width measurement
    roi_y1 = max(0, roi_y2 - scan_distance_px)  # Scan distance above that point

    # Ensure ROI is within image bounds
    roi_x1 = max(0, roi_x1)
    roi_x2 = min(image.shape[1], roi_x2)
    roi_y1 = max(0, roi_y1)
    roi_y2 = min(image.shape[0], roi_y2)

    # Extract ROI
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

    if roi.size == 0:
        return f"{class_label}: unknown"

    # Use edge detection to find horizontal edges
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find horizontal edges by scanning with dynamic row height based on extent angle
    # Calculate how many pixels tall we need to scan to capture an angled edge
    roi_height, roi_width = edges.shape

    # Calculate vertical rise across the ROI width at this angle
    vertical_rise = abs(np.tan(np.radians(extent_angle)) * roi_width)
    # Row height should be at least the vertical rise, minimum 1px, maximum 50px
    scan_row_height = max(1, min(50, int(vertical_rise)))

    print(f"    Extent angle {extent_angle:.2f}° → scan row height: {scan_row_height}px (vertical rise: {vertical_rise:.1f}px across {roi_width}px width)")

    horizontal_edge_rows = []

    # Scan in steps of scan_row_height
    edge_percentages = []
    for y in range(0, roi_height, scan_row_height):
        y_end = min(y + scan_row_height, roi_height)
        # Count edge pixels in this row band
        edge_count = np.sum(edges[y:y_end, :] > 0)
        total_pixels = (y_end - y) * roi_width
        edge_percentage = edge_count / total_pixels if total_pixels > 0 else 0
        edge_percentages.append(edge_percentage)

        # Lower threshold for taller bands (shelf edges are thin, maybe 2-3px)
        # For a 3px thick edge across full width in a 36px tall band:
        # 3 × 370 = 1110 pixels out of 13,320 total = 8.3%
        threshold = 0.015  # 1.5% threshold
        if edge_percentage > threshold:
            horizontal_edge_rows.append(y + (y_end - y) // 2)  # Use middle of the band

    # Show statistics
    if edge_percentages:
        print(f"    Edge % range: min={min(edge_percentages)*100:.2f}%, max={max(edge_percentages)*100:.2f}%, avg={np.mean(edge_percentages)*100:.2f}%")

    print(f"    Found {len(horizontal_edge_rows)} row bands with >{threshold*100:.0f}% edge pixels (using {scan_row_height}px tall bands)")

    # Cluster nearby rows (within 20 pixels) into single edges
    edge_count = 0
    clustered_edges = []  # Store the Y positions of clustered edges
    if horizontal_edge_rows:
        edge_count = 1
        cluster_start = horizontal_edge_rows[0]
        for i in range(1, len(horizontal_edge_rows)):
            if horizontal_edge_rows[i] - horizontal_edge_rows[i-1] > 20:  # Changed from 10 to 20
                # Save the middle of the previous cluster
                clustered_edges.append((cluster_start + horizontal_edge_rows[i-1]) // 2)
                edge_count += 1
                cluster_start = horizontal_edge_rows[i]
        # Save the last cluster
        clustered_edges.append((cluster_start + horizontal_edge_rows[-1]) // 2)

    scan_height = roi_y2 - roi_y1  # Height of scanned area

    # Store edge visualization data
    edge_viz_data = {
        'roi_bounds': (roi_x1, roi_y1, roi_x2, roi_y2),
        'horizontal_edges': [(roi_x1, roi_y1 + y, roi_x2, roi_y1 + y) for y in clustered_edges]
    }

    # Determine drawer configuration based on edge count
    if edge_count <= 5:
        return f"{class_label}: 1 drawer ({edge_count}E {scan_height}px)", edge_viz_data
    else:
        return f"{class_label}: multiple drawers ({edge_count}E {scan_height}px)", edge_viz_data
