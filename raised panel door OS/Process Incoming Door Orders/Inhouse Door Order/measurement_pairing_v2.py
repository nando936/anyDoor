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

    print(f"\n  [DOWN ARROW SCAN] Total heights: {len(heights)}")

    # For each height measurement, look for a down arrow below it
    for height in heights:
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

        # Save debug images
        import os
        debug_dir = os.path.dirname(image.__class__.__name__)  # Get directory
        # Actually, get image path from somewhere... for now just save to /tmp
        cv2.imwrite(f"/tmp/arrow_search_roi_{int(height_x)}_{int(height_y)}.png", roi)
        cv2.imwrite(f"/tmp/arrow_search_green_{int(height_x)}_{int(height_y)}.png", green_mask)

        # Detect edges only on green-filtered image
        edges = cv2.Canny(green_mask, 30, 100)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)

        has_down_arrow = False
        if lines is not None and len(lines) >= 2:
            print(f"    Found {len(lines)} lines in green-filtered ROI")

            # Print ALL angles first to debug
            all_angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Normalize angle to 0-360 range
                if angle < 0:
                    angle += 360
                all_angles.append(angle)
            print(f"    All line angles: {[f'{a:.1f}' for a in sorted(all_angles)]}")

            # Look for down arrow: two lines forming V pointing down
            # Based on actual measurements:
            # Right leg: 20-88° (going down-right from apex, allows steeper arrows)
            # Left leg: 285-360° (going down-left, allows very narrow V like 28 1/16)
            down_right_lines = []  # Lines going down-right (20-80°)
            down_left_lines = []   # Lines going down-left (285-360°)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                # Normalize angle to 0-360 range
                if angle < 0:
                    angle += 360

                # Check if line is right leg of V (20° to 88°)
                if 20 <= angle <= 88:
                    down_right_lines.append((x1, y1, x2, y2, angle))
                # Check if line is left leg of V (285° to 360°, allows very narrow V)
                elif 285 <= angle <= 360:
                    down_left_lines.append((x1, y1, x2, y2, angle))

            print(f"    Down-right lines (20-88°): {len(down_right_lines)}")
            print(f"    Down-left lines (285-360°): {len(down_left_lines)}")

            # Check if we have at least one line in each direction
            # If both leg types exist, we have an arrow (even if fragmented)
            if down_right_lines and down_left_lines:
                has_down_arrow = True
                print(f"      FOUND ARROW: {len(down_right_lines)} DR lines, {len(down_left_lines)} DL lines")

        print(f"    Down arrow detected: {has_down_arrow}")

        if has_down_arrow:
            # Found a down arrow - record the Y position (near top of search area, where arrow actually is)
            # Arrow is typically 50px below the height text, not at center of 200px ROI
            arrow_y = int(roi_y1 + 50)  # 50px from top of ROI (closer to actual arrow position)
            arrow_x = int(height_x)
            down_arrow_positions.append((arrow_x, arrow_y))
            print(f"    Added down arrow at ({arrow_x}, {arrow_y})")

    return down_arrow_positions


def pair_measurements_by_proximity(classified_measurements, all_measurements, image=None, image_path=None):
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
    down_arrow_y_positions = []  # Initialize for return value
    if widths and image is not None:
        import cv2
        import numpy as np

        # Find down arrows below height measurements
        down_arrow_y_positions = find_down_arrows_near_heights(image, heights)

        # Store bottom width reference line for visualization
        bottom_width_line = None

        if down_arrow_y_positions:
            # LINE-BASED APPROACH:
            # 1. Split arrows into LEFT side vs RIGHT side (using image midpoint)
            # 2. Find the LOWEST arrow on LEFT side (max Y)
            # 3. Find the LOWEST arrow on RIGHT side (max Y)
            # 4. Connect those two arrows

            print(f"\n  Down arrow classification (LINE-BASED):")
            print(f"    Found {len(down_arrow_y_positions)} total down arrows")

            # Get image midpoint X
            image_midpoint_x = image.shape[1] / 2
            print(f"    Image midpoint X: {image_midpoint_x:.0f}")

            # Split arrows into left and right sides
            left_arrows = [pos for pos in down_arrow_y_positions if pos[0] < image_midpoint_x]
            right_arrows = [pos for pos in down_arrow_y_positions if pos[0] >= image_midpoint_x]

            print(f"    Left side arrows: {len(left_arrows)}")
            print(f"    Right side arrows: {len(right_arrows)}")

            if len(left_arrows) >= 1 and len(right_arrows) >= 1:
                # Find lowest arrow on left side (maximum Y)
                leftmost_lowest_arrow = max(left_arrows, key=lambda pos: pos[1])
                # Find lowest arrow on right side (maximum Y)
                rightmost_lowest_arrow = max(right_arrows, key=lambda pos: pos[1])

                print(f"    Lowest arrow on LEFT side: ({leftmost_lowest_arrow[0]}, {leftmost_lowest_arrow[1]})")
                print(f"    Lowest arrow on RIGHT side: ({rightmost_lowest_arrow[0]}, {rightmost_lowest_arrow[1]})")

                # Store line for visualization
                bottom_width_line = {
                    'start': leftmost_lowest_arrow,
                    'end': rightmost_lowest_arrow
                }

                # Calculate GLOBAL smallest height value across ALL heights
                # This will be used to create ONE shared offset reference line
                from shared_utils import fraction_to_decimal
                global_smallest_height_value = None
                global_smallest_height_y = None

                for height in heights:
                    height_decimal = fraction_to_decimal(height['text'])
                    if height_decimal:
                        if global_smallest_height_value is None or height_decimal < global_smallest_height_value:
                            global_smallest_height_value = height_decimal
                            global_smallest_height_y = height['y']

                # Calculate ONE offset reference line based on global smallest height
                if global_smallest_height_y:
                    # Find a representative bottom width Y position (use leftmost_lowest_arrow Y)
                    ref_y = leftmost_lowest_arrow[1]
                    scan_distance_px = int(ref_y - global_smallest_height_y)  # Distance to smallest height (no extra offset)
                    print(f"    Global smallest height: {global_smallest_height_value}\" at Y={global_smallest_height_y}")
                    print(f"    Global offset distance: {scan_distance_px}px (to smallest height)")

                    # Calculate perpendicular unit vector for the reference line (pointing UP)
                    ref_x1, ref_y1 = leftmost_lowest_arrow
                    ref_x2, ref_y2 = rightmost_lowest_arrow
                    ref_dx = ref_x2 - ref_x1
                    ref_dy = ref_y2 - ref_y1
                    ref_length = np.sqrt(ref_dx**2 + ref_dy**2)

                    if ref_length > 0:
                        # Unit direction vector
                        ref_ux = ref_dx / ref_length
                        ref_uy = ref_dy / ref_length

                        # Perpendicular unit vector (rotate 90° counterclockwise)
                        perp_x = -ref_uy
                        perp_y = ref_ux

                        # We want the perpendicular that points UP (towards smaller Y)
                        if perp_y > 0:  # Points down (larger Y)
                            perp_x = -perp_x
                            perp_y = -perp_y

                        # Offset both endpoints of reference line UP by scan_distance_px
                        offset_ref_x1 = ref_x1 + perp_x * scan_distance_px
                        offset_ref_y1 = ref_y1 + perp_y * scan_distance_px
                        offset_ref_x2 = ref_x2 + perp_x * scan_distance_px
                        offset_ref_y2 = ref_y2 + perp_y * scan_distance_px

                        # Store the SHARED offset reference line
                        bottom_width_line['offset_reference_line'] = {
                            'start': (int(offset_ref_x1), int(offset_ref_y1)),
                            'end': (int(offset_ref_x2), int(offset_ref_y2))
                        }
                        print(f"    Global offset reference line: ({offset_ref_x1:.1f}, {offset_ref_y1:.1f}) to ({offset_ref_x2:.1f}, {offset_ref_y2:.1f})")

                # Distance threshold from the line
                perpendicular_tolerance = 200

                print(f"    Bottom width threshold: ±{perpendicular_tolerance}px from reference line")

                # Classify widths based on perpendicular distance from the line
                for width in widths:
                    width_point = (width['x'], width['y'])

                    # Calculate perpendicular distance from width to the line
                    # Line equation: from (x1,y1) to (x2,y2)
                    x1, y1 = leftmost_lowest_arrow
                    x2, y2 = rightmost_lowest_arrow
                    wx, wy = width_point

                    # If line is just a point (leftmost == rightmost), use simple Y distance
                    if abs(x2 - x1) < 1 and abs(y2 - y1) < 1:
                        distance = abs(wy - y1)
                    else:
                        # Perpendicular distance formula: |ax + by + c| / sqrt(a^2 + b^2)
                        # Line in form: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
                        # So: a = (y2-y1), b = -(x2-x1), c = (x2-x1)y1 - (y2-y1)x1
                        a = y2 - y1
                        b = -(x2 - x1)
                        c = (x2 - x1) * y1 - (y2 - y1) * x1

                        distance = abs(a * wx + b * wy + c) / np.sqrt(a**2 + b**2)

                    if distance <= perpendicular_tolerance:
                        width['position_class'] = 'bottom'
                        print(f"      Classified '{width['text']}' at ({width['x']:.0f}, {width['y']:.0f}) as BOTTOM (dist={distance:.1f}px)")

                        # IMPORTANT: Copy position_class back to original measurements_list
                        # So visualization can show "BOTTOM WIDTH" label
                        for meas in all_measurements:
                            if (meas.get('text') == width['text'] and
                                abs(meas['position'][0] - width['x']) < 1 and
                                abs(meas['position'][1] - width['y']) < 1):
                                meas['position_class'] = 'bottom'
                                meas['bottom_width_line'] = bottom_width_line  # Store line for viz
                                break
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

    # Calculate extent and analyze drawer configuration ONLY for bottom widths
    print(f"\n=== CALCULATING EXTENT AND ANALYZING DRAWER CONFIGURATION (BOTTOM WIDTHS ONLY) ===")
    for i, meas in enumerate(all_measurements):
        if i < len(classified_measurements):
            category = classified_measurements[i]
            if category == 'width':
                # Find corresponding width in widths list
                for width in widths:
                    if width['x'] == meas['position'][0] and width['y'] == meas['position'][1]:
                        # Only calculate extent and analyze if this is a classified bottom width
                        if width.get('position_class') == 'bottom':
                            # Copy bottom_width_line from meas to width dict (if available)
                            if 'bottom_width_line' in meas:
                                width['bottom_width_line'] = meas['bottom_width_line']

                            # Calculate extent for this bottom width
                            print(f"  Calculating extent for bottom width: {meas['text']} at ({meas['position'][0]:.0f}, {meas['position'][1]:.0f})")
                            from line_detection import extend_roi_and_get_full_extent
                            extent = extend_roi_and_get_full_extent(image, meas, 'width', image_path)

                            # Check if extent calculation succeeded or failed
                            if extent and not extent.get('failed'):
                                # Store extent in both width dict and meas dict
                                width['width_extent'] = extent
                                meas['width_extent'] = extent

                                # Analyze drawer configuration using the calculated extent
                                drawer_config, edge_viz_data = analyze_drawer_above_width(width, extent, image, heights)
                                width['drawer_config'] = drawer_config
                                width['edge_viz_data'] = edge_viz_data
                                # Store drawer_config and edge_viz_data back in all_measurements so viz can access it
                                meas['drawer_config'] = drawer_config
                                meas['edge_viz_data'] = edge_viz_data
                                print(f"  {meas['text']}: {drawer_config}")
                            elif extent and extent.get('failed'):
                                # Extent calculation failed, but store it anyway for visualization (has debug_rois)
                                meas['width_extent'] = extent
                                print(f"  WARNING: Failed to calculate extent for {meas['text']} (arrows not found), but stored debug_rois for visualization")
                            else:
                                print(f"  WARNING: Failed to calculate extent for {meas['text']}")
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
        # Add padding ONLY to left side (where drawer heights are positioned)
        x_padding_left = 120  # Padding on left side to capture drawer heights
        scan_left = width_left - x_padding_left
        scan_right = width_right  # No padding on right side
        print(f"    Scan area X range: {scan_left:.0f} to {scan_right:.0f} (extent with {x_padding_left}px left padding)")

        # Find heights within the scan area X range (extent + horizontal padding)
        candidate_heights = []
        y_padding = 50  # Padding in pixels to account for skew

        # Get extent line Y coordinates (use max as baseline since Y increases downward)
        extent_left_y = extent.get('left_y', width['y'])
        extent_right_y = extent.get('right_y', width['y'])
        extent_baseline_y = max(extent_left_y, extent_right_y)  # Lowest point of extent line

        for height in heights:
            # Check if height CENTER is within scan area
            # Use center position (X) regardless of bounds
            height_x = height['x']
            overlaps = scan_left <= height_x <= scan_right

            if overlaps:
                # Only consider heights above the EXTENT LINE (not width text position)
                if height['y'] < extent_baseline_y + y_padding:
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

                    # Add position_class if width has it (e.g., 'bottom')
                    if 'position_class' in width:
                        opening['position_class'] = width['position_class']

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

    return openings, unpaired_heights_info, down_arrow_y_positions, bottom_width_line


def analyze_drawer_above_width(width, extent, image, heights=None):
    """
    Analyze the area above a width measurement to determine drawer configuration.
    Scans for H-lines to determine if there's 1 drawer or multiple drawers above.

    Args:
        width: Width measurement dict with 'position_class' field
        extent: Width extent dict with left/right bounds and Y coordinates
        image: The source image
        heights: List of height measurements (optional, used to determine scan distance)

    Returns:
        String: drawer configuration with position class label
    """
    from line_detection import detect_lines_in_roi, extract_rotated_roi
    import cv2

    if image is None:
        return "unknown"

    # Get extent line endpoints (with their Y coordinates)
    width_left = int(extent['left'])
    width_right = int(extent['right'])
    width_left_y = int(extent.get('left_y', width['y']))  # Use extent Y if available
    width_right_y = int(extent.get('right_y', width['y']))  # Otherwise fall back to width Y
    width_y = int(width['y'])

    # Use the angle from the extent dict (calculated from h_lines during classification)
    extent_angle = extent.get('angle', 0)
    print(f"    Extent angle from dict: {extent_angle:.2f}°")

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

    # Calculate angled parallelogram scan area
    # Bottom edge: Extent line offset 12px above width measurement
    # Top edge: Parallel line at smallest height Y position (for bottom widths) or scan_distance_px above (for top widths)
    offset_from_width = 12  # 1/8 inch at 96 DPI

    # Bottom edge corners (12px above the extent line endpoints)
    bottom_left_x = width_left
    bottom_left_y = width_left_y - offset_from_width
    bottom_right_x = width_right
    bottom_right_y = width_right_y - offset_from_width

    # Get bottom width reference line if available (for bottom widths only)
    bottom_width_line = width.get('bottom_width_line', None)

    # Initialize offset reference line coordinates (will be retrieved from bottom_width_line if available)
    offset_ref_x1 = None
    offset_ref_y1 = None
    offset_ref_x2 = None
    offset_ref_y2 = None

    # Use SHARED offset reference line from bottom_width_line (calculated once in pair_measurements_by_proximity)
    if position_class == 'bottom' and bottom_width_line is not None:
        # Check if shared offset reference line exists
        if 'offset_reference_line' in bottom_width_line:
            # Retrieve the SHARED offset reference line (already calculated)
            offset_line = bottom_width_line['offset_reference_line']
            offset_ref_x1, offset_ref_y1 = offset_line['start']
            offset_ref_x2, offset_ref_y2 = offset_line['end']

            # Get bottom width reference line endpoints for logging
            ref_x1, ref_y1 = bottom_width_line['start']
            ref_x2, ref_y2 = bottom_width_line['end']

            print(f"    Using SHARED offset reference line:")
            print(f"      Bottom width reference: ({ref_x1}, {ref_y1}) to ({ref_x2}, {ref_y2})")
            print(f"      Offset reference: ({offset_ref_x1:.1f}, {offset_ref_y1:.1f}) to ({offset_ref_x2:.1f}, {offset_ref_y2:.1f})")

            # Now find where VERTICAL lines from extent endpoints intersect the offset reference line
            # For trapezoid with vertical left/right sides:
            # - Shoot vertical line UP from bottom-left corner
            # - Shoot vertical line UP from bottom-right corner
            # - Find where these vertical lines intersect the offset reference line

            # Vertical line direction (pointing UP)
            vertical_dir_x = 0
            vertical_dir_y = -1  # Negative Y is up

            # Line-line intersection:
            # Line 1 (vertical from left): P = (bottom_left_x, bottom_left_y) + t * (0, -1)
            # Line 2 (offset ref): Q = (offset_ref_x1, offset_ref_y1) + s * (offset_ref_x2 - offset_ref_x1, offset_ref_y2 - offset_ref_y1)

            # Solve for intersection
            def line_intersection(p1, d1, p2, d2):
                """Find intersection of two lines defined by point + direction.
                p1, d1: point and direction of line 1
                p2, d2: point and direction of line 2
                Returns intersection point or None if parallel
                """
                # Using parametric form:
                # p1 + t*d1 = p2 + s*d2
                # Solve: t*d1 - s*d2 = p2 - p1

                det = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(det) < 1e-10:
                    return None  # Lines are parallel

                dp = (p2[0] - p1[0], p2[1] - p1[1])
                t = (dp[0] * d2[1] - dp[1] * d2[0]) / det

                # Calculate intersection point
                intersection = (p1[0] + t * d1[0], p1[1] + t * d1[1])
                return intersection

            # Calculate left intersection (vertical line UP from bottom-left)
            left_extent_point = (bottom_left_x, bottom_left_y)
            left_extent_dir = (vertical_dir_x, vertical_dir_y)
            offset_ref_point = (offset_ref_x1, offset_ref_y1)
            offset_ref_dir = (offset_ref_x2 - offset_ref_x1, offset_ref_y2 - offset_ref_y1)

            left_intersection = line_intersection(left_extent_point, left_extent_dir, offset_ref_point, offset_ref_dir)

            # Calculate right intersection (vertical line UP from bottom-right)
            right_extent_point = (bottom_right_x, bottom_right_y)
            right_extent_dir = (vertical_dir_x, vertical_dir_y)

            right_intersection = line_intersection(right_extent_point, right_extent_dir, offset_ref_point, offset_ref_dir)

            if left_intersection and right_intersection:
                top_left_x, top_left_y = left_intersection
                top_right_x, top_right_y = right_intersection
                print(f"    Trapezoid top edge from extent/offset-ref intersections:")
                print(f"      Left intersection: ({top_left_x:.1f}, {top_left_y:.1f})")
                print(f"      Right intersection: ({top_right_x:.1f}, {top_right_y:.1f})")
            else:
                # Fallback: use horizontal line at smallest_height_y
                print(f"    WARNING: Could not calculate intersections, falling back to horizontal line")
                top_edge_y = smallest_height_y if smallest_height_y else (bottom_left_y - scan_distance_px)
                dy_extent = bottom_right_y - bottom_left_y
                top_left_x = bottom_left_x
                top_left_y = top_edge_y
                top_right_x = bottom_right_x
                top_right_y = top_edge_y + dy_extent
        else:
            # Fallback: reference line is a point
            print(f"    WARNING: Reference line has zero length, falling back to horizontal line")
            top_edge_y = smallest_height_y if smallest_height_y else (bottom_left_y - scan_distance_px)
            dy_extent = bottom_right_y - bottom_left_y
            top_left_x = bottom_left_x
            top_left_y = top_edge_y
            top_right_x = bottom_right_x
            top_right_y = top_edge_y + dy_extent
    else:
        # OLD APPROACH: For top widths or when no bottom width reference line available
        # Calculate top edge Y position
        if position_class == 'bottom' and smallest_height_y:
            # For bottom widths: use smallest height Y as target
            top_edge_y = smallest_height_y
        else:
            # For top widths: use scan_distance_px above bottom edge
            # Use the average Y of bottom edge
            avg_bottom_y = (bottom_left_y + bottom_right_y) / 2
            top_edge_y = avg_bottom_y - scan_distance_px

        # Calculate top edge corners to create trapezoid with parallel top/bottom edges
        # Keep X coordinates same (vertical left/right sides), adjust Y to maintain parallel slope

        # Calculate the Y difference across the extent line (this maintains the angle)
        dy_extent = bottom_right_y - bottom_left_y

        # Top edge: same X positions as bottom edge, Y offset to maintain parallel angle
        top_left_x = bottom_left_x
        top_left_y = top_edge_y
        top_right_x = bottom_right_x
        top_right_y = top_edge_y + dy_extent  # Maintains same slope as extent line

    # Define trapezoid corners (clockwise from top-left)
    # Top and bottom edges are parallel (at extent angle), left and right edges are vertical
    trapezoid_corners = np.array([
        [top_left_x, top_left_y],       # Top-left
        [top_right_x, top_right_y],     # Top-right
        [bottom_right_x, bottom_right_y],  # Bottom-right
        [bottom_left_x, bottom_left_y]     # Bottom-left
    ], dtype=np.float32)

    print(f"    Trapezoid corners:")
    print(f"      Top-left: ({top_left_x:.1f}, {top_left_y:.1f})")
    print(f"      Top-right: ({top_right_x:.1f}, {top_right_y:.1f})")
    print(f"      Bottom-right: ({bottom_right_x:.1f}, {bottom_right_y:.1f})")
    print(f"      Bottom-left: ({bottom_left_x:.1f}, {bottom_left_y:.1f})")

    # Calculate dimensions of the straightened ROI
    roi_width = int(np.sqrt((top_right_x - top_left_x)**2 + (top_right_y - top_left_y)**2))
    # For trapezoid with vertical sides, height is the vertical distance (same X coordinates)
    roi_height = int(abs(bottom_left_y - top_left_y))

    # Define destination rectangle (straightened trapezoid)
    dst_corners = np.array([
        [0, 0],                      # Top-left
        [roi_width, 0],              # Top-right
        [roi_width, roi_height],     # Bottom-right
        [0, roi_height]              # Bottom-left
    ], dtype=np.float32)

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(trapezoid_corners, dst_corners)
    # Get inverse transform (to map ROI coords back to image coords)
    M_inv = cv2.getPerspectiveTransform(dst_corners, trapezoid_corners)

    # Extract and straighten the trapezoid ROI
    roi = cv2.warpPerspective(image, M, (roi_width, roi_height))

    # Store trapezoid corners and inverse matrix for visualization
    roi_parallelogram = trapezoid_corners  # Keep field name for backward compatibility
    inverse_matrix = M_inv

    if roi.size == 0:
        return f"{class_label}: unknown"

    # Use edge detection to find horizontal edges in the straightened ROI
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find horizontal edges by scanning rows
    # Since ROI is now straightened (horizontal), we can use simple row scanning
    roi_height, roi_width = edges.shape
    scan_row_height = 50  # Fixed row height since ROI is straightened

    print(f"    Straightened ROI size: {roi_width}px × {roi_height}px, scanning with {scan_row_height}px bands")

    horizontal_edge_rows = []

    # Scan in steps of scan_row_height
    edge_percentages = []
    for y in range(0, roi_height, scan_row_height):
        y_end = min(y + scan_row_height, roi_height)
        # Count edge pixels in this row band
        edge_count_in_band = np.sum(edges[y:y_end, :] > 0)
        total_pixels = (y_end - y) * roi_width
        edge_percentage = edge_count_in_band / total_pixels if total_pixels > 0 else 0
        edge_percentages.append(edge_percentage)

        # Lower threshold for detecting drawer edges
        threshold = 0.015  # 1.5% threshold
        if edge_percentage > threshold:
            horizontal_edge_rows.append(y + (y_end - y) // 2)  # Use middle of the band

    # Show statistics
    if edge_percentages:
        print(f"    Edge % range: min={min(edge_percentages)*100:.2f}%, max={max(edge_percentages)*100:.2f}%, avg={np.mean(edge_percentages)*100:.2f}%")

    print(f"    Found {len(horizontal_edge_rows)} row bands with >{threshold*100:.0f}% edge pixels")

    # Cluster nearby rows (within 20 pixels) into single edges
    edge_count = 0
    clustered_edges_roi = []  # Store Y positions in ROI coordinates
    if horizontal_edge_rows:
        edge_count = 1
        cluster_start = horizontal_edge_rows[0]
        for i in range(1, len(horizontal_edge_rows)):
            if horizontal_edge_rows[i] - horizontal_edge_rows[i-1] > 20:
                # Save the middle of the previous cluster
                clustered_edges_roi.append((cluster_start + horizontal_edge_rows[i-1]) // 2)
                edge_count += 1
                cluster_start = horizontal_edge_rows[i]
        # Save the last cluster
        clustered_edges_roi.append((cluster_start + horizontal_edge_rows[-1]) // 2)

    # Transform edge positions back to original image coordinates
    # Each edge is a horizontal line in the straightened ROI
    # We need to transform it back to an angled line in the original image
    horizontal_edges_original = []
    for edge_y_roi in clustered_edges_roi:
        # Create points at left and right ends of edge in ROI coordinates
        points_roi = np.array([
            [[0, edge_y_roi]],
            [[roi_width, edge_y_roi]]
        ], dtype=np.float32)

        # Transform back to original image coordinates using perspective transform
        points_orig = cv2.perspectiveTransform(points_roi, inverse_matrix)

        # Extract coordinates
        left_x, left_y = points_orig[0][0]
        right_x, right_y = points_orig[1][0]

        # Store as line endpoints
        horizontal_edges_original.append((
            int(left_x), int(left_y),
            int(right_x), int(right_y)
        ))

    # Calculate scan height (vertical distance of parallelogram)
    scan_height = int(abs(bottom_left_y - top_left_y))

    # Store edge visualization data with parallelogram corners
    edge_viz_data = {
        'roi_parallelogram': roi_parallelogram.tolist(),  # 4 corners of parallelogram
        'horizontal_edges': horizontal_edges_original  # List of (x1, y1, x2, y2) tuples
    }

    # NOTE: Offset reference line is NOT stored here because it's shared across all bottom widths
    # It will be drawn once from bottom_width_line['offset_reference_line'] in visualization

    # Determine drawer configuration based on edge count
    if edge_count <= 5:
        return f"{class_label}: 1 drawer ({edge_count}E {scan_height}px)", edge_viz_data
    else:
        return f"{class_label}: multiple drawers ({edge_count}E {scan_height}px)", edge_viz_data
