#!/usr/bin/env python3
"""
Measurement pairing logic V2.
New improved approach for pairing width and height measurements.
"""

import numpy as np
import cv2
import subprocess
import tempfile
import os
import platform
from measurement_config import HSV_CONFIG


def verify_arrow_with_claude(roi_image, arrow_direction, image_path=None, measurement_id=None, measurement_value=None):
    """
    Use Claude CLI to verify if an arrow is visible in the ROI image.

    Args:
        roi_image: OpenCV color image (numpy array) of the ROI
        arrow_direction: 'down', 'left', or 'right'
        image_path: Path to the original page image (for saving verification image in same folder)
        measurement_id: Measurement ID (e.g., "M3")
        measurement_value: Measurement value (e.g., "22 7/8")

    Returns:
        True if Claude sees the arrow, False otherwise
    """
    # Debug: print received parameters
    print(f"    [DEBUG] image_path={image_path}, measurement_id={measurement_id}, measurement_value={measurement_value}")

    # Determine save path
    if image_path and measurement_id and measurement_value:
        # Save in same directory as page image with descriptive name
        image_dir = os.path.dirname(os.path.abspath(image_path))
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Clean base name (remove underscores/dashes for consistency)
        base_name_clean = base_name.replace('_', '').replace('-', '')
        # Clean measurement value (replace spaces and slashes)
        value_clean = measurement_value.replace(' ', '').replace('/', '-')
        # Create descriptive filename
        verification_filename = f"{base_name_clean}_{measurement_id}_{value_clean}_claude_verification.png"
        save_path = os.path.join(image_dir, verification_filename)
    else:
        # Fallback to temp directory
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, f"arrow_verify_{arrow_direction}.png")

    # Isolate green arrow with edge enhancement on white background
    # Step 1: Convert to HSV for green color filtering
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    # Step 2: Create mask for green pixels using HSV_CONFIG ranges
    lower_green = np.array(HSV_CONFIG['lower_green'])
    upper_green = np.array(HSV_CONFIG['upper_green'])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Step 3: Create white background image
    white_background = np.ones_like(roi_image, dtype=np.uint8) * 255

    # Step 4: Copy only green pixels to white background
    # Where green_mask is 255 (green pixel detected), copy original pixel
    # Where green_mask is 0 (no green), keep white background
    green_isolated = white_background.copy()
    green_isolated[green_mask > 0] = roi_image[green_mask > 0]

    # Step 5: Add edge enhancement on the green mask
    # Detect edges on the green mask (not the full image)
    edges = cv2.Canny(green_mask, 50, 150)

    # Step 6: Draw edges in red for high visibility
    # Where edges exist, set pixel to red (BGR: 0, 0, 255)
    edge_mask = edges > 0
    green_isolated[edge_mask] = [0, 0, 255]  # Red in BGR

    # Save cleaned-up image (green arrow + red edges on white background)
    cv2.imwrite(save_path, green_isolated)
    print(f"    [CLAUDE VERIFICATION] Saved verification image: {save_path}")
    print(f"    [CLAUDE VERIFICATION] Image shows: green arrow isolated on white background with red edge enhancement")

    try:
        # Determine claude command based on platform
        if platform.system() == "Windows":
            claude_cmd = "claude.cmd"
        else:
            claude_cmd = "claude"

        # Ask Claude if arrow is visible with clearer prompt
        if arrow_direction == "down":
            question = "Look at this image. Do you see a green arrow pointing downward? Answer only YES or NO."
        elif arrow_direction == "left":
            question = "Look at this image. Do you see a green arrow pointing to the left? Answer only YES or NO."
        elif arrow_direction == "right":
            question = "Look at this image. Do you see a green arrow pointing to the right? Answer only YES or NO."
        else:
            question = f"Look at this image. Do you see a green arrow pointing {arrow_direction}? Answer only YES or NO."

        # Get image directory for --add-dir permission
        image_dir = os.path.dirname(os.path.abspath(save_path))

        # Build prompt with image path
        prompt = f"{question}\n\nImage: {save_path}"

        # Run claude CLI with correct syntax (same as claude_verification.py)
        result = subprocess.run(
            f'{claude_cmd} --print --add-dir "{image_dir}"',
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True
        )

        # Parse response
        response = result.stdout.strip().upper()
        claude_sees_arrow = "YES" in response

        print(f"    [CLAUDE VERIFICATION] {arrow_direction} arrow: Claude says {'YES' if claude_sees_arrow else 'NO'}")
        print(f"    [CLAUDE RESPONSE] {response[:100]}")

        return claude_sees_arrow

    except Exception as e:
        print(f"    [CLAUDE VERIFICATION ERROR] {str(e)}")
        return False


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


def find_down_arrows_near_heights(image, heights, all_measurements, non_measurement_text_exclusions=None, image_path=None):
    """
    Find down arrows below each height measurement (includes HEIGHT and UNCLASSIFIED).
    Returns a list of Y-coordinates where down arrows are found.

    Args:
        image: The source image
        heights: List of height measurement dicts (HEIGHT + UNCLASSIFIED) with 'x', 'y', and optionally 'height_extent'
        all_measurements: List of all measurements (used to exclude text areas from arrow search)
        non_measurement_text_exclusions: List of non-measurement items (OL/room names) to exclude from arrow detection
        image_path: Path to source image (for saving debug visualizations)
    """
    import cv2
    import numpy as np

    if image is None or not heights:
        return []

    down_arrow_positions = []

    # Get image directory and base name for debug images
    import os
    if image_path:
        image_dir = os.path.dirname(os.path.abspath(image_path))
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Remove underscores and dashes from base name for consistency with other debug files
        base_name_clean = base_name.replace('_', '').replace('-', '')
    else:
        image_dir = None
        base_name_clean = None

    print(f"\n  [DOWN ARROW SCAN] Scanning {len(heights)} measurements (HEIGHT + UNCLASSIFIED)")

    # For each height measurement (includes UNCLASSIFIED), look for a down arrow below it
    for height in heights:
        print(f"  [DOWN ARROW SCAN] Checking height '{height['text']}' at ({height['x']:.0f}, {height['y']:.0f})")
        height_x = height['x']
        height_y = height['y']

        # Define search region below the height measurement
        # Start search below text bottom with minimal gap (matches V-ROI gap from Phase 3)
        if 'bounds' in height and height['bounds']:
            text_bottom = height['bounds']['bottom']
            search_y_start = int(text_bottom + 5)  # 5px gap below text (matches V-ROI)
        else:
            # Fallback if bounds not available - estimate bottom from center
            search_y_start = int(height_y + 5)  # Assume ~5px from center to bottom + gap

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

        # GREEN ISOLATION PREPROCESSING (same as Claude verification)
        # This isolates the green arrow on a white background, removing cabinet edges, shadows, and noise

        # Step 1: Convert to HSV for green color filtering
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Step 2: Create mask for green pixels using HSV_CONFIG ranges
        lower_green = np.array(HSV_CONFIG['lower_green'])
        upper_green = np.array(HSV_CONFIG['upper_green'])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Step 3: Create white background image
        white_background = np.ones_like(roi, dtype=np.uint8) * 255

        # Step 4: Copy only green pixels to white background
        green_isolated = white_background.copy()
        green_isolated[green_mask > 0] = roi[green_mask > 0]

        # Step 5: Create mask for text exclusion (on green_mask, not on grayscale)
        # Start with the green mask (only green pixels), then further exclude text areas
        text_excluded_mask = green_mask.copy()

        # Exclude other measurement text areas from search to avoid detecting arrows in text
        for other_meas in all_measurements:
            # Skip the current target height measurement
            if other_meas.get('text') == height.get('text') and \
               abs(other_meas.get('x', 0) - height.get('x', 0)) < 5 and \
               abs(other_meas.get('y', 0) - height.get('y', 0)) < 5:
                continue

            if 'bounds' not in other_meas or not other_meas['bounds']:
                continue

            bounds = other_meas['bounds']
            # Calculate intersection with search ROI
            intersect_x1 = max(roi_x1, int(bounds['left']))
            intersect_y1 = max(roi_y1, int(bounds['top']))
            intersect_x2 = min(roi_x2, int(bounds['right']))
            intersect_y2 = min(roi_y2, int(bounds['bottom']))

            if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                # This text region intersects the search ROI - exclude it
                roi_ix1 = intersect_x1 - roi_x1
                roi_iy1 = intersect_y1 - roi_y1
                roi_ix2 = intersect_x2 - roi_x1
                roi_iy2 = intersect_y2 - roi_y1
                # Zero out this region in the mask
                text_excluded_mask[roi_iy1:roi_iy2, roi_ix1:roi_ix2] = 0

        # Also exclude OL text and room names from search
        if non_measurement_text_exclusions:
            for excluded_item in non_measurement_text_exclusions:
                if 'x' not in excluded_item or 'y' not in excluded_item:
                    continue

                # Estimate text bounds (assume ~100px wide, ~30px tall around center)
                ex = excluded_item['x']
                ey = excluded_item['y']
                text_width = 100
                text_height = 30

                # Calculate intersection with search ROI
                intersect_x1 = max(roi_x1, int(ex - text_width/2))
                intersect_y1 = max(roi_y1, int(ey - text_height/2))
                intersect_x2 = min(roi_x2, int(ex + text_width/2))
                intersect_y2 = min(roi_y2, int(ey + text_height/2))

                if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                    # This OL/room text intersects the search ROI - exclude it
                    roi_ix1 = intersect_x1 - roi_x1
                    roi_iy1 = intersect_y1 - roi_y1
                    roi_ix2 = intersect_x2 - roi_x1
                    roi_iy2 = intersect_y2 - roi_y1
                    # Zero out this region in the mask
                    text_excluded_mask[roi_iy1:roi_iy2, roi_ix1:roi_ix2] = 0

        # Step 6: Apply text exclusion to green isolated image
        green_isolated_masked = white_background.copy()
        green_isolated_masked[text_excluded_mask > 0] = green_isolated[text_excluded_mask > 0]

        # SAVE DEBUG IMAGE: Save green-isolated arrow detection image for ALL measurements
        if image_dir and base_name_clean:
            # Create visualization showing green isolated image with red edge overlay
            debug_vis = green_isolated_masked.copy()
            # Overlay red edges on the debug visualization
            edge_preview = cv2.Canny(cv2.cvtColor(green_isolated_masked, cv2.COLOR_BGR2GRAY), 50, 150)
            debug_vis[edge_preview > 0] = [0, 0, 255]  # Red edges

            # Get measurement ID for descriptive filename (same logic as Claude verification)
            meas_id = height.get('id', '')
            if not meas_id:
                # Try to find measurement ID from all_measurements
                for i, meas in enumerate(all_measurements):
                    if (abs(meas.get('x', 0) - height_x) < 5 and
                        abs(meas.get('y', 0) - height_y) < 5 and
                        meas.get('text') == height.get('text')):
                        meas_id = meas.get('id', i+1)
                        break

            # Use index+1 as fallback if no ID found
            if not meas_id:
                meas_id = heights.index(height) + 1

            # Format measurement value for filename (replace spaces and slashes)
            meas_value_clean = height['text'].replace(' ', '-').replace('/', '-')

            # Save with descriptive filename
            debug_filename = f"{base_name_clean}_M{meas_id}_{meas_value_clean}_down_arrow_detection.png"
            debug_path = os.path.join(image_dir, debug_filename)
            cv2.imwrite(debug_path, debug_vis)
            print(f"    [DEBUG] Saved arrow detection image: {debug_filename}")

        # Step 7: Convert green-isolated image to grayscale for edge detection
        gray_green = cv2.cvtColor(green_isolated_masked, cv2.COLOR_BGR2GRAY)

        # Step 8: Detect edges on the green-isolated grayscale image
        edges = cv2.Canny(gray_green, 50, 150)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)

        has_down_arrow = False
        if lines is not None and len(lines) >= 2:
            print(f"    Found {len(lines)} lines in edge-detected ROI")

            # Print ALL detected lines with FULL details (coordinates, lengths, angles, colors)
            print(f"    [DOWN ARROW LINE DETAILS] All detected lines:")
            all_angles = []
            for idx, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Normalize angle to 0-360 range
                if angle < 0:
                    angle += 360
                all_angles.append(angle)

                # Calculate line length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Determine color assignment based on angle range
                if 20 <= angle <= 88:
                    color_label = "GREEN (down-right)"
                elif 285 <= angle <= 360:
                    color_label = "BLUE (down-left)"
                else:
                    color_label = "NONE (out of range)"

                print(f"      Line {idx+1}: coords=({x1},{y1})-({x2},{y2}), angle={angle:.1f}°, length={length:.1f}px, color={color_label}")

            print(f"    All line angles: {[f'{a:.1f}' for a in sorted(all_angles)]}")

            # Look for down arrow: two lines forming V pointing down
            # Based on proven code from line_detection.py:
            # Right leg: 20-88° (going down-right from apex)
            # Left leg: 285-360° (going down-left from apex)
            down_right_lines = []  # Lines going down-right (20-88°)
            down_left_lines = []   # Lines going down-left (285-360°)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                # Normalize angle to 0-360 range
                if angle < 0:
                    angle += 360

                # Calculate line length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Check if line is right leg of V (20° to 88°)
                if 20 <= angle <= 88:
                    down_right_lines.append((x1, y1, x2, y2, angle, length))
                # Check if line is left leg of V (285° to 360°)
                elif 285 <= angle <= 360:
                    down_left_lines.append((x1, y1, x2, y2, angle, length))

            print(f"    Down-right lines (20-88°): {len(down_right_lines)}")
            print(f"    Down-left lines (285-360°): {len(down_left_lines)}")

            # Check if we have at least one line in each direction
            # Then validate that the lines are close together to form a valid arrow
            if down_right_lines and down_left_lines:
                # Check all combinations of DR and DL lines for valid arrow pairs
                arrow_detected = False
                for dr_line in down_right_lines:
                    x1_dr, y1_dr, x2_dr, y2_dr, angle_dr, length_dr = dr_line
                    center_dr_x = (x1_dr + x2_dr) / 2
                    center_dr_y = (y1_dr + y2_dr) / 2

                    for dl_line in down_left_lines:
                        x1_dl, y1_dl, x2_dl, y2_dl, angle_dl, length_dl = dl_line
                        center_dl_x = (x1_dl + x2_dl) / 2
                        center_dl_y = (y1_dl + y2_dl) / 2

                        # Calculate Euclidean distance between line centers
                        distance = np.sqrt((center_dl_x - center_dr_x)**2 + (center_dl_y - center_dr_y)**2)

                        if distance <= 50:
                            # DIRECTION VALIDATION: Verify arrow actually points DOWN
                            # For DOWN arrow (v shape):
                            # - Lines should be CLOSER together at TOP (smaller Y, apex)
                            # - Lines should SPREAD apart at BOTTOM (larger Y)

                            # Get X coordinates at top and bottom of each line
                            # REVERSED: smaller Y (higher on screen) is actually the BOTTOM (spread point)
                            # larger Y (lower on screen) is actually the TOP (apex)
                            if y1_dr < y2_dr:  # y1 is higher on screen (smaller Y) = bottom endpoint
                                dr_x_top, dr_y_top = x2_dr, y2_dr      # Apex is at y2 (larger Y)
                                dr_x_bottom, dr_y_bottom = x1_dr, y1_dr  # Spread point at y1 (smaller Y)
                            else:  # y2 is higher on screen (smaller Y) = bottom endpoint
                                dr_x_top, dr_y_top = x1_dr, y1_dr      # Apex is at y1 (larger Y)
                                dr_x_bottom, dr_y_bottom = x2_dr, y2_dr  # Spread point at y2 (smaller Y)

                            if y1_dl < y2_dl:  # y1 is higher on screen (smaller Y) = bottom endpoint
                                dl_x_top, dl_y_top = x2_dl, y2_dl      # Apex is at y2 (larger Y)
                                dl_x_bottom, dl_y_bottom = x1_dl, y1_dl  # Spread point at y1 (smaller Y)
                            else:  # y2 is higher on screen (smaller Y) = bottom endpoint
                                dl_x_top, dl_y_top = x1_dl, y1_dl      # Apex is at y1 (larger Y)
                                dl_x_bottom, dl_y_bottom = x2_dl, y2_dl  # Spread point at y2 (smaller Y)

                            # Calculate X-distance at top vs bottom
                            x_dist_top = abs(dr_x_top - dl_x_top)
                            x_dist_bottom = abs(dr_x_bottom - dl_x_bottom)

                            # For DOWN arrow: lines should SPREAD (x_dist_bottom > x_dist_top)
                            # For LEFT/RIGHT arrow: lines would be parallel or converge sideways
                            is_pointing_down = x_dist_bottom > x_dist_top

                            if is_pointing_down:
                                # Valid arrow pair found
                                arrow_detected = True
                                has_down_arrow = True
                                print(f"      [DOWN ARROW PAIR FORMED]")
                                print(f"        DR Line: coords=({x1_dr},{y1_dr})-({x2_dr},{y2_dr}), angle={angle_dr:.1f}°, length={length_dr:.1f}px, color=GREEN")
                                print(f"        DL Line: coords=({x1_dl},{y1_dl})-({x2_dl},{y2_dl}), angle={angle_dl:.1f}°, length={length_dl:.1f}px, color=BLUE")

                                # Calculate distances between the paired lines
                                horiz_distance = abs(center_dr_x - center_dl_x)
                                vert_distance = abs(center_dr_y - center_dl_y)
                                print(f"        Distance between arrow legs: horizontal={horiz_distance:.1f}px, vertical={vert_distance:.1f}px, euclidean={distance:.1f}px")
                                print(f"        Direction validation: x_dist_top={x_dist_top:.1f}px < x_dist_bottom={x_dist_bottom:.1f}px (lines SPREAD downward)")
                                print(f"      FOUND ARROW: {len(down_right_lines)} DR lines, {len(down_left_lines)} DL lines")
                                break  # Found valid arrow, stop searching
                            else:
                                print(f"      [DOWN ARROW PAIR REJECTED] Not pointing downward - lines converge at bottom (wrong direction)")
                                print(f"        DR Line: coords=({x1_dr},{y1_dr})-({x2_dr},{y2_dr}), angle={angle_dr:.1f}°")
                                print(f"        DL Line: coords=({x1_dl},{y1_dl})-({x2_dl},{y2_dl}), angle={angle_dl:.1f}°")
                                print(f"        Direction check: x_dist_top={x_dist_top:.1f}px >= x_dist_bottom={x_dist_bottom:.1f}px (NOT spreading downward)")

                    if arrow_detected:
                        break  # Found valid arrow, stop outer loop

                if not arrow_detected:
                    # Lines exist but are too far apart
                    print(f"      [DOWN ARROW REJECTED] Lines too far apart")
                    for dr_idx, dr_line in enumerate(down_right_lines):
                        x1, y1, x2, y2, angle, length = dr_line
                        print(f"        DR Line {dr_idx+1}: coords=({x1},{y1})-({x2},{y2}), angle={angle:.1f}°, length={length:.1f}px, color=GREEN")
                    for dl_idx, dl_line in enumerate(down_left_lines):
                        x1, y1, x2, y2, angle, length = dl_line
                        print(f"        DL Line {dl_idx+1}: coords=({x1},{y1})-({x2},{y2}), angle={angle:.1f}°, length={length:.1f}px, color=BLUE")

                    # Show why each pair was rejected
                    for dr_line in down_right_lines:
                        x1_dr, y1_dr, x2_dr, y2_dr, angle_dr, length_dr = dr_line
                        center_dr_x = (x1_dr + x2_dr) / 2
                        center_dr_y = (y1_dr + y2_dr) / 2

                        for dl_line in down_left_lines:
                            x1_dl, y1_dl, x2_dl, y2_dl, angle_dl, length_dl = dl_line
                            center_dl_x = (x1_dl + x2_dl) / 2
                            center_dl_y = (y1_dl + y2_dl) / 2

                            distance = np.sqrt((center_dl_x - center_dr_x)**2 + (center_dl_y - center_dr_y)**2)
                            print(f"        Pair distance: {distance:.1f}px > 50px threshold")

            # SINGLE LEG DETECTION: Check if only one leg exists
            elif (len(down_right_lines) > 0 and len(down_left_lines) == 0) or (len(down_left_lines) > 0 and len(down_right_lines) == 0):
                print(f"    [SINGLE LEG DETECTED] Found {len(down_right_lines)} down-right and {len(down_left_lines)} down-left lines")

                # Get measurement ID for descriptive filename
                meas_id = height.get('id', '')
                if not meas_id:
                    # Try to find measurement ID from all_measurements
                    for i, meas in enumerate(all_measurements):
                        if (abs(meas.get('x', 0) - height_x) < 5 and
                            abs(meas.get('y', 0) - height_y) < 5 and
                            meas.get('text') == height.get('text')):
                            meas_id = meas.get('id', i+1)
                            break

                # Format measurement ID as "M{id}"
                # Use index+1 as fallback if no ID found
                if not meas_id:
                    meas_id = heights.index(height) + 1
                measurement_id_str = f"M{meas_id}"
                measurement_value_str = height.get('text', '')

                # Verify with Claude vision API
                # Pass colored ROI for edge enhancement visualization
                claude_result = verify_arrow_with_claude(
                    roi,
                    'down',
                    image_path=image_path,
                    measurement_id=measurement_id_str,
                    measurement_value=measurement_value_str
                )

                if claude_result:
                    print(f"    [CLAUDE CONFIRMED] Arrow detected by Claude vision API")
                    has_down_arrow = True
                else:
                    print(f"    [CLAUDE REJECTED] No arrow confirmed by Claude vision API")

        print(f"    Down arrow detected: {has_down_arrow}")

        # Create debug visualization showing the scan region and detected arrows
        if image_dir and base_name_clean:
            # Create visualization on copy of ROI
            roi_viz = roi.copy()

            # ALWAYS draw lines found (even if arrow detection failed)
            # Color code: GREEN for down-right lines, BLUE for down-left lines
            if 'down_right_lines' in locals() and 'down_left_lines' in locals():
                # Draw down-right lines in GREEN
                for x1, y1, x2, y2, angle, length in down_right_lines:
                    cv2.line(roi_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw down-left lines in BLUE
                for x1, y1, x2, y2, angle, length in down_left_lines:
                    cv2.line(roi_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # If arrow was detected, circle the center
                if has_down_arrow:
                    # Calculate arrow center in ROI coordinates for visualization
                    all_y_coords_viz = []
                    all_x_coords_viz = []
                    for x1, y1, x2, y2, angle, length in down_right_lines:
                        all_y_coords_viz.extend([y1, y2])
                        all_x_coords_viz.extend([x1, x2])
                    for x1, y1, x2, y2, angle, length in down_left_lines:
                        all_y_coords_viz.extend([y1, y2])
                        all_x_coords_viz.extend([x1, x2])

                    if all_y_coords_viz and all_x_coords_viz:
                        center_y_viz = int(sum(all_y_coords_viz) / len(all_y_coords_viz))
                        center_x_viz = int(sum(all_x_coords_viz) / len(all_x_coords_viz))
                        # Circle the arrow center in YELLOW/CYAN
                        cv2.circle(roi_viz, (center_x_viz, center_y_viz), 20, (0, 255, 255), 3)

            # Draw scan region boundary
            cv2.rectangle(roi_viz, (0, 0), (roi_viz.shape[1]-1, roi_viz.shape[0]-1), (255, 0, 0), 2)

            # Add text indicating detection result
            if has_down_arrow:
                result_text = "ARROW FOUND"
                result_color = (0, 255, 0)
            else:
                # Show line counts when arrow NOT detected
                dr_count = len(down_right_lines) if 'down_right_lines' in locals() else 0
                dl_count = len(down_left_lines) if 'down_left_lines' in locals() else 0
                result_text = f"NO ARROW - DR:{dr_count} DL:{dl_count}"
                result_color = (0, 0, 255)
            cv2.putText(roi_viz, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, result_color, 2)

            # Save debug image with measurement info in filename
            # Format: page-X_MX_[text]_down_arrow_scan.png
            # Get measurement ID if available
            meas_id = height.get('id', '')
            if not meas_id:
                # Try to find measurement ID from all_measurements
                for i, meas in enumerate(all_measurements):
                    if (abs(meas.get('position', (0, 0))[0] - height_x) < 5 and
                        abs(meas.get('position', (0, 0))[1] - height_y) < 5 and
                        meas.get('text') == height.get('text')):
                        meas_id = meas.get('id', i+1)
                        break

            text_safe = height.get('text', '').replace('/', '-').replace(' ', '')[:20]
            debug_filename = f"{base_name_clean}_M{meas_id}_{text_safe}_down_arrow_scan.png"
            debug_path = os.path.join(image_dir, debug_filename)
            cv2.imwrite(debug_path, roi_viz)
            print(f"    [SAVED] Down arrow scan visualization: {debug_filename}")

        if has_down_arrow:
            # Found a down arrow - calculate visual center from detected lines
            # Visual center is the midpoint between top and bottom of arrow
            all_y_coords = []
            all_x_coords = []

            # Collect Y coordinates from all detected arrow lines
            for x1, y1, x2, y2, angle, length in down_right_lines:
                all_y_coords.extend([y1, y2])
                all_x_coords.extend([x1, x2])
            for x1, y1, x2, y2, angle, length in down_left_lines:
                all_y_coords.extend([y1, y2])
                all_x_coords.extend([x1, x2])

            # Visual center: midpoint between top and bottom of arrow in ROI coordinates
            min_y_roi = min(all_y_coords)
            max_y_roi = max(all_y_coords)
            center_y_roi = (min_y_roi + max_y_roi) / 2
            center_x_roi = sum(all_x_coords) / len(all_x_coords)  # Average X position

            # Convert from ROI coordinates to image coordinates
            arrow_y = int(roi_y1 + center_y_roi)
            arrow_x = int(roi_x1 + center_x_roi)

            down_arrow_positions.append((arrow_x, arrow_y))
            print(f"    Added down arrow at ({arrow_x}, {arrow_y}) [visual center from detected lines]")

    return down_arrow_positions


def is_height_above_width_with_angle(height_pos, width_pos, hline_angle_degrees):
    """
    Determine if height is "above" a width, accounting for the width's H-line angle.

    "Above" means traveling in the perpendicular direction from the width's dimension line.
    Uses perpendicular projection to determine if the height is on the "above" side.

    Args:
        height_pos: (x, y) tuple for height position
        width_pos: (x, y) tuple for width position
        hline_angle_degrees: Angle of width's H-line in degrees (-180 to 180)

    Returns:
        True if height is above the width's dimension line, False otherwise
    """
    import math

    height_x, height_y = height_pos
    width_x, width_y = width_pos

    # Convert angle to radians
    angle_rad = math.radians(hline_angle_degrees)

    # Width dimension line direction vector
    width_dir_x = math.cos(angle_rad)
    width_dir_y = math.sin(angle_rad)

    # Perpendicular direction (rotate 90° counterclockwise)
    perp_x = -width_dir_y
    perp_y = width_dir_x

    # Ensure perpendicular points upward (toward smaller Y)
    if perp_y > 0:  # Points downward (larger Y)
        perp_x = -perp_x
        perp_y = -perp_y

    # Vector from width to height
    to_height_x = height_x - width_x
    to_height_y = height_y - width_y

    # Dot product: if negative, height is in the perpendicular direction (above)
    projection = to_height_x * perp_x + to_height_y * perp_y

    return projection < 0  # Negative projection means "above"


def pair_measurements_by_proximity(classified_measurements, all_measurements, image=None, image_path=None, non_measurement_text_exclusions=None, x2_multiplier=1):
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
                    'is_finished_size': meas.get('is_finished_size', False),
                    'id': meas.get('id', i+1)
                }
                # Capture h_lines if available
                if 'h_lines' in meas:
                    width_data['h_lines'] = meas['h_lines']
                # Capture width extent if available
                if 'width_extent' in meas:
                    width_data['width_extent'] = meas['width_extent']
                widths.append(width_data)
            elif category in ['height', 'unclassified']:
                height_data = {
                    'text': meas['text'],
                    'x': meas['position'][0],
                    'y': meas['position'][1],
                    'bounds': meas.get('bounds', None),
                    'v_line_angle': meas.get('v_line_angle', 90),
                    'is_finished_size': meas.get('is_finished_size', False),
                    'id': meas.get('id', i+1)
                }
                if 'notation' in meas:
                    height_data['notation'] = meas['notation']
                # Capture height extent if available
                if 'height_extent' in meas:
                    height_data['height_extent'] = meas['height_extent']
                heights.append(height_data)

    if not widths or not heights:
        print("  Cannot pair - need both widths and heights")
        return [], [], [], None

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
        down_arrow_y_positions = find_down_arrows_near_heights(image, heights, all_measurements, non_measurement_text_exclusions, image_path)

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

            Y_DELTA_THRESHOLD = 350  # Max Y difference between left and right arrows

            # Initialize variables
            leftmost_lowest_arrow = None
            rightmost_lowest_arrow = None
            arrows_rejected = False

            # Case 1: Both left and right arrows exist
            if len(left_arrows) >= 1 and len(right_arrows) >= 1:
                leftmost_lowest_arrow = max(left_arrows, key=lambda pos: pos[1])
                rightmost_lowest_arrow = max(right_arrows, key=lambda pos: pos[1])

                # Validate Y positions are similar
                y_delta = abs(leftmost_lowest_arrow[1] - rightmost_lowest_arrow[1])

                if y_delta > Y_DELTA_THRESHOLD:
                    print(f"    WARNING: Y-delta between arrows ({y_delta}px) exceeds threshold ({Y_DELTA_THRESHOLD}px)")
                    print(f"    Skipping bottom width classification - arrows not at same horizontal level")
                    print(f"    Drawing reference line anyway (for visualization)")
                    arrows_rejected = True
                    # Still create bottom_width_line for visualization (continue to line creation below)
                else:
                    print(f"    Y-delta between arrows: {y_delta}px (within {Y_DELTA_THRESHOLD}px threshold)")
                    print(f"    Lowest arrow on LEFT side: ({leftmost_lowest_arrow[0]}, {leftmost_lowest_arrow[1]})")
                    print(f"    Lowest arrow on RIGHT side: ({rightmost_lowest_arrow[0]}, {rightmost_lowest_arrow[1]})")

            # Case 2: Only left arrows exist
            elif len(left_arrows) >= 1 and len(right_arrows) == 0:
                leftmost_lowest_arrow = max(left_arrows, key=lambda pos: pos[1])
                y_pos = leftmost_lowest_arrow[1]
                # Create horizontal line across full image width
                rightmost_lowest_arrow = (image.shape[1], y_pos)
                leftmost_lowest_arrow = (0, y_pos)
                print(f"    Only LEFT arrows found - using horizontal line at Y={y_pos}")
                print(f"    Lowest arrow on LEFT side: ({leftmost_lowest_arrow[0]}, {leftmost_lowest_arrow[1]})")
                print(f"    Lowest arrow on RIGHT side: ({rightmost_lowest_arrow[0]}, {rightmost_lowest_arrow[1]})")

            # Case 3: Only right arrows exist
            elif len(left_arrows) == 0 and len(right_arrows) >= 1:
                rightmost_lowest_arrow = max(right_arrows, key=lambda pos: pos[1])
                y_pos = rightmost_lowest_arrow[1]
                # Create horizontal line across full image width
                leftmost_lowest_arrow = (0, y_pos)
                rightmost_lowest_arrow = (image.shape[1], y_pos)
                print(f"    Only RIGHT arrows found - using horizontal line at Y={y_pos}")
                print(f"    Lowest arrow on LEFT side: ({leftmost_lowest_arrow[0]}, {leftmost_lowest_arrow[1]})")
                print(f"    Lowest arrow on RIGHT side: ({rightmost_lowest_arrow[0]}, {rightmost_lowest_arrow[1]})")

            # Case 4: No arrows at all
            else:
                print(f"    No arrows found on either side")
                bottom_width_line = None

            # Store line for visualization (always create if arrows exist)
            if leftmost_lowest_arrow is not None and rightmost_lowest_arrow is not None:
                # Create bottom_width_line for visualization even if arrows_rejected=True
                # This allows visualization to draw the reference line regardless of Y-delta threshold
                bottom_width_line = {
                    'start': leftmost_lowest_arrow,
                    'end': rightmost_lowest_arrow
                }

            # Only proceed with bottom width classification if we have a valid line AND arrows not rejected
            if bottom_width_line is not None and not arrows_rejected:
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
                perpendicular_tolerance = 100

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
        # Check if this is a bottom width
        is_bottom_width = width.get('position_class') == 'bottom'

        # Only skip BOTTOM widths without extent
        # Non-bottom widths don't need extent for pairing
        if is_bottom_width and 'width_extent' not in width:
            print(f"  Skipping {width['text']} - no extent available (bottom width)")
            continue

        # Get extent info if available (bottom widths will have it, non-bottom might not)
        extent = width.get('width_extent', None)
        if extent:
            width_left = extent['left']
            width_right = extent['right']
        else:
            width_left = width['x']
            width_right = width['x']

        drawer_config = width.get('drawer_config', '')

        # Check if this is a bottom width with multiple drawers
        is_bottom_multiple = ('bottom width' in drawer_config and 'multiple drawers' in drawer_config)

        print(f"\n  Pairing {width['text']} at ({width['x']:.0f}, {width['y']:.0f})")
        print(f"    Width extent: {width_left:.0f} to {width_right:.0f}")
        print(f"    Drawer config: {drawer_config}")
        print(f"    Is bottom width with multiple drawers: {is_bottom_multiple}")

        # Pairing logic based on width type
        if is_bottom_width:
            # BOTTOM WIDTH: Calculate scan area and find candidate heights using extent
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
                        # Check if there's another width between this bottom width and the height
                        has_blocking_width = False
                        for other_width in widths:
                            # Skip current width
                            if other_width.get('x') == width['x'] and other_width.get('y') == width['y']:
                                continue

                            other_x = other_width['x']
                            other_y = other_width['y']

                            # Check if other width is in X-range and between current width and height
                            if scan_left <= other_x <= scan_right:
                                print(f"        DEBUG: Checking width '{other_width['text']}' at ({other_x:.0f}, {other_y:.0f})")
                                print(f"        DEBUG: height_y={height['y']:.0f}, other_y={other_y:.0f}, width_y={width['y']:.0f}")
                                print(f"        DEBUG: {height['y']:.0f} < {other_y:.0f} < {width['y']:.0f} = {height['y'] < other_y < width['y']}")
                                if height['y'] < other_y < width['y']:
                                    has_blocking_width = True
                                    print(f"        BLOCKED: Width '{other_width['text']}' at ({other_x:.0f}, {other_y:.0f}) blocks this height")
                                    break

                        if not has_blocking_width:
                            candidate_heights.append(height)
                            print(f"      Candidate height: {height['text']} at ({height['x']:.0f}, {height['y']:.0f})")

            print(f"    Found {len(candidate_heights)} candidate heights in X range")

            # Apply pairing rules based on drawer configuration
            if is_bottom_multiple:
                # Multiple drawers: MUST have 2+ heights
                if len(candidate_heights) >= 2:
                    print(f"    Multiple heights found ({len(candidate_heights)}) - this width belongs to the drawer bank, will pair with ALL")
                else:
                    # Not enough for multiple drawers - search left width's extent
                    print(f"    Insufficient heights ({len(candidate_heights)}) for multiple drawers - searching left width's extent")

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

                    # Re-check if we have enough heights for multiple drawers
                    if len(candidate_heights) < 2:
                        print(f"    SKIP: Still insufficient heights ({len(candidate_heights)}) for multiple drawers after left search")
                        continue
            else:
                # 1 drawer: Allow pairing with 1+ height
                if len(candidate_heights) >= 1:
                    print(f"    Found {len(candidate_heights)} height(s) for 1 drawer configuration")
                else:
                    # No heights in own extent - try left search
                    print(f"    No heights in own extent (1 drawer) - searching left width's extent")

                    candidate_heights = []

                    # Find the bottom width to the left of current width
                    left_width = None
                    for w in widths:
                        if (w.get('position_class') == 'bottom' and
                            w['x'] < width['x'] and
                            'width_extent' in w):
                            if left_width is None or w['x'] > left_width['x']:
                                left_width = w

                    if left_width:
                        print(f"    Found left width: {left_width['text']} at ({left_width['x']:.0f}, {left_width['y']:.0f})")

                        left_width_id = id(left_width)
                        if left_width_id in width_to_heights:
                            candidate_heights = width_to_heights[left_width_id][:]
                            print(f"    Using {len(candidate_heights)} height(s) from left width's pairings:")
                            for h in candidate_heights:
                                print(f"      {h['text']} at ({h['x']:.0f}, {h['y']:.0f})")
                        else:
                            print(f"    Left width has not paired yet - cannot use its heights")

                    # Re-check: For 1 drawer, need at least 1 height
                    if len(candidate_heights) < 1:
                        print(f"    SKIP: No heights found after left search")
                        continue

            # BOTTOM WIDTH PAIRING: Pair with ALL candidate heights
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
                        'width_angle': extent.get('angle', 0),
                        'width_hline_angle': extent.get('hline_angle', extent.get('angle', 0)),
                        'width_is_finished': width.get('is_finished_size', False),
                        'height_is_finished': height.get('is_finished_size', False),
                        'quantity': x2_multiplier
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
            # NON-BOTTOM WIDTHS: pair with closest height above using three-tier priority
            # Priority: 1) center (directly above), 2) left, 3) right
            print(f"    Looking for single closest height above this width")

            # Find heights above this width by projecting dimension line at h_line_angle
            # Line from width: y = width_y + (x - width_x) * tan(h_line_angle)
            # Height is above if height_y < line_y at height_x
            import math
            hline_angle = width.get('h_line_angle', 0)
            print(f"    Using H-line angle: {hline_angle:.1f}°")

            heights_above = []

            # Split h-lines by ROI side (left vs right)
            left_h_lines = []
            right_h_lines = []
            if 'h_lines' in width and width['h_lines']:
                for line in width['h_lines']:
                    if line.get('roi_side') == 'left':
                        left_h_lines.append(line)
                    elif line.get('roi_side') == 'right':
                        right_h_lines.append(line)

            # Find furthest-down h-line in each group
            # Filter for nearly-horizontal lines first to exclude arrow lines
            left_line = None
            right_line = None
            horizontal_threshold = 8.0  # degrees - same as classification
            if left_h_lines:
                nearly_horizontal_left = [l for l in left_h_lines if abs(l.get('angle', 0)) <= horizontal_threshold]
                if nearly_horizontal_left:
                    left_line = max(nearly_horizontal_left, key=lambda l: (l['coords'][1] + l['coords'][3]) / 2)
                else:
                    # Fallback: use closest to horizontal
                    left_line = min(left_h_lines, key=lambda l: abs(l.get('angle', 0)))
            if right_h_lines:
                nearly_horizontal_right = [l for l in right_h_lines if abs(l.get('angle', 0)) <= horizontal_threshold]
                if nearly_horizontal_right:
                    right_line = max(nearly_horizontal_right, key=lambda l: (l['coords'][1] + l['coords'][3]) / 2)
                else:
                    # Fallback: use closest to horizontal
                    right_line = min(right_h_lines, key=lambda l: abs(l.get('angle', 0)))

            print(f"    Left H-ROI h-lines: {len(left_h_lines)}, Right H-ROI h-lines: {len(right_h_lines)}")
            if left_line:
                print(f"    Left extent line: ({left_line['coords'][0]:.0f}, {left_line['coords'][1]:.0f}) to ({left_line['coords'][2]:.0f}, {left_line['coords'][3]:.0f})")
            if right_line:
                print(f"    Right extent line: ({right_line['coords'][0]:.0f}, {right_line['coords'][1]:.0f}) to ({right_line['coords'][2]:.0f}, {right_line['coords'][3]:.0f})")

            for h in heights:
                # Determine which extent line to use based on height's X position relative to width center
                if h['x'] < width['x']:
                    # Left zone: use LEFT H-ROI's furthest-down h-line
                    if left_line:
                        x1, y1, x2, y2 = left_line['coords']
                        # Use leftmost point of left line to project leftward
                        if x1 <= x2:
                            ref_x, ref_y = x1, y1
                        else:
                            ref_x, ref_y = x2, y2

                        # Calculate angle from this line
                        if x2 != x1:
                            line_angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
                        else:
                            line_angle = 90

                        # Calculate Y on dimension line at this height's X position
                        dx = h['x'] - ref_x
                        dy = dx * math.tan(math.radians(line_angle))
                        line_y_at_height_x = ref_y + dy
                    else:
                        # Fallback to center projection if no left h-line
                        line_y_at_height_x = width['y'] + (h['x'] - width['x']) * math.tan(math.radians(hline_angle))
                else:
                    # Right zone: use RIGHT H-ROI's furthest-down h-line
                    if right_line:
                        x1, y1, x2, y2 = right_line['coords']
                        # Use rightmost point of right line to project rightward
                        if x1 >= x2:
                            ref_x, ref_y = x1, y1
                        else:
                            ref_x, ref_y = x2, y2

                        # Calculate angle from this line
                        if x2 != x1:
                            line_angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
                        else:
                            line_angle = 90

                        # Calculate Y on dimension line at this height's X position
                        dx = h['x'] - ref_x
                        dy = dx * math.tan(math.radians(line_angle))
                        line_y_at_height_x = ref_y + dy
                    else:
                        # Fallback to center projection if no right h-line
                        line_y_at_height_x = width['y'] + (h['x'] - width['x']) * math.tan(math.radians(hline_angle))

                is_above = h['y'] < line_y_at_height_x
                if is_above:
                    heights_above.append(h)
                    print(f"      Height {h['text']} at ({h['x']:.0f}, {h['y']:.0f}): line_y={line_y_at_height_x:.1f}, is_above={is_above}")

            if not heights_above:
                print(f"    No heights found above this width - SKIP")
                continue

            # Calculate center zone using actual text bounds
            # Use text left and right edges instead of center ± tolerance
            width_bounds = width.get('bounds')
            if width_bounds and 'left' in width_bounds and 'right' in width_bounds:
                left_bound = width_bounds['left']
                right_bound = width_bounds['right']
                text_width = right_bound - left_bound
            else:
                # Fallback if bounds not available
                left_bound = width['x'] - 25
                right_bound = width['x'] + 25
                text_width = 50

            print(f"    Center zone using text bounds: [{left_bound:.0f}, {right_bound:.0f}] (text_width={text_width:.0f}px)")

            # THREE-TIER PRIORITY:
            # 1. Heights in center range (within text bounds) - NO height value limit
            # 2. Heights to the left (if no center heights found) - ONLY heights <= 8 inches
            # 3. Heights to the right (if no center or left heights found) - ONLY heights <= 8 inches

            # Maximum height value (in inches) allowed for left/right pairing
            # This prevents tall openings from pairing with far-left or far-right widths
            MAX_HEIGHT_FOR_SIDE_PAIRING = 8.0  # inches

            heights_center = [h for h in heights_above if left_bound <= h['x'] <= right_bound]

            # Filter left/right heights by height value (must be <= 8 inches)
            # First, separate by position (left/right)
            heights_left_all = [h for h in heights_above if h['x'] < left_bound]
            heights_right_all = [h for h in heights_above if h['x'] > right_bound]

            # Import fraction_to_decimal for height value checking
            from shared_utils import fraction_to_decimal

            # Then filter by height value constraint
            heights_left = [h for h in heights_left_all
                           if fraction_to_decimal(h['text']) <= MAX_HEIGHT_FOR_SIDE_PAIRING]
            heights_right = [h for h in heights_right_all
                            if fraction_to_decimal(h['text']) <= MAX_HEIGHT_FOR_SIDE_PAIRING]

            # Log filtered heights for debugging
            if len(heights_left_all) != len(heights_left):
                filtered_left = [h for h in heights_left_all if h not in heights_left]
                for h in filtered_left:
                    print(f"      Filtered out LEFT height {h['text']} ({fraction_to_decimal(h['text']):.2f}\") - exceeds {MAX_HEIGHT_FOR_SIDE_PAIRING}\" limit")
            if len(heights_right_all) != len(heights_right):
                filtered_right = [h for h in heights_right_all if h not in heights_right]
                for h in filtered_right:
                    print(f"      Filtered out RIGHT height {h['text']} ({fraction_to_decimal(h['text']):.2f}\") - exceeds {MAX_HEIGHT_FOR_SIDE_PAIRING}\" limit")

            closest_height = None

            if heights_center:
                # Find closest in center range (by distance)
                closest_height = min(heights_center, key=lambda h:
                    ((width['x'] - h['x'])**2 + (width['y'] - h['y'])**2)**0.5)
                print(f"    Found height in center range: {closest_height['text']} at ({closest_height['x']:.0f}, {closest_height['y']:.0f})")
            elif heights_left:
                # No heights in center, find closest to the left (by distance)
                closest_height = min(heights_left, key=lambda h:
                    ((width['x'] - h['x'])**2 + (width['y'] - h['y'])**2)**0.5)
                height_value = fraction_to_decimal(closest_height['text'])
                print(f"    No heights in center, found height to the left: {closest_height['text']} ({height_value:.2f}\") at ({closest_height['x']:.0f}, {closest_height['y']:.0f})")
            elif heights_right:
                # No heights in center or left, find closest to the right (by distance)
                closest_height = min(heights_right, key=lambda h:
                    ((width['x'] - h['x'])**2 + (width['y'] - h['y'])**2)**0.5)
                height_value = fraction_to_decimal(closest_height['text'])
                print(f"    No heights in center or left, found height to the right: {closest_height['text']} ({height_value:.2f}\") at ({closest_height['x']:.0f}, {closest_height['y']:.0f})")

            if closest_height:
                opening = {
                    'width': width['text'],
                    'height': closest_height['text'],
                    'width_pos': (width['x'], width['y']),
                    'height_pos': (closest_height['x'], closest_height['y']),
                    'distance': ((width['x'] - closest_height['x'])**2 + (width['y'] - closest_height['y'])**2)**0.5,
                    'width_angle': extent.get('angle', 0) if extent else 0,
                    'width_hline_angle': width.get('h_line_angle', 0),
                    'width_is_finished': width.get('is_finished_size', False),
                    'height_is_finished': closest_height.get('is_finished_size', False),
                    'quantity': x2_multiplier
                }

                if 'notation' in closest_height:
                    opening['notation'] = closest_height['notation']

                openings.append(opening)
                paired_heights.add(id(closest_height))
                print(f"    PAIRED: {width['text']} x {closest_height['text']}")
            else:
                print(f"    No suitable height found - SKIP")

    # Pair any unpaired heights with widths at least 10px above them
    # This prevents side-by-side measurements from incorrectly pairing
    print(f"\n=== PAIRING UNPAIRED HEIGHTS WITH WIDTHS ABOVE ===")
    unpaired_heights = [h for h in heights if id(h) not in paired_heights]

    if unpaired_heights:
        print(f"  Found {len(unpaired_heights)} unpaired heights:")
        for h in unpaired_heights:
            print(f"    {h['text']} at ({h['x']:.0f}, {h['y']:.0f})")

    for height in unpaired_heights:
        print(f"\n  Looking for width above unpaired height: {height['text']} at ({height['x']:.0f}, {height['y']:.0f})")

        # Find all widths above this height (at least 10px above to avoid side-by-side pairing)
        # Width must be at least 10 pixels ABOVE the height (w_y < h_y - 10)
        min_vertical_distance = 10  # pixels
        widths_above = [w for w in widths if w['y'] < (height['y'] - min_vertical_distance)]

        if not widths_above:
            print(f"    No widths found at least {min_vertical_distance}px above this height - trying search BELOW")
            # Fallback: search for widths BELOW this height
            widths_below = [w for w in widths if w['y'] > height['y']]

            if not widths_below:
                print(f"    No widths found below either - SKIP")
                continue

            # Use widths_below for the 3-zone search
            widths_above = widths_below
            print(f"    Found {len(widths_below)} widths below, will search these")

        # Use text bounds with 75px padding on each side
        edge_padding = 75
        height_bounds = height.get('bounds', {})
        if height_bounds and 'left' in height_bounds and 'right' in height_bounds:
            zone_left = height_bounds['left'] - edge_padding
            zone_right = height_bounds['right'] + edge_padding
            print(f"    Center zone using height text bounds: [{zone_left:.0f}, {zone_right:.0f}] with {edge_padding}px padding")
        else:
            # Fallback to center ± 100px if no bounds
            zone_left = height['x'] - 100
            zone_right = height['x'] + 100
            print(f"    Center zone using fallback (no bounds): [{zone_left:.0f}, {zone_right:.0f}]")

        # First: Look for width in center zone (based on height's text bounds)
        widths_center = [w for w in widths_above
                        if zone_left <= w['x'] <= zone_right]

        # Second: Look for widths to the left
        widths_left = [w for w in widths_above if w['x'] < zone_left]

        # Third: Look for widths to the right
        widths_right = [w for w in widths_above if w['x'] > zone_right]

        closest_width = None

        if widths_center:
            # Find closest in center range (by Y-distance only)
            closest_width = min(widths_center, key=lambda w: abs(height['y'] - w['y']))
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
            # Get width angles from extent if available
            width_angle = 0
            width_hline_angle = 0
            if 'width_extent' in closest_width:
                width_angle = closest_width['width_extent'].get('angle', 0)
                width_hline_angle = closest_width['width_extent'].get('hline_angle', width_angle)

            opening = {
                'width': closest_width['text'],
                'height': height['text'],
                'width_pos': (closest_width['x'], closest_width['y']),
                'height_pos': (height['x'], height['y']),
                'distance': ((closest_width['x'] - height['x'])**2 + (closest_width['y'] - height['y'])**2)**0.5,
                'width_angle': width_angle,
                'width_hline_angle': width_hline_angle,
                'width_is_finished': closest_width.get('is_finished_size', False),
                'height_is_finished': height.get('is_finished_size', False),
                'quantity': x2_multiplier
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
