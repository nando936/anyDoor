#!/usr/bin/env python3
"""
Line detection and measurement classification.
Detects horizontal/vertical dimension lines near measurements and classifies them.
"""

import cv2
import numpy as np
import re
from measurement_config import HSV_CONFIG


def extract_rotated_roi(image, x1, y1, x2, y2, angle_degrees, center_point):
    """Extract pixels within a rotated rectangle from the image"""
    # Calculate the four corners of the rectangle
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)

    # Get center point for rotation
    cx, cy = center_point

    # Rotate corners around center point
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rotated_corners = []
    for px, py in corners:
        # Translate to origin
        px_rel = px - cx
        py_rel = py - cy
        # Rotate
        px_rot = px_rel * cos_a - py_rel * sin_a
        py_rot = px_rel * sin_a + py_rel * cos_a
        # Translate back
        rotated_corners.append([px_rot + cx, py_rot + cy])

    rotated_corners = np.array(rotated_corners, dtype=np.float32)

    # Get the bounding box of the rotated rectangle
    min_x = int(np.floor(rotated_corners[:, 0].min()))
    max_x = int(np.ceil(rotated_corners[:, 0].max()))
    min_y = int(np.floor(rotated_corners[:, 1].min()))
    max_y = int(np.ceil(rotated_corners[:, 1].max()))

    # Clamp to image bounds
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image.shape[1], max_x)
    max_y = min(image.shape[0], max_y)

    # Calculate the dimensions for the output ROI
    width = x2 - x1
    height = y2 - y1

    # Define destination points (unrotated rectangle of same size)
    dst_corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(rotated_corners, dst_corners)

    # Apply perspective transform to get the rotated region
    roi = cv2.warpPerspective(image, M, (width, height))

    return roi


def detect_lines_in_roi(roi_image, roi_x_offset, roi_y_offset):
    """Detect lines in an ROI and return coordinates adjusted to full image"""
    if roi_image.size == 0:
        return []

    # Apply HSV filter to detect only green dimension lines (not cabinet edges)
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array(HSV_CONFIG['lower_green'])
    upper_green = np.array(HSV_CONFIG['upper_green'])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to connect nearby pixels
    kernel = np.ones((3,3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Detect edges only on the green-filtered image
    edges = cv2.Canny(green_mask, 30, 100)

    # Find lines with HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=15,
                            minLineLength=20,
                            maxLineGap=30)

    # Convert line coordinates back to full image coordinates
    if lines is not None:
        adjusted_lines = []
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            adjusted_lines.append([lx1 + roi_x_offset, ly1 + roi_y_offset,
                                  lx2 + roi_x_offset, ly2 + roi_y_offset])
        return adjusted_lines
    return []


def find_lines_near_measurement(image, measurement, save_roi_debug=False):
    """Find lines near a specific measurement position that match the text color"""
    x = int(measurement['position'][0])
    y = int(measurement['position'][1])

    # Enable debug for troubleshooting
    DEBUG_MODE = True  # Enable to see width calculation and H-ROI positioning details

    # Use bounds if available
    if 'bounds' in measurement:
        bounds = measurement['bounds']
        text_left = int(bounds['left'])
        text_right = int(bounds['right'])
        text_top = int(bounds['top'])
        text_bottom = int(bounds['bottom'])

        text_width = text_right - text_left
        text_height = text_bottom - text_top

        # Update x,y to be the actual center of the text bounds
        x = int((text_left + text_right) / 2)
        y = int((text_top + text_bottom) / 2)
    else:
        # Fallback to estimates
        text_height = 30
        text_width = len(measurement.get('text', '')) * 15

    # Add padding for image skew or misalignment
    padding = 10
    text_height_with_padding = text_height + padding
    text_width_with_padding = text_width + padding

    # Create ROIs for horizontal and vertical line detection
    # For horizontal lines: Look LEFT and RIGHT of the text
    # Use 1.0x text width but clamp to 60 pixels (5/8 inch at 96 DPI)
    h_strip_extension = min(60, max(60, int(text_width * 1.0)))

    # IMPROVED: Ensure minimum gap between text edge and H-ROI to avoid false line detection
    min_gap = 10  # Minimum 10px gap between text and H-ROI

    # Use normal H-ROI height
    h_roi_height_multiplier = 1.0  # Normal height (1x text height)

    # Left horizontal ROI - with validated gap from text edge
    text_left_edge = int(x - text_width//2)
    h_left_x2 = max(0, text_left_edge - min_gap)  # Ensure gap from text
    h_left_x1 = max(0, h_left_x2 - h_strip_extension)  # Extend left from x2
    h_left_y1 = int(y - text_height * h_roi_height_multiplier)  # Adjustable height
    h_left_y2 = int(y + text_height * h_roi_height_multiplier)

    # Right horizontal ROI - with validated gap from text edge
    text_right_edge = int(x + text_width//2)
    h_right_x1 = min(image.shape[1], text_right_edge + min_gap)  # Ensure gap from text
    h_right_x2 = min(image.shape[1], h_right_x1 + h_strip_extension)  # Extend right from x1
    h_right_y1 = h_left_y1
    h_right_y2 = h_left_y2

    if DEBUG_MODE:
        print(f"  Text bounds: left={text_left_edge}, right={text_right_edge}, center_x={x}, width={text_width}")
        print(f"  H-strip extension: {h_strip_extension}px (1.0x text_width)")
        print(f"  Left H-ROI: x=[{h_left_x1}, {h_left_x2}], width={h_left_x2-h_left_x1}, gap_from_text={text_left_edge - h_left_x2}")
        print(f"  Right H-ROI: x=[{h_right_x1}, {h_right_x2}], width={h_right_x2-h_right_x1}, gap_from_text={h_right_x1 - text_right_edge}")

    # For vertical lines: Look ABOVE and BELOW the text
    # Make ROIs TALLER and NARROWER for better vertical line detection
    v_strip_extension = int(text_height * 4)  # Increased from 2x to 4x height for taller ROI

    # Use full text width for vertical ROIs
    # Top vertical ROI - full width of text
    v_top_x1 = int(x - text_width//2)
    v_top_x2 = int(x + text_width//2)
    v_top_y1 = max(0, int(y - text_height//2 - v_strip_extension))
    v_top_y2 = int(y - text_height//2 - 5)

    # Bottom vertical ROI - full width
    v_bottom_x1 = v_top_x1
    v_bottom_x2 = v_top_x2
    v_bottom_y1 = int(y + text_height//2 + 5)
    v_bottom_y2 = min(image.shape[0], int(y + text_height//2 + v_strip_extension))

    # Process horizontal and vertical ROIs separately
    horizontal_lines = []
    vertical_lines = []

    # Helper function to detect arrows in an ROI
    def detect_arrow_in_roi(roi_image, direction):
        """
        Detect arrow pointing in specified direction by looking for converging lines
        direction: 'up', 'down', 'left', 'right'
        """
        if roi_image.size == 0:
            return False

        # Apply HSV filter to detect only green dimension arrows (not cabinet edges)
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array(HSV_CONFIG['lower_green'])
        upper_green = np.array(HSV_CONFIG['upper_green'])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply morphological operations to connect nearby pixels
        kernel = np.ones((2,2), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Detect edges only on the green-filtered image
        edges = cv2.Canny(green_mask, 30, 100)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)

        if lines is None or len(lines) < 2:
            return False

        # Debug: Show what lines we found in arrow detection
        if DEBUG_MODE and direction == 'up':
            print(f"        Arrow detection found {len(lines)} lines in ROI")

        # Look for converging lines that could form an arrow
        # For up arrow: lines should converge at top (Y decreases as they meet)
        # For down arrow: lines should converge at bottom (Y increases as they meet)

        converging_pairs = 0
        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            for j in range(i + 1, len(lines)):
                x3, y3, x4, y4 = lines[j][0]
                angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3))

                # Check if lines have opposite angles (forming a V or ^ shape)
                angle_diff = abs(angle1 + angle2)  # If one is positive and one negative, sum is small

                if direction == 'up':
                    # Debug angle pairs
                    if DEBUG_MODE and i == 0 and j == 1:
                        print(f"        First pair angles: {angle1:.1f}° and {angle2:.1f}°, diff={angle_diff:.1f}")

                    # For up arrow, look for lines that converge upward (like ^)
                    # Check for any pair of lines with significantly different angles
                    # that could form the two sides of an arrow
                    if abs(angle1 - angle2) > 15:  # Lines with different angles
                        # Could be arrow sides
                        converging_pairs += 1
                        if DEBUG_MODE:
                            print(f"        Found converging pair: {angle1:.1f}° and {angle2:.1f}°")

                elif direction == 'down':
                    # For down arrow, look for lines that converge downward (like v)
                    # Angles should be roughly -135 and +135 (or similar)
                    if angle_diff > 150:  # Angles pointing down from opposite sides
                        converging_pairs += 1

                elif direction == 'left':
                    # For left arrow, look for lines that converge leftward (like <)
                    # Angles should form a < shape (one line angled up-left, one down-left)
                    if abs(angle1 - angle2) > 15:  # Lines with different angles
                        converging_pairs += 1

                elif direction == 'right':
                    # For right arrow, look for lines that converge rightward (like >)
                    # Angles should form a > shape (one line angled up-right, one down-right)
                    if abs(angle1 - angle2) > 15:  # Lines with different angles
                        converging_pairs += 1

        return converging_pairs > 0

    # Check if we need to apply rotation to ROIs
    roi_rotation_angle = measurement.get('roi_rotation_angle', 0.0)

    # Store the actual ROI coordinates in the measurement for visualization
    measurement['actual_h_left_roi'] = (h_left_x1, h_left_y1, h_left_x2, h_left_y2)
    measurement['actual_h_right_roi'] = (h_right_x1, h_right_y1, h_right_x2, h_right_y2)
    measurement['actual_v_top_roi'] = (v_top_x1, v_top_y1, v_top_x2, v_top_y2)
    measurement['actual_v_bottom_roi'] = (v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2)

    # Initialize arrow detection flags
    has_left_arrow = False
    has_right_arrow = False

    # Search for horizontal lines
    if h_left_x2 > h_left_x1 and h_left_y2 > h_left_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Left H-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            h_left_roi = extract_rotated_roi(image, h_left_x1, h_left_y1, h_left_x2, h_left_y2,
                                             roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            h_left_roi = image[h_left_y1:h_left_y2, h_left_x1:h_left_x2]

        if DEBUG_MODE:
            print(f"      ACTUAL Left H-ROI coords: x=[{h_left_x1}, {h_left_x2}], y=[{h_left_y1}, {h_left_y2}]")

        left_h_lines = detect_lines_in_roi(h_left_roi, h_left_x1, h_left_y1)
        print(f"      Left H-ROI: shape={h_left_roi.shape}, found {len(left_h_lines)} lines")

        # If no lines found, try arrow detection
        if not left_h_lines:
            has_left_arrow = detect_arrow_in_roi(h_left_roi, 'left')
            if has_left_arrow:
                print(f"        Arrow detection found left arrow")

        if left_h_lines:
            for line in left_h_lines:
                lx1, ly1, lx2, ly2 = line
                angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                print(f"        Raw line: angle={angle:.1f}°")
    else:
        left_h_lines = []
        print(f"      Left H-ROI: Invalid bounds")

    if h_right_x2 > h_right_x1 and h_right_y2 > h_right_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Right H-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            h_right_roi = extract_rotated_roi(image, h_right_x1, h_right_y1, h_right_x2, h_right_y2,
                                              roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            h_right_roi = image[h_right_y1:h_right_y2, h_right_x1:h_right_x2]

        if DEBUG_MODE:
            print(f"      ACTUAL Right H-ROI coords: x=[{h_right_x1}, {h_right_x2}], y=[{h_right_y1}, {h_right_y2}]")

        right_h_lines = detect_lines_in_roi(h_right_roi, h_right_x1, h_right_y1)
        print(f"      Right H-ROI: shape={h_right_roi.shape}, found {len(right_h_lines)} lines")

        # If no lines found, try arrow detection
        if not right_h_lines:
            has_right_arrow = detect_arrow_in_roi(h_right_roi, 'right')
            if has_right_arrow:
                print(f"        Arrow detection found right arrow")

        if right_h_lines:
            for line in right_h_lines:
                lx1, ly1, lx2, ly2 = line
                angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                print(f"        Raw line: angle={angle:.1f}°")
    else:
        right_h_lines = []
        print(f"      Right H-ROI: Invalid bounds")

    # Filter for horizontal lines (more tolerant angles)
    # When ROI is rotated, adjust the expected angle
    for line in left_h_lines + right_h_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))

        # Try both angle adjustments to see which makes the line horizontal
        # Adjustment 1: angle + rotation (lines rotated opposite direction)
        adjusted_angle_1 = angle + roi_rotation_angle
        while adjusted_angle_1 > 180:
            adjusted_angle_1 -= 360
        while adjusted_angle_1 < -180:
            adjusted_angle_1 += 360

        # Adjustment 2: angle - rotation (lines rotated same direction)
        adjusted_angle_2 = angle - roi_rotation_angle
        while adjusted_angle_2 > 180:
            adjusted_angle_2 -= 360
        while adjusted_angle_2 < -180:
            adjusted_angle_2 += 360

        abs_adjusted_1 = abs(adjusted_angle_1)
        abs_adjusted_2 = abs(adjusted_angle_2)

        # Check if either adjustment makes it horizontal
        # More tolerant: 0-35° or 145-180° for "generally horizontal"
        is_horizontal_1 = abs_adjusted_1 < 35 or abs_adjusted_1 > 145
        is_horizontal_2 = abs_adjusted_2 < 35 or abs_adjusted_2 > 145

        if is_horizontal_1 or is_horizontal_2:
            # Use whichever adjustment is closer to horizontal
            if is_horizontal_1 and (not is_horizontal_2 or abs_adjusted_1 < abs_adjusted_2):
                adjusted_angle = adjusted_angle_1
            else:
                adjusted_angle = adjusted_angle_2

            horizontal_lines.append({
                'coords': (lx1, ly1, lx2, ly2),
                'distance': abs(y - (ly1 + ly2) / 2),
                'type': 'horizontal_line',
                'angle': angle,
                'adjusted_angle': adjusted_angle
            })

    if horizontal_lines:
        print(f"      Found {len(horizontal_lines)} horizontal line candidates")

    # Search for vertical lines and arrows
    has_up_arrow = False
    has_down_arrow = False

    if v_top_x2 > v_top_x1 and v_top_y2 > v_top_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Top V-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            v_top_roi = extract_rotated_roi(image, v_top_x1, v_top_y1, v_top_x2, v_top_y2,
                                            roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            v_top_roi = image[v_top_y1:v_top_y2, v_top_x1:v_top_x2]

        top_v_lines = detect_lines_in_roi(v_top_roi, v_top_x1, v_top_y1)
        has_up_arrow = detect_arrow_in_roi(v_top_roi, 'up')
        print(f"      Top V-ROI: shape={v_top_roi.shape}, found {len(top_v_lines)} lines, up-arrow={has_up_arrow}")
    else:
        top_v_lines = []
        print(f"      Top V-ROI: Invalid bounds")

    if v_bottom_x2 > v_bottom_x1 and v_bottom_y2 > v_bottom_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Bottom V-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            v_bottom_roi = extract_rotated_roi(image, v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2,
                                               roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            v_bottom_roi = image[v_bottom_y1:v_bottom_y2, v_bottom_x1:v_bottom_x2]

        bottom_v_lines = detect_lines_in_roi(v_bottom_roi, v_bottom_x1, v_bottom_y1)
        has_down_arrow = detect_arrow_in_roi(v_bottom_roi, 'down')
        print(f"      Bottom V-ROI: shape={v_bottom_roi.shape}, found {len(bottom_v_lines)} lines, down-arrow={has_down_arrow}")
    else:
        bottom_v_lines = []
        print(f"      Bottom V-ROI: Invalid bounds")

    # Store original line counts before filtering
    original_top_line_count = len(top_v_lines)
    original_bottom_line_count = len(bottom_v_lines)

    # Filter for vertical lines (more tolerant angles) and track their source
    # When ROI is rotated, adjust the expected angle
    for line in top_v_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))

        # Adjust angle for rotation
        adjusted_angle = angle + roi_rotation_angle

        # Normalize to -180 to 180
        while adjusted_angle > 180:
            adjusted_angle -= 360
        while adjusted_angle < -180:
            adjusted_angle += 360

        abs_adjusted_angle = abs(adjusted_angle)

        # Debug angle detection
        if DEBUG_MODE and measurement.get('text') == '5 1/2':
            x_dist = abs(x - (lx1 + lx2) / 2)
            print(f"        Top line: angle={angle:.1f}°, adjusted={adjusted_angle:.1f}°, x_dist={x_dist:.1f}, coords=({lx1},{ly1})-({lx2},{ly2})")
        # More tolerant: 55-125° for "generally vertical"
        if 55 < abs_adjusted_angle < 125:
            x_distance = abs(x - (lx1 + lx2) / 2)
            # Keep the distance check but make it less strict
            if x_distance > 2:  # Reduced from 5 to 2
                vertical_lines.append({
                    'coords': (lx1, ly1, lx2, ly2),
                    'distance': x_distance,
                    'type': 'vertical_line',
                    'source': 'top',  # Track that this came from top ROI
                    'angle': angle,  # Store the original angle
                    'adjusted_angle': adjusted_angle  # Store adjusted angle for skew detection
                })
                if DEBUG_MODE:
                    print(f"        Top V-line: angle={angle:.1f}°, adjusted={adjusted_angle:.1f}°, x_dist={x_distance:.1f}")

    for line in bottom_v_lines:
        lx1, ly1, lx2, ly2 = line
        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))

        # Adjust angle for rotation
        adjusted_angle = angle + roi_rotation_angle

        # Normalize to -180 to 180
        while adjusted_angle > 180:
            adjusted_angle -= 360
        while adjusted_angle < -180:
            adjusted_angle += 360

        abs_adjusted_angle = abs(adjusted_angle)

        # More tolerant: 55-125° for "generally vertical"
        if 55 < abs_adjusted_angle < 125:
            x_distance = abs(x - (lx1 + lx2) / 2)
            # Keep the distance check but make it less strict
            if x_distance > 2:  # Reduced from 5 to 2
                vertical_lines.append({
                    'coords': (lx1, ly1, lx2, ly2),
                    'distance': x_distance,
                    'type': 'vertical_line',
                    'source': 'bottom',  # Track that this came from bottom ROI
                    'angle': angle,  # Store the original angle
                    'adjusted_angle': adjusted_angle  # Store adjusted angle for skew detection
                })
                if DEBUG_MODE:
                    print(f"        Bottom V-line: angle={angle:.1f}°, adjusted={adjusted_angle:.1f}°, x_dist={x_distance:.1f}")

    if vertical_lines:
        print(f"      Found {len(vertical_lines)} vertical line candidates")

    # NEW LOGIC: Sequential check requiring HORIZONTAL lines on BOTH sides for WIDTH,
    # or VERTICAL lines on BOTH sides for HEIGHT

    # Step 1: Check for WIDTH - must have HORIZONTAL lines on BOTH left AND right
    # horizontal_lines already contains only horizontal lines (angle filtered)
    # Check which side of the text center each horizontal line is on
    left_h_lines = [l for l in horizontal_lines if l['coords'][0] < x and l['coords'][2] < x]  # Both endpoints left of center
    right_h_lines = [l for l in horizontal_lines if l['coords'][0] > x and l['coords'][2] > x]  # Both endpoints right of center

    has_left_horizontal = len(left_h_lines) > 0 or has_left_arrow
    has_right_horizontal = len(right_h_lines) > 0 or has_right_arrow

    if has_left_horizontal and has_right_horizontal:
        # Found HORIZONTAL lines or arrows on BOTH left and right - classify as WIDTH and stop
        best_h = min(horizontal_lines, key=lambda l: l['distance']) if horizontal_lines else None
        arrow_msg = ""
        if has_left_arrow:
            arrow_msg += " (left arrow)"
        if has_right_arrow:
            arrow_msg += " (right arrow)"
        print(f"      Found HORIZONTAL lines/arrows on BOTH left and right{arrow_msg} → WIDTH")

        # Store all horizontal lines for pairing logic (to extend lines and check for heights)
        measurement['h_lines'] = horizontal_lines

        # Calculate average angle of horizontal lines for intersection calculation
        if horizontal_lines:
            avg_angle = sum(l['angle'] for l in horizontal_lines) / len(horizontal_lines)
            measurement['h_line_angle'] = avg_angle

            # Calculate width extent (leftmost and rightmost X from all h_lines)
            all_x_coords = []
            for line in horizontal_lines:
                lx1, ly1, lx2, ly2 = line['coords']
                all_x_coords.extend([lx1, lx2])

            extent_left = min(all_x_coords)
            extent_right = max(all_x_coords)

            # If only one arrow found, mirror the distance to the other side
            if has_left_arrow and not has_right_arrow:
                # Have left arrow, missing right arrow - mirror left distance to right
                distance_left = abs(x - extent_left)
                extent_right = x + distance_left
                print(f"        Only left arrow found - mirroring distance ({distance_left:.0f}px) to right side")
            elif has_right_arrow and not has_left_arrow:
                # Have right arrow, missing left arrow - mirror right distance to left
                distance_right = abs(extent_right - x)
                extent_left = x - distance_right
                print(f"        Only right arrow found - mirroring distance ({distance_right:.0f}px) to left side")

            measurement['width_extent'] = {
                'left': extent_left,
                'right': extent_right,
                'span': extent_right - extent_left
            }

        if best_h:
            return {
                'line': best_h['coords'],
                'orientation': 'horizontal_line',
                'distance': best_h['distance'],
                'angle': best_h.get('angle', 0)
            }
        else:
            # Arrows detected but no actual lines - still classify as WIDTH
            return {
                'line': None,
                'orientation': 'horizontal_line',
                'distance': 0,
                'angle': 0
            }

    # Step 2: If not WIDTH, check for HEIGHT
    # Use the 'source' field to determine which ROI each line came from
    top_v_lines = [l for l in vertical_lines if l.get('source') == 'top']
    bottom_v_lines = [l for l in vertical_lines if l.get('source') == 'bottom']

    # Debug: Show what lines we found
    if DEBUG_MODE and len(vertical_lines) > 0:
        print(f"      Vertical lines analysis:")
        print(f"        Lines from TOP ROI: {len(top_v_lines)}")
        print(f"        Lines from BOTTOM ROI: {len(bottom_v_lines)}")

    # Check for vertical lines - must have lines from BOTH top AND bottom ROIs
    has_top_vertical = len(top_v_lines) > 0
    has_bottom_vertical = len(bottom_v_lines) > 0

    if has_top_vertical and has_bottom_vertical:
        # Found VERTICAL lines on BOTH top and bottom - classify as HEIGHT
        best_v = min(vertical_lines, key=lambda l: l['distance'])
        print(f"      Found VERTICAL lines on BOTH top and bottom → HEIGHT")

        # Calculate average angle of vertical lines for intersection calculation
        if vertical_lines:
            avg_angle = sum(l['angle'] for l in vertical_lines) / len(vertical_lines)
            measurement['v_line_angle'] = avg_angle

        return {
            'line': best_v['coords'],
            'orientation': 'vertical_line',
            'distance': best_v['distance'],
            'angle': best_v.get('angle', 90)
        }

    # Step 2b: If no vertical lines on both sides, check for arrows as fallback
    if has_up_arrow and has_down_arrow:
        print(f"      Found vertical arrows (up and down) → HEIGHT")
        # Use any vertical lines if they exist, otherwise create placeholder
        if vertical_lines:
            best_v = min(vertical_lines, key=lambda l: l['distance'])
            return {
                'line': best_v['coords'],
                'orientation': 'vertical_line',
                'distance': best_v['distance'],
                'arrow_based': True
            }
        else:
            # No vertical lines but arrows detected - use center position
            return {
                'line': (x, y-20, x, y+20),
                'orientation': 'vertical_line',
                'distance': 0,
                'arrow_based': True
            }

    # Step 3: Neither condition met - UNCLASSIFIED
    print(f"      No lines on both sides (L-horiz:{has_left_horizontal} R-horiz:{has_right_horizontal} T-vert:{has_top_vertical} B-vert:{has_bottom_vertical}) → UNCLASSIFIED")

    # Calculate skew angle from detected vertical lines for fallback retry
    skew_angle = None
    if vertical_lines:
        # Use the closest vertical line's angle to estimate skew
        closest_vline = min(vertical_lines, key=lambda l: l['distance'])
        vline_angle = closest_vline.get('angle', 90.0)
        # Skew is deviation from perfect vertical (90°)
        skew_angle = 90.0 - vline_angle
        print(f"      Detected vertical line angle: {vline_angle:.1f}°, skew from vertical: {skew_angle:.1f}°")

    # Return None but include info about whether vertical lines were found
    # This info will be used by the classification function for fallback retry
    return {
        'unclassified': True,
        'has_vertical_lines': has_top_vertical or has_bottom_vertical,
        'has_left_horizontal': has_left_horizontal,
        'has_right_horizontal': has_right_horizontal,
        'skew_angle': skew_angle  # Include skew for second fallback
    }


def check_for_arrows_at_line_ends(roi_image, roi_x_offset, roi_y_offset, line_coords, direction):
    """
    Check if there are arrow indicators at the ends of a line.

    Args:
        roi_image: The ROI image
        roi_x_offset, roi_y_offset: ROI offsets in full image
        line_coords: (lx1, ly1, lx2, ly2) in full image coordinates
        direction: 'left', 'right', 'up', 'down'

    Returns:
        True if arrows found at line ends
    """
    lx1, ly1, lx2, ly2 = line_coords

    # Convert to ROI coordinates
    roi_lx1 = lx1 - roi_x_offset
    roi_ly1 = ly1 - roi_y_offset
    roi_lx2 = lx2 - roi_x_offset
    roi_ly2 = ly2 - roi_y_offset

    # Check bounds
    if roi_lx1 < 0 or roi_ly1 < 0 or roi_lx2 >= roi_image.shape[1] or roi_ly2 >= roi_image.shape[0]:
        return False

    # Create small search regions at both ends of the line
    search_size = 40

    if direction in ['left', 'right']:
        # Search at left and right ends
        left_region = roi_image[
            max(0, roi_ly1 - search_size):min(roi_image.shape[0], roi_ly1 + search_size),
            max(0, roi_lx1 - search_size):min(roi_image.shape[1], roi_lx1 + search_size)
        ]
        right_region = roi_image[
            max(0, roi_ly2 - search_size):min(roi_image.shape[0], roi_ly2 + search_size),
            max(0, roi_lx2 - search_size):min(roi_image.shape[1], roi_lx2 + search_size)
        ]

        # Detect lines in these regions
        left_lines = detect_lines_in_roi(left_region, 0, 0) if left_region.size > 0 else []
        right_lines = detect_lines_in_roi(right_region, 0, 0) if right_region.size > 0 else []

        # Check for converging line pairs (arrows)
        # LEFT ARROW: Lines converge pointing LEFT (<)
        # Should have one line angling up-left and one angling down-left
        has_left_arrow = False
        if len(left_lines) >= 2:
            for i, line1 in enumerate(left_lines):
                lx1a, ly1a, lx2a, ly2a = line1
                angle1 = np.degrees(np.arctan2(ly2a - ly1a, lx2a - lx1a))

                for line2 in left_lines[i+1:]:
                    lx1b, ly1b, lx2b, ly2b = line2
                    angle2 = np.degrees(np.arctan2(ly2b - ly1b, lx2b - lx1b))

                    # Check if angles are opposite (forming V pattern)
                    # One should be positive, one negative (or 90-270 degrees apart)
                    angle_diff = abs(angle1 - angle2)
                    # Normalize to 0-180 range
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff

                    # Lines should be roughly opposite (90-170 degrees apart)
                    if 90 < angle_diff < 170:
                        has_left_arrow = True
                        break

                if has_left_arrow:
                    break

        # RIGHT ARROW: Lines converge pointing RIGHT (>)
        # Should have one line angling up-right and one angling down-right
        has_right_arrow = False
        if len(right_lines) >= 2:
            for i, line1 in enumerate(right_lines):
                lx1a, ly1a, lx2a, ly2a = line1
                angle1 = np.degrees(np.arctan2(ly2a - ly1a, lx2a - lx1a))

                for line2 in right_lines[i+1:]:
                    lx1b, ly1b, lx2b, ly2b = line2
                    angle2 = np.degrees(np.arctan2(ly2b - ly1b, lx2b - lx1b))

                    # Check if angles are opposite (forming V pattern)
                    angle_diff = abs(angle1 - angle2)
                    # Normalize to 0-180 range
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff

                    # Lines should be roughly opposite (90-170 degrees apart)
                    if 90 < angle_diff < 170:
                        has_right_arrow = True
                        break

                if has_right_arrow:
                    break

        return has_left_arrow or has_right_arrow

    elif direction in ['up', 'down']:
        # Search at top and bottom ends
        top_region = roi_image[
            max(0, roi_ly1 - search_size):min(roi_image.shape[0], roi_ly1 + search_size),
            max(0, roi_lx1 - search_size):min(roi_image.shape[1], roi_lx1 + search_size)
        ]
        bottom_region = roi_image[
            max(0, roi_ly2 - search_size):min(roi_image.shape[0], roi_ly2 + search_size),
            max(0, roi_lx2 - search_size):min(roi_image.shape[1], roi_lx2 + search_size)
        ]

        top_lines = detect_lines_in_roi(top_region, 0, 0) if top_region.size > 0 else []
        bottom_lines = detect_lines_in_roi(bottom_region, 0, 0) if bottom_region.size > 0 else []

        has_top_arrow = len(top_lines) >= 2
        has_bottom_arrow = len(bottom_lines) >= 2

        return has_top_arrow or has_bottom_arrow

    return False


def extend_roi_and_get_full_extent(image, measurement, category):
    """
    After classification, extend the ROI and rescan to get full line extent.
    Only includes lines that have arrows at their ends.

    Args:
        image: The full image
        measurement: The measurement dict with ROI info
        category: 'width' or 'height'

    Returns:
        Extent info dict with left/right (for width) or top/bottom (for height)
    """
    print(f"    Extending ROI for {category.upper()} measurement...")

    x = int(measurement['position'][0])
    y = int(measurement['position'][1])

    # Get text bounds
    if 'bounds' in measurement:
        bounds = measurement['bounds']
        text_left = int(bounds['left'])
        text_right = int(bounds['right'])
        text_top = int(bounds['top'])
        text_bottom = int(bounds['bottom'])
        text_width = text_right - text_left
        text_height = text_bottom - text_top
    else:
        text_height = 30
        text_width = len(measurement.get('text', '')) * 15
        text_left = x - text_width // 2
        text_right = x + text_width // 2
        text_top = y - text_height // 2
        text_bottom = y + text_height // 2

    roi_rotation_angle = measurement.get('roi_rotation_angle', 0)
    h_roi_height_multiplier = 1.0
    v_roi_width_multiplier = 1.0
    min_gap = 15

    if category == 'width':
        # Extend H-ROI maximum 2 inches (192px at 96 DPI) on each side
        max_extension = 192  # 2 inches at 96 DPI

        # Left H-ROI - extend up to 2 inches left
        h_left_x2 = max(0, text_left - min_gap)
        h_left_x1 = max(0, h_left_x2 - max_extension)
        h_left_y1 = int(y - text_height * h_roi_height_multiplier)
        h_left_y2 = int(y + text_height * h_roi_height_multiplier)

        # Right H-ROI - extend up to 2 inches right
        h_right_x1 = min(image.shape[1], text_right + min_gap)
        h_right_x2 = min(image.shape[1], h_right_x1 + max_extension)
        h_right_y1 = h_left_y1
        h_right_y2 = h_left_y2

        print(f"      Extended Left H-ROI: x=[{h_left_x1}, {h_left_x2}] (width: {h_left_x2-h_left_x1}px)")
        print(f"      Extended Right H-ROI: x=[{h_right_x1}, {h_right_x2}] (width: {h_right_x2-h_right_x1}px)")

        # Extract ROIs and detect lines
        if roi_rotation_angle != 0:
            h_left_roi = extract_rotated_roi(image, h_left_x1, h_left_y1, h_left_x2, h_left_y2, roi_rotation_angle, (x, y))
            h_right_roi = extract_rotated_roi(image, h_right_x1, h_right_y1, h_right_x2, h_right_y2, roi_rotation_angle, (x, y))
        else:
            h_left_roi = image[h_left_y1:h_left_y2, h_left_x1:h_left_x2]
            h_right_roi = image[h_right_y1:h_right_y2, h_right_x1:h_right_x2]

        # Detect lines in extended ROIs
        left_h_lines = detect_lines_in_roi(h_left_roi, h_left_x1, h_left_y1)
        right_h_lines = detect_lines_in_roi(h_right_roi, h_right_x1, h_right_y1)

        # Filter for horizontal lines WITH ARROWS
        all_h_lines = []
        lines_with_arrows = []

        for line in left_h_lines + right_h_lines:
            lx1, ly1, lx2, ly2 = line
            angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
            adjusted_angle = (angle + 360) % 360
            if adjusted_angle > 180:
                adjusted_angle = adjusted_angle - 360

            if abs(adjusted_angle) <= 15:
                all_h_lines.append({'coords': (lx1, ly1, lx2, ly2), 'angle': adjusted_angle})

        print(f"      Found {len(all_h_lines)} horizontal lines, checking for arrows...")

        # Check which lines have arrows at their ends
        for line_info in all_h_lines:
            lx1, ly1, lx2, ly2 = line_info['coords']

            # Determine which ROI this line came from
            if lx1 < x:
                # Line is in left ROI
                roi_img = h_left_roi
                roi_x = h_left_x1
                roi_y = h_left_y1
                side = 'left'
            else:
                # Line is in right ROI
                roi_img = h_right_roi
                roi_x = h_right_x1
                roi_y = h_right_y1
                side = 'right'

            # Check for arrows at EITHER end of the line (left OR right direction)
            has_left_arrows = check_for_arrows_at_line_ends(roi_img, roi_x, roi_y, (lx1, ly1, lx2, ly2), 'left')
            has_right_arrows = check_for_arrows_at_line_ends(roi_img, roi_x, roi_y, (lx1, ly1, lx2, ly2), 'right')

            if has_left_arrows or has_right_arrows:
                # Store line with arrow direction info
                line_with_arrow_info = line_info.copy()
                line_with_arrow_info['has_left_arrow'] = has_left_arrows
                line_with_arrow_info['has_right_arrow'] = has_right_arrows
                lines_with_arrows.append(line_with_arrow_info)
                arrow_dir = []
                if has_left_arrows:
                    arrow_dir.append('LEFT')
                if has_right_arrows:
                    arrow_dir.append('RIGHT')
                print(f"        Line ({lx1:.0f},{ly1:.0f})->({lx2:.0f},{ly2:.0f}): HAS {'/'.join(arrow_dir)} ARROWS ({side} ROI)")
            else:
                print(f"        Line ({lx1:.0f},{ly1:.0f})->({lx2:.0f},{ly2:.0f}): no arrows (filtered out, {side} ROI)")

        # Get full extent from lines WITH arrows only
        if lines_with_arrows:
            # Collect arrow endpoints based on arrow direction
            left_arrow_data = []  # (X, Y) tuples from lines with left arrows
            right_arrow_data = []  # (X, Y) tuples from lines with right arrows

            for line in lines_with_arrows:
                lx1, ly1, lx2, ly2 = line['coords']

                # If line has left arrow, the leftmost endpoint is the arrow tip
                if line.get('has_left_arrow'):
                    if lx1 <= lx2:
                        left_arrow_data.append((lx1, ly1))
                        print(f"          [DEBUG] Left arrow: line ({lx1:.0f},{ly1:.0f})->({lx2:.0f},{ly2:.0f}), chose point1 ({lx1:.0f},{ly1:.0f})")
                    else:
                        left_arrow_data.append((lx2, ly2))
                        print(f"          [DEBUG] Left arrow: line ({lx1:.0f},{ly1:.0f})->({lx2:.0f},{ly2:.0f}), chose point2 ({lx2:.0f},{ly2:.0f})")

                # If line has right arrow, the rightmost endpoint is the arrow tip
                if line.get('has_right_arrow'):
                    if lx1 >= lx2:
                        right_arrow_data.append((lx1, ly1))
                        print(f"          [DEBUG] Right arrow: line ({lx1:.0f},{ly1:.0f})->({lx2:.0f},{ly2:.0f}), chose point1 ({lx1:.0f},{ly1:.0f})")
                    else:
                        right_arrow_data.append((lx2, ly2))
                        print(f"          [DEBUG] Right arrow: line ({lx1:.0f},{ly1:.0f})->({lx2:.0f},{ly2:.0f}), chose point2 ({lx2:.0f},{ly2:.0f})")

            # Calculate average angle of lines with arrows
            avg_angle = sum(l['angle'] for l in lines_with_arrows) / len(lines_with_arrows)

            # Determine extent based on which arrows were found
            if left_arrow_data:
                left_arrow_data.sort(key=lambda p: p[0])  # Sort by X
                extent_left = left_arrow_data[0][0]  # Leftmost X
                left_y = left_arrow_data[0][1]  # Y at that point
            else:
                extent_left = None
                left_y = None

            if right_arrow_data:
                right_arrow_data.sort(key=lambda p: p[0], reverse=True)  # Sort by X descending
                extent_right = right_arrow_data[0][0]  # Rightmost X
                right_y = right_arrow_data[0][1]  # Y at that point
            else:
                extent_right = None
                right_y = None

            # If only one arrow found, mirror the distance to the other side
            if extent_left is not None and extent_right is None:
                # Have left arrow only - mirror left distance to right
                distance_left = abs(x - extent_left)
                extent_right = x + distance_left
                # Calculate Y for mirrored right side using the angle
                if left_y is not None:
                    dx = extent_right - extent_left
                    dy = dx * np.tan(np.radians(avg_angle))
                    right_y = left_y + dy
                else:
                    right_y = left_y
                print(f"        [Extended] Only left arrow found at {extent_left:.0f} - mirroring distance ({distance_left:.0f}px) to right side")
            elif extent_right is not None and extent_left is None:
                # Have right arrow only - mirror right distance to left
                distance_right = abs(extent_right - x)
                extent_left = x - distance_right
                # Calculate Y for mirrored left side using the angle
                if right_y is not None:
                    dx = extent_left - extent_right
                    dy = dx * np.tan(np.radians(avg_angle))
                    left_y = right_y + dy
                else:
                    left_y = right_y
                print(f"        [Extended] Only right arrow found at {extent_right:.0f} - mirroring distance ({distance_right:.0f}px) to left side")
            elif extent_left is None and extent_right is None:
                print(f"        [Extended] ERROR: No arrow coords found even though lines_with_arrows exists")
                return None

            extent = {
                'left': extent_left,
                'right': extent_right,
                'left_y': left_y,
                'right_y': right_y,
                'span': extent_right - extent_left,
                'angle': avg_angle,
                # Add ROI bounds for visualization
                'debug_rois': {
                    'left': (h_left_x1, h_left_y1, h_left_x2, h_left_y2),
                    'right': (h_right_x1, h_right_y1, h_right_x2, h_right_y2)
                }
            }
            print(f"      Found full width extent (arrows only): {extent['left']:.0f} to {extent['right']:.0f} (span: {extent['span']:.0f}px, angle: {avg_angle:.1f}°)")
            print(f"      Used {len(lines_with_arrows)} lines with arrows (filtered out {len(all_h_lines) - len(lines_with_arrows)} without arrows)")
            return extent
        else:
            print(f"      No lines with arrows found in extended ROI")
            return None

    elif category == 'height':
        # Extend V-ROI to full image height
        v_strip_extension = text_height * 5

        # Top V-ROI - extend to top edge
        v_top_x1 = int(x - text_width * v_roi_width_multiplier)
        v_top_x2 = int(x + text_width * v_roi_width_multiplier)
        v_top_y1 = 0
        v_top_y2 = max(0, text_top - min_gap)

        # Bottom V-ROI - extend to bottom edge
        v_bottom_x1 = v_top_x1
        v_bottom_x2 = v_top_x2
        v_bottom_y1 = min(image.shape[0], text_bottom + min_gap)
        v_bottom_y2 = image.shape[0]

        print(f"      Extended Top V-ROI: y=[{v_top_y1}, {v_top_y2}] (height: {v_top_y2-v_top_y1}px)")
        print(f"      Extended Bottom V-ROI: y=[{v_bottom_y1}, {v_bottom_y2}] (height: {v_bottom_y2-v_bottom_y1}px)")

        # Extract ROIs and detect lines
        if roi_rotation_angle != 0:
            v_top_roi = extract_rotated_roi(image, v_top_x1, v_top_y1, v_top_x2, v_top_y2, roi_rotation_angle, (x, y))
            v_bottom_roi = extract_rotated_roi(image, v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2, roi_rotation_angle, (x, y))
        else:
            v_top_roi = image[v_top_y1:v_top_y2, v_top_x1:v_top_x2]
            v_bottom_roi = image[v_bottom_y1:v_bottom_y2, v_bottom_x1:v_bottom_x2]

        # Detect lines in extended ROIs
        top_v_lines = detect_lines_in_roi(v_top_roi, v_top_x1, v_top_y1)
        bottom_v_lines = detect_lines_in_roi(v_bottom_roi, v_bottom_x1, v_bottom_y1)

        # Filter for vertical lines
        all_v_lines = []
        for line in top_v_lines + bottom_v_lines:
            lx1, ly1, lx2, ly2 = line
            angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
            adjusted_angle = (angle + 360) % 360
            if adjusted_angle > 180:
                adjusted_angle = adjusted_angle - 360

            if 75 <= abs(adjusted_angle) <= 105:
                all_v_lines.append({'coords': (lx1, ly1, lx2, ly2), 'angle': adjusted_angle})

        # Get full extent
        if all_v_lines:
            all_y_coords = []
            for line in all_v_lines:
                lx1, ly1, lx2, ly2 = line['coords']
                all_y_coords.extend([ly1, ly2])

            extent = {
                'top': min(all_y_coords),
                'bottom': max(all_y_coords),
                'span': max(all_y_coords) - min(all_y_coords)
            }
            print(f"      Found full height extent: {extent['top']:.0f} to {extent['bottom']:.0f} (span: {extent['span']:.0f}px)")
            return extent
        else:
            print(f"      No lines found in extended ROI")
            return None

    return None


def classify_measurements_by_lines(image, measurements):
    """
    Classify measurements as WIDTH, HEIGHT, or UNCLASSIFIED based on nearby dimension lines

    Returns:
        - classified: dict with 'width', 'height', 'unclassified' lists
        - measurement_categories: list of categories for each measurement
    """
    classified = {
        'width': [],
        'height': [],
        'unclassified': []
    }

    measurement_categories = []

    for i, meas in enumerate(measurements):
        print(f"\n  Analyzing measurement {i+1}: '{meas['text']}' at ({meas['position'][0]:.0f}, {meas['position'][1]:.0f})")

        # Find lines near this measurement
        line_info = find_lines_near_measurement(image, meas)

        # Check if it's actually classified or unclassified
        if line_info and not line_info.get('unclassified'):
            print(f"    Found {line_info['orientation']} at distance {line_info['distance']:.1f} pixels")
            # Horizontal line = WIDTH measurement
            # Vertical line = HEIGHT measurement
            if line_info['orientation'] == 'horizontal_line':
                classified['width'].append(meas['text'])
                measurement_categories.append('width')
                print(f"    → Classified as WIDTH")
                # Extend ROI and get full width extent
                full_extent = extend_roi_and_get_full_extent(image, meas, 'width')
                if full_extent:
                    meas['width_extent'] = full_extent
            elif line_info['orientation'] == 'vertical_line':
                classified['height'].append(meas['text'])
                measurement_categories.append('height')
                print(f"    → Classified as HEIGHT")
                # Extend ROI and get full height extent
                full_extent = extend_roi_and_get_full_extent(image, meas, 'height')
                if full_extent:
                    meas['height_extent'] = full_extent
        else:
            # UNCLASSIFIED - try clockwise rotation fallback
            print(f"    UNCLASSIFIED - Trying clockwise rotation fallback...")

            # Try clockwise rotation (22.5°)
            print(f"    Attempting +22.5° rotation...")
            meas_cw = meas.copy()
            meas_cw['roi_rotation_angle'] = 22.5
            line_info_cw = find_lines_near_measurement(image, meas_cw)

            if line_info_cw and not line_info_cw.get('unclassified'):
                print(f"    ROTATION FALLBACK SUCCESS: Found {line_info_cw['orientation']} with +22.5° rotation!")
                if line_info_cw['orientation'] == 'horizontal_line':
                    classified['width'].append(meas['text'])
                    measurement_categories.append('width')
                    meas['roi_rotation_angle'] = 22.5
                    meas['actual_h_left_roi'] = meas_cw.get('actual_h_left_roi')
                    meas['actual_h_right_roi'] = meas_cw.get('actual_h_right_roi')
                    meas['actual_v_top_roi'] = meas_cw.get('actual_v_top_roi')
                    meas['actual_v_bottom_roi'] = meas_cw.get('actual_v_bottom_roi')
                    print(f"    → Classified as WIDTH (via +22.5° rotation)")
                    # Extend ROI and get full width extent
                    full_extent = extend_roi_and_get_full_extent(image, meas, 'width')
                    if full_extent:
                        meas['width_extent'] = full_extent
                elif line_info_cw['orientation'] == 'vertical_line':
                    classified['height'].append(meas['text'])
                    measurement_categories.append('height')
                    meas['roi_rotation_angle'] = 22.5
                    meas['actual_h_left_roi'] = meas_cw.get('actual_h_left_roi')
                    meas['actual_h_right_roi'] = meas_cw.get('actual_h_right_roi')
                    meas['actual_v_top_roi'] = meas_cw.get('actual_v_top_roi')
                    meas['actual_v_bottom_roi'] = meas_cw.get('actual_v_bottom_roi')
                    print(f"    → Classified as HEIGHT (via +22.5° rotation)")
                    # Extend ROI and get full height extent
                    full_extent = extend_roi_and_get_full_extent(image, meas, 'height')
                    if full_extent:
                        meas['height_extent'] = full_extent
            else:
                # Still unclassified after rotation - reset to 0° for viz
                meas['roi_rotation_angle'] = 0.0  # Reset to regular ROIs in viz
                meas['rotation_failed'] = True
                # Don't store the rotated ROI coords - keep original non-rotated ones
                print(f"    No dimension lines found even with rotation")
                classified['unclassified'].append(meas['text'])
                measurement_categories.append('unclassified')
                print(f"    → Classified as UNCLASSIFIED (rotation attempted but failed, ROIs reset to 0°)")

    return classified, measurement_categories
