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
    """Extract pixels within a rotated rectangle from the image

    Returns:
        tuple: (roi, inverse_transform_matrix) where inverse_transform can map ROI coords back to image coords
    """
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

    # Get perspective transform matrix (maps rotated corners in image -> unrotated ROI)
    M = cv2.getPerspectiveTransform(rotated_corners, dst_corners)

    # Get inverse transform (maps ROI coords -> image coords)
    M_inv = cv2.getPerspectiveTransform(dst_corners, rotated_corners)

    # Apply perspective transform to get the rotated region
    roi = cv2.warpPerspective(image, M, (width, height))

    return roi, M_inv


def detect_lines_in_roi(roi_image, roi_x_offset, roi_y_offset, inverse_transform=None, all_measurements=None, exclude_items=None, measurement=None, roi_x1=None, roi_y1=None, roi_x2=None, roi_y2=None):
    """Detect lines in an ROI and return coordinates adjusted to full image

    Args:
        roi_image: The ROI image
        roi_x_offset: X offset for simple translation (used when inverse_transform is None)
        roi_y_offset: Y offset for simple translation (used when inverse_transform is None)
        inverse_transform: Optional 3x3 perspective transform matrix to map ROI coords to image coords
        all_measurements: List of all measurements for text exclusion masking
        exclude_items: List of items to exclude (OL text, room names)
        measurement: Current measurement being analyzed
        roi_x1, roi_y1, roi_x2, roi_y2: ROI bounds in image coordinates

    Returns:
        List of line coordinates in full image space
    """
    if roi_image.size == 0:
        return []

    # Apply HSV filter to detect only green dimension lines (not cabinet edges)
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array(HSV_CONFIG['lower_green'])
    upper_green = np.array(HSV_CONFIG['upper_green'])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to connect nearby pixels
    kernel = np.ones((2,2), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Apply text exclusion masking to prevent detecting lines in text regions
    # Exclude other measurement text areas from line detection
    if all_measurements is not None and roi_x1 is not None and roi_y1 is not None and roi_x2 is not None and roi_y2 is not None:
        for other_meas in all_measurements:
            # Skip the current measurement being analyzed
            if measurement and other_meas.get('text') == measurement.get('text') and \
               abs(other_meas.get('x', 0) - measurement.get('x', 0)) < 5 and \
               abs(other_meas.get('y', 0) - measurement.get('y', 0)) < 5:
                continue

            if 'bounds' not in other_meas or not other_meas['bounds']:
                continue

            bounds = other_meas['bounds']
            # Calculate intersection with ROI
            intersect_x1 = max(roi_x1, int(bounds['left']))
            intersect_y1 = max(roi_y1, int(bounds['top']))
            intersect_x2 = min(roi_x2, int(bounds['right']))
            intersect_y2 = min(roi_y2, int(bounds['bottom']))

            if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                # Convert to ROI-relative coordinates
                roi_ix1 = intersect_x1 - roi_x1
                roi_iy1 = intersect_y1 - roi_y1
                roi_ix2 = intersect_x2 - roi_x1
                roi_iy2 = intersect_y2 - roi_y1
                # Zero out this region in the mask
                green_mask[roi_iy1:roi_iy2, roi_ix1:roi_ix2] = 0

    # Also exclude OL text and room names from line detection
    if exclude_items is not None and roi_x1 is not None and roi_y1 is not None and roi_x2 is not None and roi_y2 is not None:
        for excluded_item in exclude_items:
            if 'x' not in excluded_item or 'y' not in excluded_item:
                continue

            # Use actual bounding box if available, otherwise estimate
            if 'bounds' in excluded_item and excluded_item['bounds']:
                bounds = excluded_item['bounds']
                ex_left = int(bounds['left'])
                ex_right = int(bounds['right'])
                ex_top = int(bounds['top'])
                ex_bottom = int(bounds['bottom'])
            else:
                # Fallback: estimate bounds (for old data)
                ex = excluded_item['x']
                ey = excluded_item['y']
                text_width = 100
                text_height = 30
                ex_left = int(ex - text_width/2)
                ex_right = int(ex + text_width/2)
                ex_top = int(ey - text_height/2)
                ex_bottom = int(ey + text_height/2)

            # Calculate intersection with ROI
            intersect_x1 = max(roi_x1, ex_left)
            intersect_y1 = max(roi_y1, ex_top)
            intersect_x2 = min(roi_x2, ex_right)
            intersect_y2 = min(roi_y2, ex_bottom)

            if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                print(f"          MASKED exclude_item '{excluded_item.get('text')}' bounds=[{ex_left},{ex_right},{ex_top},{ex_bottom}] - intersect: ({intersect_x1},{intersect_y1}) to ({intersect_x2},{intersect_y2})")
                # Convert to ROI-relative coordinates
                roi_ix1 = intersect_x1 - roi_x1
                roi_iy1 = intersect_y1 - roi_y1
                roi_ix2 = intersect_x2 - roi_x1
                roi_iy2 = intersect_y2 - roi_y1
                # Zero out this region in the mask
                green_mask[roi_iy1:roi_iy2, roi_ix1:roi_ix2] = 0

    # Detect edges only on the green-filtered image
    edges = cv2.Canny(green_mask, 30, 100)

    # Find lines with HoughLinesP (using relaxed parameters to catch arrow segments)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=10,
                            minLineLength=10,
                            maxLineGap=5)

    # Convert line coordinates back to full image coordinates
    if lines is not None:
        adjusted_lines = []
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]

            if inverse_transform is not None:
                # Apply perspective transform to map ROI coordinates to image coordinates
                # Transform point 1
                pt1 = np.array([[[lx1, ly1]]], dtype=np.float32)
                pt1_transformed = cv2.perspectiveTransform(pt1, inverse_transform)
                img_x1, img_y1 = pt1_transformed[0][0]

                # Transform point 2
                pt2 = np.array([[[lx2, ly2]]], dtype=np.float32)
                pt2_transformed = cv2.perspectiveTransform(pt2, inverse_transform)
                img_x2, img_y2 = pt2_transformed[0][0]

                # Store both ROI coords (for arrow detection) and image coords (for visualization)
                adjusted_lines.append({
                    'image_coords': [img_x1, img_y1, img_x2, img_y2],
                    'roi_coords': [lx1, ly1, lx2, ly2]
                })
            else:
                # Simple offset for non-rotated ROIs
                img_coords = [lx1 + roi_x_offset, ly1 + roi_y_offset,
                             lx2 + roi_x_offset, ly2 + roi_y_offset]
                adjusted_lines.append({
                    'image_coords': img_coords,
                    'roi_coords': [lx1, ly1, lx2, ly2]
                })
        return adjusted_lines
    return []


def find_lines_near_measurement(image, measurement, save_roi_debug=False, image_path=None, exclude_items=None, all_measurements=None):
    """Find lines near a specific measurement position that match the text color

    Args:
        image: Source image
        measurement: Measurement dict with position, text, bounds
        save_roi_debug: Whether to save debug ROI images
        image_path: Path to image for debug output
        exclude_items: List of items to exclude from arrow detection (OL text, room names)
        all_measurements: List of all measurements for text exclusion
    """
    import os

    x = int(measurement['position'][0])
    y = int(measurement['position'][1])

    # Enable debug for troubleshooting
    DEBUG_MODE = True  # Enable to see width calculation and H-ROI positioning details

    # Setup for classification ROI debug image saving
    output_dir = None
    base_name = None
    m_id = None
    m_text = None
    if image_path and save_roi_debug:
        output_dir = os.path.dirname(os.path.abspath(image_path))
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        m_id = measurement.get('id', 'unknown')
        m_text = measurement['text'].replace(' ', '').replace('/', '-').replace('*', '').replace('$', '')

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

    # Check if we need to apply rotation to ROIs - retrieve early to adjust gaps
    roi_rotation_angle = measurement.get('roi_rotation_angle', 0.0)

    # Create ROIs for horizontal and vertical line detection
    # For horizontal lines: Look LEFT and RIGHT of the text
    # Use 1.0x text width but clamp to 60 pixels (5/8 inch at 96 DPI)
    h_strip_extension = min(60, max(60, int(text_width * 1.0)))

    # IMPROVED: Ensure minimum gap between text edge and H-ROI to avoid false line detection
    # When ROI is rotated, increase gap to prevent capturing text strokes
    base_h_gap = 10  # Base gap between text and H-ROI
    base_v_gap = 5   # Base gap between text and V-ROI

    min_gap = base_h_gap
    v_gap = base_v_gap

    if roi_rotation_angle != 0:
        # H-ROI: DECREASE gap when rotated to get closer to angled arrows
        h_gap_multiplier = 0.4  # Reduce gap to get closer
        min_gap = max(2, int(base_h_gap * h_gap_multiplier))  # Minimum 2px

        # V-ROI: Calculate extra clearance based on rotation geometry
        # When V-ROI (width = text_width) is rotated, its vertical extent increases by: width * sin(angle)
        import math
        angle_rad = math.radians(abs(roi_rotation_angle))
        v_extra_clearance = int(text_width * math.sin(angle_rad))
        v_gap = base_v_gap + v_extra_clearance

        if DEBUG_MODE:
            print(f"  ROI rotation detected ({roi_rotation_angle:.1f}°): adjusted gaps")
            print(f"    H-ROI: base={base_h_gap}px * {h_gap_multiplier} = {min_gap}px (CLOSER to text)")
            print(f"    V-ROI: base={base_v_gap}px + extra={v_extra_clearance}px = {v_gap}px (FURTHER from text)")

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
    v_top_y2 = int(y - text_height//2 - v_gap)

    # Bottom vertical ROI - full width
    v_bottom_x1 = v_top_x1
    v_bottom_x2 = v_top_x2
    v_bottom_y1 = int(y + text_height//2 + v_gap)
    v_bottom_y2 = min(image.shape[0], int(y + text_height//2 + v_strip_extension))

    # Process horizontal and vertical ROIs separately
    horizontal_lines = []
    vertical_lines = []

    # Helper function to detect arrows in an ROI
    def detect_arrow_in_roi(roi_image, direction, roi_x1=None, roi_y1=None, roi_x2=None, roi_y2=None,
                           all_measurements=None, exclude_items=None, measurement=None):
        """
        Detect arrow pointing in specified direction by looking for converging lines
        direction: 'up', 'down', 'left', 'right'
        roi_x1, roi_y1, roi_x2, roi_y2: ROI bounds in full image coordinates (for text exclusion)

        Returns: (has_correct_arrow, has_wrong_arrow)
            has_correct_arrow: True if arrow pointing correct direction found
            has_wrong_arrow: True if arrow pointing wrong direction found (should ignore v-lines)
        """
        if roi_image.size == 0:
            return (False, False)

        # Apply HSV filter to detect only green dimension arrows (not cabinet edges)
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array(HSV_CONFIG['lower_green'])
        upper_green = np.array(HSV_CONFIG['upper_green'])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply morphological operations to connect nearby pixels
        kernel = np.ones((2,2), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Exclude other measurement text areas from arrow detection to avoid false positives
        if roi_x1 is not None and roi_y1 is not None and all_measurements:
            print(f"        [TEXT EXCLUSION] Starting - ROI bounds: x=[{roi_x1},{roi_x2}], y=[{roi_y1},{roi_y2}]")
            print(f"        [TEXT EXCLUSION] Checking {len(all_measurements)} measurements for intersection")
            excluded_count = 0
            for other_meas in all_measurements:
                # Skip the current measurement
                if other_meas.get('text') == measurement.get('text') and \
                   abs(other_meas.get('x', 0) - measurement.get('x', 0)) < 5 and \
                   abs(other_meas.get('y', 0) - measurement.get('y', 0)) < 5:
                    print(f"          SKIP '{other_meas.get('text')}' - current measurement")
                    continue

                if 'bounds' not in other_meas or not other_meas['bounds']:
                    print(f"          SKIP '{other_meas.get('text')}' - no bounds")
                    continue

                bounds = other_meas['bounds']
                # Expand bounds to include potential dimension lines around the measurement
                # Dimension lines can extend 200px in any direction
                expand_padding = 200
                expanded_left = int(bounds['left']) - expand_padding
                expanded_right = int(bounds['right']) + expand_padding
                expanded_top = int(bounds['top']) - expand_padding
                expanded_bottom = int(bounds['bottom']) + expand_padding

                # Calculate intersection with ROI using expanded bounds
                intersect_x1 = max(roi_x1, expanded_left)
                intersect_y1 = max(roi_y1, expanded_top)
                intersect_x2 = min(roi_x2, expanded_right)
                intersect_y2 = min(roi_y2, expanded_bottom)

                if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                    # This text region intersects the ROI - exclude it from mask
                    roi_ix1 = intersect_x1 - roi_x1
                    roi_iy1 = intersect_y1 - roi_y1
                    roi_ix2 = intersect_x2 - roi_x1
                    roi_iy2 = intersect_y2 - roi_y1
                    # Zero out this region in the mask
                    green_mask[roi_iy1:roi_iy2, roi_ix1:roi_ix2] = 0
                    excluded_count += 1
                    print(f"          MASKED '{other_meas.get('text')}' bounds=[{int(bounds['left'])},{int(bounds['right'])},{int(bounds['top'])},{int(bounds['bottom'])}] - intersect: ({intersect_x1},{intersect_y1}) to ({intersect_x2},{intersect_y2})")
                else:
                    print(f"          NO INTERSECT '{other_meas.get('text')}' bounds=[{int(bounds['left'])},{int(bounds['right'])},{int(bounds['top'])},{int(bounds['bottom'])}]")
            print(f"        [TEXT EXCLUSION] Complete - excluded {excluded_count} measurements from arrow detection")
        else:
            print(f"        [TEXT EXCLUSION] SKIPPED - roi_x1={roi_x1}, roi_y1={roi_y1}, all_measurements={'None' if all_measurements is None else len(all_measurements)}")

        # Also exclude OL text and room names from arrow detection
        if roi_x1 is not None and roi_y1 is not None and exclude_items:
            print(f"        [EXCLUDE ITEMS] Checking {len(exclude_items)} excluded items (OL/room names)")
            for excluded_item in exclude_items:
                if 'x' not in excluded_item or 'y' not in excluded_item:
                    continue

                # Use actual bounding box if available, otherwise estimate
                if 'bounds' in excluded_item and excluded_item['bounds']:
                    bounds = excluded_item['bounds']
                    ex_left = int(bounds['left'])
                    ex_right = int(bounds['right'])
                    ex_top = int(bounds['top'])
                    ex_bottom = int(bounds['bottom'])
                else:
                    # Fallback: estimate bounds (for old data)
                    ex = excluded_item['x']
                    ey = excluded_item['y']
                    text_width_excl = 200
                    text_height_excl = 50
                    ex_left = int(ex - text_width_excl/2)
                    ex_right = int(ex + text_width_excl/2)
                    ex_top = int(ey - text_height_excl/2)
                    ex_bottom = int(ey + text_height_excl/2)

                # Calculate intersection with ROI
                intersect_x1 = max(roi_x1, ex_left)
                intersect_y1 = max(roi_y1, ex_top)
                intersect_x2 = min(roi_x2, ex_right)
                intersect_y2 = min(roi_y2, ex_bottom)

                if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                    print(f"          MASKED exclude_item '{excluded_item.get('text')}' bounds=[{ex_left},{ex_right},{ex_top},{ex_bottom}] - intersect: ({intersect_x1},{intersect_y1}) to ({intersect_x2},{intersect_y2})")
                    # This OL/room text intersects the ROI - exclude it from mask
                    roi_ix1 = intersect_x1 - roi_x1
                    roi_iy1 = intersect_y1 - roi_y1
                    roi_ix2 = intersect_x2 - roi_x1
                    roi_iy2 = intersect_y2 - roi_y1
                    # Zero out this region in the mask
                    green_mask[roi_iy1:roi_iy2, roi_ix1:roi_ix2] = 0

        # Detect edges only on the green-filtered image
        edges = cv2.Canny(green_mask, 30, 100)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)

        if lines is None or len(lines) < 2:
            if DEBUG_MODE and direction in ['right', 'left']:
                line_count = 0 if lines is None else len(lines)
                print(f"        Arrow detection ({direction}): only {line_count} lines found, need >=2")
            return (False, False)

        # Debug: Show what lines we found in arrow detection
        if DEBUG_MODE and direction in ['up', 'right', 'left', 'down']:
            print(f"        Arrow detection ({direction}) found {len(lines)} lines in ROI")

        # Look for converging lines that could form an arrow
        # For up arrow: lines should converge at top (Y decreases as they meet)
        # For down arrow: lines should converge at bottom (Y increases as they meet)

        converging_pairs = 0
        wrong_direction_pairs = 0  # Track arrows pointing wrong way
        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Normalize to 0-360° range (matching working arrow detection code)
            if angle1 < 0:
                angle1 += 360

            for j in range(i + 1, len(lines)):
                x3, y3, x4, y4 = lines[j][0]
                angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3))
                # Normalize to 0-360° range (matching working arrow detection code)
                if angle2 < 0:
                    angle2 += 360

                # Check if lines have opposite angles (forming a V or ^ shape)
                angle_diff = abs(angle1 + angle2)  # If one is positive and one negative, sum is small

                if direction == 'up':
                    # Debug angle pairs
                    if DEBUG_MODE and i == 0 and j == 1:
                        print(f"        First pair angles: {angle1:.1f}° and {angle2:.1f}°, diff={angle_diff:.1f}")

                    # For up arrow, look for ^ shape pointing upward
                    # Inverse of DOWN arrows (which use 20-88° and 285-360°):
                    # Right leg: 92-160° (up-right from apex)
                    # Left leg: 200-268° (up-left from apex)

                    # Check if we have both legs of a ^
                    has_right_leg_up = (92 <= angle1 <= 160) or (92 <= angle2 <= 160)
                    has_left_leg_up = (200 <= angle1 <= 268) or (200 <= angle2 <= 268)

                    if has_right_leg_up and has_left_leg_up:
                        # Both legs present - valid UP arrow
                        converging_pairs += 1
                        if DEBUG_MODE:
                            print(f"        Found UP arrow: angles {angle1:.1f}° and {angle2:.1f}° (right leg 92-160°, left leg 200-268°)")

                    # Check for wrong-direction DOWN arrow in UP ROI
                    # DOWN arrows: right leg 20-88°, left leg 285-360°
                    has_right_leg = (20 <= angle1 <= 88) or (20 <= angle2 <= 88)
                    has_left_leg = (285 <= angle1 <= 360) or (285 <= angle2 <= 360)

                    if has_right_leg and has_left_leg:
                        # Found DOWN arrow in UP ROI - wrong direction
                        wrong_direction_pairs += 1
                        if DEBUG_MODE:
                            print(f"        Rejected DOWN arrow in UP ROI: {angle1:.1f}° and {angle2:.1f}°")

                elif direction == 'down':
                    # For down arrow, look for V shape pointing downward (like v)
                    # Based on working code from measurement_pairing_v2.py:
                    # Right leg: 20-88° (down-right from apex)
                    # Left leg: 285-360° (down-left from apex)

                    # Check if we have both legs of a V
                    has_right_leg = (20 <= angle1 <= 88) or (20 <= angle2 <= 88)
                    has_left_leg = (285 <= angle1 <= 360) or (285 <= angle2 <= 360)

                    if has_right_leg and has_left_leg:
                        # Both legs present - valid DOWN arrow
                        converging_pairs += 1
                        if DEBUG_MODE:
                            print(f"        Found DOWN arrow: angles {angle1:.1f}° and {angle2:.1f}° (right leg 20-88°, left leg 285-360°)")

                    # Check for wrong-direction UP arrow in DOWN ROI
                    # UP arrows: right leg 92-160°, left leg 200-268°
                    has_right_leg_up = (92 <= angle1 <= 160) or (92 <= angle2 <= 160)
                    has_left_leg_up = (200 <= angle1 <= 268) or (200 <= angle2 <= 268)

                    if has_right_leg_up and has_left_leg_up:
                        # Found UP arrow in DOWN ROI - wrong direction
                        wrong_direction_pairs += 1
                        if DEBUG_MODE:
                            print(f"        Rejected UP arrow in DOWN ROI: {angle1:.1f}° and {angle2:.1f}°")

                elif direction == 'left':
                    # For left arrow, look for lines that converge leftward (like <)
                    # Angles should form a < shape (one line angled up-left, one down-left)
                    if abs(angle1 - angle2) > 15:  # Lines with different angles
                        converging_pairs += 1

                elif direction == 'right':
                    # For right arrow, look for lines that converge rightward (like >)
                    # Angles should form a > shape (one line angled up-right, one down-right)
                    if DEBUG_MODE and i == 0 and j == 1:
                        print(f"        First pair angles: {angle1:.1f}° and {angle2:.1f}°, diff={abs(angle1-angle2):.1f}")
                    if abs(angle1 - angle2) > 15:  # Lines with different angles
                        converging_pairs += 1
                        if DEBUG_MODE:
                            print(f"        Found converging pair for RIGHT arrow: {angle1:.1f}° and {angle2:.1f}°")

        has_correct_arrow = converging_pairs > 0
        has_wrong_arrow = wrong_direction_pairs > 0

        if DEBUG_MODE and (has_correct_arrow or has_wrong_arrow):
            if has_correct_arrow:
                print(f"        Arrow detection ({direction}): returning True ({converging_pairs} correct pairs)")
            if has_wrong_arrow:
                print(f"        Arrow detection ({direction}): found {wrong_direction_pairs} WRONG direction pairs")
        return (has_correct_arrow, has_wrong_arrow)

    # Store the actual ROI coordinates in the measurement for visualization
    measurement['actual_h_left_roi'] = (h_left_x1, h_left_y1, h_left_x2, h_left_y2)
    measurement['actual_h_right_roi'] = (h_right_x1, h_right_y1, h_right_x2, h_right_y2)
    measurement['actual_v_top_roi'] = (v_top_x1, v_top_y1, v_top_x2, v_top_y2)
    measurement['actual_v_bottom_roi'] = (v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2)

    # Initialize arrow detection flags
    has_left_arrow = False
    has_right_arrow = False

    # Search for horizontal lines
    h_left_M_inv = None
    if h_left_x2 > h_left_x1 and h_left_y2 > h_left_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Left H-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            h_left_roi, h_left_M_inv = extract_rotated_roi(image, h_left_x1, h_left_y1, h_left_x2, h_left_y2,
                                                            roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            h_left_roi = image[h_left_y1:h_left_y2, h_left_x1:h_left_x2]

        if DEBUG_MODE:
            print(f"      ACTUAL Left H-ROI coords: x=[{h_left_x1}, {h_left_x2}], y=[{h_left_y1}, {h_left_y2}]")

        left_h_lines = detect_lines_in_roi(h_left_roi, h_left_x1, h_left_y1, h_left_M_inv,
                                           all_measurements=all_measurements, exclude_items=exclude_items,
                                           measurement=measurement, roi_x1=h_left_x1, roi_y1=h_left_y1,
                                           roi_x2=h_left_x2, roi_y2=h_left_y2)
        print(f"      Left H-ROI: shape={h_left_roi.shape}, found {len(left_h_lines)} lines")

        # Save debug image of left H-ROI with detected lines
        if output_dir is not None:
            debug_img_left = h_left_roi.copy()
            # Draw detected lines on the ROI
            for line in left_h_lines:
                lx1, ly1, lx2, ly2 = line['roi_coords']
                cv2.line(debug_img_left, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 255, 255), 2)
            left_roi_path = f"{output_dir}/{base_name}_M{m_id}_{m_text}_left_hroi_class.png"
            cv2.imwrite(left_roi_path, debug_img_left)

        # If no lines found, try arrow detection
        if not left_h_lines:
            has_left_arrow, _ = detect_arrow_in_roi(h_left_roi, 'left', h_left_x1, h_left_y1, h_left_x2, h_left_y2,
                                                    all_measurements, exclude_items, measurement)
            if has_left_arrow:
                print(f"        Arrow detection found left arrow")

        if left_h_lines:
            for line in left_h_lines:
                lx1, ly1, lx2, ly2 = line['image_coords']
                angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                print(f"        Raw line: angle={angle:.1f}°")
    else:
        left_h_lines = []
        print(f"      Left H-ROI: Invalid bounds")

    h_right_M_inv = None
    if h_right_x2 > h_right_x1 and h_right_y2 > h_right_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Right H-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            h_right_roi, h_right_M_inv = extract_rotated_roi(image, h_right_x1, h_right_y1, h_right_x2, h_right_y2,
                                                              roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            h_right_roi = image[h_right_y1:h_right_y2, h_right_x1:h_right_x2]

        if DEBUG_MODE:
            print(f"      ACTUAL Right H-ROI coords: x=[{h_right_x1}, {h_right_x2}], y=[{h_right_y1}, {h_right_y2}]")

        right_h_lines = detect_lines_in_roi(h_right_roi, h_right_x1, h_right_y1, h_right_M_inv,
                                            all_measurements=all_measurements, exclude_items=exclude_items,
                                            measurement=measurement, roi_x1=h_right_x1, roi_y1=h_right_y1,
                                            roi_x2=h_right_x2, roi_y2=h_right_y2)
        print(f"      Right H-ROI: shape={h_right_roi.shape}, found {len(right_h_lines)} lines")

        # Save debug image of right H-ROI with detected lines
        if output_dir is not None:
            debug_img_right = h_right_roi.copy()
            # Draw detected lines on the ROI
            for line in right_h_lines:
                lx1, ly1, lx2, ly2 = line['roi_coords']
                cv2.line(debug_img_right, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 255, 255), 2)
            right_roi_path = f"{output_dir}/{base_name}_M{m_id}_{m_text}_right_hroi_class.png"
            cv2.imwrite(right_roi_path, debug_img_right)

        # If no lines found, try arrow detection
        if not right_h_lines:
            has_right_arrow, _ = detect_arrow_in_roi(h_right_roi, 'right', h_right_x1, h_right_y1, h_right_x2, h_right_y2,
                                                     all_measurements, exclude_items, measurement)
            if has_right_arrow:
                print(f"        Arrow detection found right arrow")

        if right_h_lines:
            for line in right_h_lines:
                lx1, ly1, lx2, ly2 = line['image_coords']
                angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                print(f"        Raw line: angle={angle:.1f}°")
    else:
        right_h_lines = []
        print(f"      Right H-ROI: Invalid bounds")

    # Tag lines with their ROI source
    for line in left_h_lines:
        line['roi_side'] = 'left'
    for line in right_h_lines:
        line['roi_side'] = 'right'

    # Filter for horizontal lines (more tolerant angles)
    # When ROI is rotated, adjust the expected angle
    for line in left_h_lines + right_h_lines:
        lx1, ly1, lx2, ly2 = line['image_coords']
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
                'adjusted_angle': adjusted_angle,
                'roi_side': line.get('roi_side', 'unknown')
            })

    if horizontal_lines:
        print(f"      Found {len(horizontal_lines)} horizontal line candidates")

    # Search for vertical lines and arrows
    has_up_arrow = False
    has_down_arrow = False

    v_top_M_inv = None
    if v_top_x2 > v_top_x1 and v_top_y2 > v_top_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Top V-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            v_top_roi, v_top_M_inv = extract_rotated_roi(image, v_top_x1, v_top_y1, v_top_x2, v_top_y2,
                                                          roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            v_top_roi = image[v_top_y1:v_top_y2, v_top_x1:v_top_x2]

        top_v_lines = detect_lines_in_roi(v_top_roi, v_top_x1, v_top_y1, v_top_M_inv,
                                          all_measurements=all_measurements, exclude_items=exclude_items,
                                          measurement=measurement, roi_x1=v_top_x1, roi_y1=v_top_y1,
                                          roi_x2=v_top_x2, roi_y2=v_top_y2)
        has_up_arrow, has_wrong_down_arrow = detect_arrow_in_roi(v_top_roi, 'up', v_top_x1, v_top_y1, v_top_x2, v_top_y2,
                                                                 all_measurements, exclude_items, measurement)

        # If wrong-direction arrow detected, ignore v-lines (they're part of that arrow)
        if has_wrong_down_arrow:
            top_v_lines = []
            print(f"      Top V-ROI: shape={v_top_roi.shape}, IGNORING v-lines (DOWN arrow detected)")
        else:
            print(f"      Top V-ROI: shape={v_top_roi.shape}, found {len(top_v_lines)} lines, up-arrow={has_up_arrow}")

        # Save debug image of top V-ROI with detected lines
        if output_dir is not None:
            debug_img_top = v_top_roi.copy()
            # Draw detected lines on the ROI
            for line in top_v_lines:
                lx1, ly1, lx2, ly2 = line['roi_coords']
                cv2.line(debug_img_top, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 255, 255), 2)
            top_roi_path = f"{output_dir}/{base_name}_M{m_id}_{m_text}_top_vroi_class.png"
            cv2.imwrite(top_roi_path, debug_img_top)
    else:
        top_v_lines = []
        print(f"      Top V-ROI: Invalid bounds")

    v_bottom_M_inv = None
    if v_bottom_x2 > v_bottom_x1 and v_bottom_y2 > v_bottom_y1:
        if roi_rotation_angle != 0:
            # Extract rotated ROI - get pixels within rotated rectangle boundaries
            print(f"      Bottom V-ROI: ROTATED by {roi_rotation_angle:.1f}°")
            v_bottom_roi, v_bottom_M_inv = extract_rotated_roi(image, v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2,
                                                                roi_rotation_angle, (x, y))
        else:
            # Normal rectangular extraction
            v_bottom_roi = image[v_bottom_y1:v_bottom_y2, v_bottom_x1:v_bottom_x2]

        bottom_v_lines = detect_lines_in_roi(v_bottom_roi, v_bottom_x1, v_bottom_y1, v_bottom_M_inv,
                                             all_measurements=all_measurements, exclude_items=exclude_items,
                                             measurement=measurement, roi_x1=v_bottom_x1, roi_y1=v_bottom_y1,
                                             roi_x2=v_bottom_x2, roi_y2=v_bottom_y2)
        has_down_arrow, has_wrong_up_arrow = detect_arrow_in_roi(v_bottom_roi, 'down', v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2,
                                                                  all_measurements, exclude_items, measurement)

        # If wrong-direction arrow detected, ignore v-lines (they're part of that arrow)
        if has_wrong_up_arrow:
            bottom_v_lines = []
            print(f"      Bottom V-ROI: shape={v_bottom_roi.shape}, IGNORING v-lines (UP arrow detected)")
        else:
            print(f"      Bottom V-ROI: shape={v_bottom_roi.shape}, found {len(bottom_v_lines)} lines, down-arrow={has_down_arrow}")

        # Save debug image of bottom V-ROI with detected lines
        if output_dir is not None:
            debug_img_bottom = v_bottom_roi.copy()
            # Draw detected lines on the ROI
            for line in bottom_v_lines:
                lx1, ly1, lx2, ly2 = line['roi_coords']
                cv2.line(debug_img_bottom, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 255, 255), 2)
            bottom_roi_path = f"{output_dir}/{base_name}_M{m_id}_{m_text}_bottom_vroi_class.png"
            cv2.imwrite(bottom_roi_path, debug_img_bottom)
    else:
        bottom_v_lines = []
        print(f"      Bottom V-ROI: Invalid bounds")

    # Store original line counts before filtering
    original_top_line_count = len(top_v_lines)
    original_bottom_line_count = len(bottom_v_lines)

    # Filter for vertical lines (more tolerant angles) and track their source
    # When ROI is rotated, adjust the expected angle
    for line in top_v_lines:
        lx1, ly1, lx2, ly2 = line['image_coords']
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
        lx1, ly1, lx2, ly2 = line['image_coords']
        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))

        # Adjust angle for rotation
        adjusted_angle = angle + roi_rotation_angle

        # Normalize to -180 to 180
        while adjusted_angle > 180:
            adjusted_angle -= 360
        while adjusted_angle < -180:
            adjusted_angle += 360

        abs_adjusted_angle = abs(adjusted_angle)
        x_distance = abs(x - (lx1 + lx2) / 2)

        # More tolerant: 55-125° for "generally vertical"
        if 55 < abs_adjusted_angle < 125:
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
            elif DEBUG_MODE:
                print(f"        Bottom V-line FILTERED (x_dist={x_distance:.1f} <= 2): angle={angle:.1f}°")
        elif DEBUG_MODE:
            print(f"        Bottom V-line FILTERED (angle={abs_adjusted_angle:.1f}° not in 55-125°): x_dist={x_distance:.1f}")

    if vertical_lines:
        print(f"      Found {len(vertical_lines)} vertical line candidates")

    # NEW LOGIC: Strength-based comparison when both WIDTH and HEIGHT criteria are met
    # Check BOTH WIDTH and HEIGHT criteria, then compare strength if both match

    # === STEP 1: Check WIDTH criteria ===
    # horizontal_lines already contains only horizontal lines (angle filtered)
    # Check which side of the text center each horizontal line is on
    left_h_lines = [l for l in horizontal_lines if l['coords'][0] < x and l['coords'][2] < x]  # Both endpoints left of center
    right_h_lines = [l for l in horizontal_lines if l['coords'][0] > x and l['coords'][2] > x]  # Both endpoints right of center

    has_left_horizontal = len(left_h_lines) > 0 or has_left_arrow
    has_right_horizontal = len(right_h_lines) > 0 or has_right_arrow

    width_criteria_met = has_left_horizontal and has_right_horizontal
    width_strength = len(left_h_lines) + (1 if has_left_arrow else 0) + len(right_h_lines) + (1 if has_right_arrow else 0)

    # === STEP 2: Check HEIGHT criteria ===
    # Use the 'source' field to determine which ROI each line came from
    top_v_lines = [l for l in vertical_lines if l.get('source') == 'top']
    bottom_v_lines = [l for l in vertical_lines if l.get('source') == 'bottom']

    # Debug: Show what lines we found
    if DEBUG_MODE and len(vertical_lines) > 0:
        print(f"      Vertical lines analysis:")
        print(f"        Lines from TOP ROI: {len(top_v_lines)}")
        print(f"        Lines from BOTTOM ROI: {len(bottom_v_lines)}")

    # Check for vertical indicators - can be either lines OR arrows
    # Top: either up-arrow OR vertical lines
    # Bottom: either down-arrow OR vertical lines
    has_top_vertical = len(top_v_lines) > 0
    has_bottom_vertical = len(bottom_v_lines) > 0

    has_top_indicator = has_up_arrow or has_top_vertical
    has_bottom_indicator = has_down_arrow or has_bottom_vertical

    # RETRY with doubled ROI height if only one side has indicators
    if has_top_indicator and not has_bottom_indicator:
        # Top has indicators but bottom doesn't - retry with doubled bottom ROI height
        print(f"      RETRY: Top has indicators but bottom doesn't - doubling bottom V-ROI height")

        v_bottom_y1_extended = v_bottom_y1
        v_bottom_y2_extended = min(image.shape[0], v_bottom_y1 + (v_bottom_y2 - v_bottom_y1) * 2)

        if v_bottom_x2 > v_bottom_x1 and v_bottom_y2_extended > v_bottom_y1_extended:
            v_bottom_roi_extended = image[v_bottom_y1_extended:v_bottom_y2_extended, v_bottom_x1:v_bottom_x2]
            bottom_v_lines_retry = detect_lines_in_roi(v_bottom_roi_extended, v_bottom_x1, v_bottom_y1_extended, None,
                                                       all_measurements=all_measurements, exclude_items=exclude_items,
                                                       measurement=measurement, roi_x1=v_bottom_x1, roi_y1=v_bottom_y1_extended,
                                                       roi_x2=v_bottom_x2, roi_y2=v_bottom_y2_extended)
            has_down_arrow_retry, has_wrong_up_arrow_retry = detect_arrow_in_roi(v_bottom_roi_extended, 'down', v_bottom_x1, v_bottom_y1_extended, v_bottom_x2, v_bottom_y2_extended,
                                                                              all_measurements, exclude_items, measurement)

            if has_wrong_up_arrow_retry:
                bottom_v_lines_retry = []

            # Filter and add retry lines
            for line in bottom_v_lines_retry:
                lx1, ly1, lx2, ly2 = line['image_coords']
                angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                adjusted_angle = angle + roi_rotation_angle
                while adjusted_angle > 180: adjusted_angle -= 360
                while adjusted_angle < -180: adjusted_angle += 360
                abs_adjusted_angle = abs(adjusted_angle)
                x_distance = abs(x - (lx1 + lx2) / 2)

                if 55 < abs_adjusted_angle < 125 and x_distance > 2:
                    vertical_lines.append({
                        'coords': (lx1, ly1, lx2, ly2),
                        'distance': x_distance,
                        'type': 'vertical_line',
                        'source': 'bottom',
                        'angle': angle,
                        'adjusted_angle': adjusted_angle
                    })
                    bottom_v_lines.append({'coords': (lx1, ly1, lx2, ly2), 'distance': x_distance})
                    if DEBUG_MODE:
                        print(f"        RETRY Bottom V-line: angle={angle:.1f}°, x_dist={x_distance:.1f}")

            has_down_arrow = has_down_arrow or has_down_arrow_retry
            has_bottom_vertical = len(bottom_v_lines) > 0
            has_bottom_indicator = has_down_arrow or has_bottom_vertical

            if has_bottom_indicator:
                print(f"      RETRY SUCCESS: Found bottom indicators with doubled ROI")

    elif has_bottom_indicator and not has_top_indicator:
        # Bottom has indicators but top doesn't - retry with doubled top ROI height
        print(f"      RETRY: Bottom has indicators but top doesn't - doubling top V-ROI height")

        v_top_y1_extended = max(0, v_top_y2 - (v_top_y2 - v_top_y1) * 2)
        v_top_y2_extended = v_top_y2

        if v_top_x2 > v_top_x1 and v_top_y2_extended > v_top_y1_extended:
            v_top_roi_extended = image[v_top_y1_extended:v_top_y2_extended, v_top_x1:v_top_x2]
            top_v_lines_retry = detect_lines_in_roi(v_top_roi_extended, v_top_x1, v_top_y1_extended, None,
                                                    all_measurements=all_measurements, exclude_items=exclude_items,
                                                    measurement=measurement, roi_x1=v_top_x1, roi_y1=v_top_y1_extended,
                                                    roi_x2=v_top_x2, roi_y2=v_top_y2_extended)
            has_up_arrow_retry, has_wrong_down_arrow_retry = detect_arrow_in_roi(v_top_roi_extended, 'up', v_top_x1, v_top_y1_extended, v_top_x2, v_top_y2_extended,
                                                                              all_measurements, exclude_items, measurement)

            if has_wrong_down_arrow_retry:
                top_v_lines_retry = []

            # Filter and add retry lines
            for line in top_v_lines_retry:
                lx1, ly1, lx2, ly2 = line['image_coords']
                angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                adjusted_angle = angle + roi_rotation_angle
                while adjusted_angle > 180: adjusted_angle -= 360
                while adjusted_angle < -180: adjusted_angle += 360
                abs_adjusted_angle = abs(adjusted_angle)
                x_distance = abs(x - (lx1 + lx2) / 2)

                if 55 < abs_adjusted_angle < 125 and x_distance > 2:
                    vertical_lines.append({
                        'coords': (lx1, ly1, lx2, ly2),
                        'distance': x_distance,
                        'type': 'vertical_line',
                        'source': 'top',
                        'angle': angle,
                        'adjusted_angle': adjusted_angle
                    })
                    top_v_lines.append({'coords': (lx1, ly1, lx2, ly2), 'distance': x_distance})
                    if DEBUG_MODE:
                        print(f"        RETRY Top V-line: angle={angle:.1f}°, x_dist={x_distance:.1f}")

            has_up_arrow = has_up_arrow or has_up_arrow_retry
            has_top_vertical = len(top_v_lines) > 0
            has_top_indicator = has_up_arrow or has_top_vertical

            if has_top_indicator:
                print(f"      RETRY SUCCESS: Found top indicators with doubled ROI")

    # Special case: For small measurements (< 8 inches), allow HEIGHT with just ONE arrow
    # Parse measurement value to check if it's under 8 inches
    measurement_text = measurement.get('text', '')
    measurement_value_inches = None
    try:
        # Parse measurement like "4 1/2", "9 1/4", etc.
        parts = measurement_text.split()
        if len(parts) >= 1:
            whole = 0
            fraction = 0

            # Check if first part is a whole number or fraction
            if '/' in parts[0]:
                # Just a fraction like "1/2"
                frac_parts = parts[0].split('/')
                fraction = float(frac_parts[0]) / float(frac_parts[1])
            else:
                # Whole number
                whole = float(parts[0])

            # Check if there's a fraction part
            if len(parts) >= 2 and '/' in parts[1]:
                frac_parts = parts[1].split('/')
                fraction = float(frac_parts[0]) / float(frac_parts[1])

            measurement_value_inches = whole + fraction
    except:
        pass

    # HEIGHT criteria: Either arrow-based OR v-line-based
    #
    # Arrow-based path (NEW): If ANY correct-direction arrow detected (up arrow in top OR down arrow in bottom)
    # This works for any size measurement and allows partial indicators
    arrow_based_height = (has_up_arrow or has_down_arrow)

    # V-line-based path (ORIGINAL): For measurements >= 8", need indicators on both sides
    is_small_measurement = measurement_value_inches is not None and measurement_value_inches < 8.0
    if is_small_measurement:
        # Small measurements already covered by arrow_based_height
        vline_based_height = False
    else:
        # Normal measurements: need indicators on BOTH top AND bottom
        vline_based_height = has_top_indicator and has_bottom_indicator

    # Qualify as HEIGHT if EITHER path succeeds
    height_criteria_met = arrow_based_height or vline_based_height

    height_strength = len(top_v_lines) + (1 if has_up_arrow else 0) + len(bottom_v_lines) + (1 if has_down_arrow else 0)

    # === STEP 3: Compare and decide ===
    if width_criteria_met and height_criteria_met:
        # BOTH WIDTH and HEIGHT criteria are met - compare strength
        print(f"      CONFLICT: Both WIDTH and HEIGHT criteria met")
        print(f"        WIDTH strength: {width_strength} (left:{len(left_h_lines)}+arrow:{1 if has_left_arrow else 0} + right:{len(right_h_lines)}+arrow:{1 if has_right_arrow else 0})")
        print(f"        HEIGHT strength: {height_strength} (top:{len(top_v_lines)}+arrow:{1 if has_up_arrow else 0} + bottom:{len(bottom_v_lines)}+arrow:{1 if has_down_arrow else 0})")

        if height_strength > width_strength:
            print(f"      → HEIGHT wins ({height_strength} > {width_strength})")
            classify_as = 'HEIGHT'
        elif width_strength > height_strength:
            print(f"      → WIDTH wins ({width_strength} > {height_strength})")
            classify_as = 'WIDTH'
        else:
            # Tie - default to WIDTH (dimension lines are usually more reliable than arrows)
            print(f"      → TIE ({width_strength} == {height_strength}), defaulting to WIDTH")
            classify_as = 'WIDTH'
    elif width_criteria_met:
        # Only WIDTH criteria met
        classify_as = 'WIDTH'
        arrow_msg = ""
        if has_left_arrow:
            arrow_msg += " (left arrow)"
        if has_right_arrow:
            arrow_msg += " (right arrow)"
        print(f"      Found HORIZONTAL lines/arrows on BOTH left and right{arrow_msg} → WIDTH")
    elif height_criteria_met:
        # Only HEIGHT criteria met
        classify_as = 'HEIGHT'
        top_desc = []
        if has_up_arrow:
            top_desc.append("up-arrow")
        if has_top_vertical:
            top_desc.append(f"{len(top_v_lines)} v-lines")
        bottom_desc = []
        if has_down_arrow:
            bottom_desc.append("down-arrow")
        if has_bottom_vertical:
            bottom_desc.append(f"{len(bottom_v_lines)} v-lines")
        print(f"      Found vertical indicators: top({'+'.join(top_desc)}) + bottom({'+'.join(bottom_desc)}) → HEIGHT")
    elif not has_left_horizontal and not has_right_horizontal and (has_top_vertical or has_bottom_vertical):
        # Fallback rule: If no H-lines but V-lines exist, and measurement < 13", classify as HEIGHT
        if measurement_value_inches is not None and measurement_value_inches < 13.0:
            classify_as = 'HEIGHT'
            v_side = 'top' if has_top_vertical else 'bottom'
            v_count = len(top_v_lines) if has_top_vertical else len(bottom_v_lines)
            print(f"      Fallback: Small measurement ({measurement_value_inches:.2f}\" < 13\") with {v_count} v-lines ({v_side}) but no h-lines → HEIGHT")
        else:
            # No h-lines but has v-lines, but measurement >= 13" - UNCLASSIFIED
            classify_as = None
            meas_size = f"{measurement_value_inches:.2f}\"" if measurement_value_inches else "unknown"
            print(f"      No h-lines on both sides, has v-lines but measurement {meas_size} >= 13\" → UNCLASSIFIED")
    else:
        # Neither criteria met - UNCLASSIFIED
        print(f"      No lines on both sides (L-horiz:{has_left_horizontal} R-horiz:{has_right_horizontal} T-vert:{has_top_vertical} B-vert:{has_bottom_vertical}) → UNCLASSIFIED")
        classify_as = None

    # === STEP 4: Return appropriate classification ===
    if classify_as == 'WIDTH':
        best_h = min(horizontal_lines, key=lambda l: l['distance']) if horizontal_lines else None

        # Store all horizontal lines for pairing logic (to extend lines and check for heights)
        measurement['h_lines'] = horizontal_lines
        measurement['v_lines'] = vertical_lines  # Store v-lines for visualization

        # Use the h-line furthest down (highest Y coordinate) - the bottom dimension line edge
        # Filter for nearly-horizontal lines first to exclude arrow lines
        if horizontal_lines:
            horizontal_threshold = 8.0  # degrees - exclude steep arrow lines
            nearly_horizontal = [l for l in horizontal_lines
                                if abs(l['angle']) <= horizontal_threshold]

            if nearly_horizontal:
                # Use the furthest-down nearly-horizontal line (true extent line)
                furthest_down_line = max(nearly_horizontal,
                                        key=lambda l: (l['coords'][1] + l['coords'][3]) / 2)
                measurement['h_line_angle'] = furthest_down_line['angle']
                print(f"    Selected h-line angle {furthest_down_line['angle']:.1f}° from {len(nearly_horizontal)} nearly-horizontal lines (threshold ±{horizontal_threshold}°)")
            else:
                # Fallback: no nearly-horizontal lines, use closest to horizontal
                furthest_down_line = min(horizontal_lines,
                                        key=lambda l: abs(l['angle']))
                measurement['h_line_angle'] = furthest_down_line['angle']
                print(f"    WARNING: No nearly-horizontal lines found, using closest to horizontal: {furthest_down_line['angle']:.1f}°")

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

    elif classify_as == 'HEIGHT':
        # Store all lines for visualization
        measurement['h_lines'] = horizontal_lines
        measurement['v_lines'] = vertical_lines

        # Use vertical lines if available, otherwise create placeholder
        if vertical_lines:
            best_v = min(vertical_lines, key=lambda l: l['distance'])
            # Calculate average angle of vertical lines for intersection calculation
            avg_angle = sum(l['angle'] for l in vertical_lines) / len(vertical_lines)
            measurement['v_line_angle'] = avg_angle

            return {
                'line': best_v['coords'],
                'orientation': 'vertical_line',
                'distance': best_v['distance'],
                'angle': best_v.get('angle', 90),
                'arrow_based': not (has_top_vertical and has_bottom_vertical)
            }
        else:
            # No vertical lines but arrows detected - use center position
            return {
                'line': (x, y-20, x, y+20),
                'orientation': 'vertical_line',
                'distance': 0,
                'arrow_based': True
            }

    # UNCLASSIFIED

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


def check_for_arrows_at_line_ends(roi_image, roi_x_offset, roi_y_offset, line_coords, direction, inverse_transform=None):
    """
    Check if there are arrow indicators in the search area ROI.

    Args:
        roi_image: The FULL search area ROI image (left or right ROI)
        roi_x_offset, roi_y_offset: ROI offsets relative to full image
        line_coords: (not used, kept for compatibility)
        direction: 'left' or 'right' - which side we're searching
        inverse_transform: Optional 3x3 perspective transform matrix for rotated ROIs

    Returns:
        (has_arrow, arrow_x): Tuple of (True if arrows found, X position of arrow tip in ROI coords)
        If no arrow found, returns (False, None)
    """
    # Search the ENTIRE ROI for arrows, not just line endpoints
    # The roi_image passed in IS the search area (left or right ROI)

    if roi_image is None or roi_image.size == 0:
        return (False, None)

    if direction in ['left', 'right']:
        # Detect lines in the FULL ROI
        lines = detect_lines_in_roi(roi_image, roi_x_offset, roi_y_offset, inverse_transform) if roi_image.size > 0 else []

        # Need at least 2 lines to form an arrow
        if len(lines) < 2:
            return (False, None)

        # Check for converging line pairs (arrows)
        # For LEFT/RIGHT arrows, check that lines converge HORIZONTALLY, not vertically
        for i, line1 in enumerate(lines):
            lx1a, ly1a, lx2a, ly2a = line1['roi_coords']
            angle1 = np.degrees(np.arctan2(ly2a - ly1a, lx2a - lx1a))

            for j, line2 in enumerate(lines[i+1:], start=i+1):
                lx1b, ly1b, lx2b, ly2b = line2['roi_coords']
                angle2 = np.degrees(np.arctan2(ly2b - ly1b, lx2b - lx1b))

                # Check if angles are opposite (forming V pattern)
                angle_diff = abs(angle1 - angle2)
                # Normalize to 0-180 range
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                # Lines should be converging (15-170 degrees apart)
                # Broadened from 90-170 to catch LEFT/RIGHT arrows in rotated ROIs
                if 15 < angle_diff < 170:
                    # Check if this is a HORIZONTAL arrow (</>) or VERTICAL arrow (^/v)
                    # by comparing deviation from horizontal vs vertical

                    # For each line, calculate:
                    # - Deviation from horizontal: min distance to 0° or 180°
                    # - Deviation from vertical: distance to ±90°

                    abs_angle1 = abs(angle1)
                    abs_angle2 = abs(angle2)

                    # Deviation from horizontal (0° or 180°)
                    dev_horiz_1 = min(abs_angle1, abs(180 - abs_angle1))
                    dev_horiz_2 = min(abs_angle2, abs(180 - abs_angle2))

                    # Deviation from vertical (90° or -90°)
                    dev_vert_1 = abs(abs_angle1 - 90)
                    dev_vert_2 = abs(abs_angle2 - 90)

                    # Check if EITHER line is very close to vertical (±90°)
                    # True LEFT/RIGHT arrows have both lines at reasonable angles from horizontal
                    # Vertical arrows (^ v) will have at least one line very close to ±90°
                    line1_near_vertical = dev_vert_1 < 20  # Within 20° of ±90°
                    line2_near_vertical = dev_vert_2 < 20  # Within 20° of ±90°

                    if line1_near_vertical or line2_near_vertical:
                        # At least one line is very close to vertical - this is a vertical arrow
                        # Skip this pair when searching for horizontal arrows
                        print(f"        [ARROW FILTER] Rejected near-vertical arrow: angles {angle1:.1f}° and {angle2:.1f}°")
                        print(f"        [ARROW FILTER]   Line1: vert_dev={dev_vert_1:.1f}° → {'NEAR_VERT' if line1_near_vertical else 'OK'}")
                        print(f"        [ARROW FILTER]   Line2: vert_dev={dev_vert_2:.1f}° → {'NEAR_VERT' if line2_near_vertical else 'OK'}")
                        continue

                    # === ARROW VALIDATION: Line Length and Convergence ===
                    # Calculate line lengths
                    length1 = np.sqrt((lx2a - lx1a)**2 + (ly2a - ly1a)**2)
                    length2 = np.sqrt((lx2b - lx1b)**2 + (ly2b - ly1b)**2)

                    # Filter 1: Minimum length (reject noise/text fragments)
                    MIN_ARROW_LENGTH = 12  # pixels
                    if length1 < MIN_ARROW_LENGTH or length2 < MIN_ARROW_LENGTH:
                        print(f"        [ARROW FILTER] Rejected short lines: len1={length1:.1f}px, len2={length2:.1f}px (min={MIN_ARROW_LENGTH}px)")
                        continue

                    # Filter 2: Maximum length (reject long dimension lines)
                    MAX_ARROW_LENGTH = 40  # pixels
                    if length1 > MAX_ARROW_LENGTH or length2 > MAX_ARROW_LENGTH:
                        print(f"        [ARROW FILTER] Rejected long lines: len1={length1:.1f}px, len2={length2:.1f}px (max={MAX_ARROW_LENGTH}px)")
                        continue

                    # Filter 3: Length ratio (both lines should be similar length)
                    min_length = min(length1, length2)
                    max_length = max(length1, length2)
                    length_ratio = min_length / max_length if max_length > 0 else 0
                    MIN_LENGTH_RATIO = 0.5  # Lines should be within 2x of each other
                    if length_ratio < MIN_LENGTH_RATIO:
                        print(f"        [ARROW FILTER] Rejected dissimilar lengths: len1={length1:.1f}px, len2={length2:.1f}px, ratio={length_ratio:.2f} (min={MIN_LENGTH_RATIO})")
                        continue

                    # Filter 4: Convergence distance (lines must meet at a point)
                    # Calculate intersection point of the two lines
                    dx1 = lx2a - lx1a
                    dy1 = ly2a - ly1a
                    dx2 = lx2b - lx1b
                    dy2 = ly2b - ly1b

                    # Check if lines are parallel (determinant near zero)
                    det = dx1 * dy2 - dy1 * dx2
                    if abs(det) < 0.01:
                        # Lines are parallel or nearly parallel
                        print(f"        [ARROW FILTER] Rejected parallel lines: angles {angle1:.1f}° and {angle2:.1f}°")
                        continue

                    # Calculate intersection point
                    t = ((lx1b - lx1a) * dy2 - (ly1b - ly1a) * dx2) / det
                    intersect_x = lx1a + t * dx1
                    intersect_y = ly1a + t * dy1

                    # Calculate distance from line endpoints to intersection (convergence distance)
                    # For a true arrow, the endpoints should be close to the intersection
                    d1a = np.sqrt((intersect_x - lx1a)**2 + (intersect_y - ly1a)**2)
                    d1b = np.sqrt((intersect_x - lx2a)**2 + (intersect_y - ly2a)**2)
                    d2a = np.sqrt((intersect_x - lx1b)**2 + (intersect_y - ly1b)**2)
                    d2b = np.sqrt((intersect_x - lx2b)**2 + (intersect_y - ly2b)**2)

                    # Use the minimum distance from each line (closest endpoint)
                    conv_dist_1 = min(d1a, d1b)
                    conv_dist_2 = min(d2a, d2b)
                    conv_dist = max(conv_dist_1, conv_dist_2)  # Both lines should be close

                    MAX_CONVERGENCE_DIST = 15  # pixels
                    if conv_dist > MAX_CONVERGENCE_DIST:
                        print(f"        [ARROW FILTER] Rejected non-converging lines: conv_dist={conv_dist:.1f}px (max={MAX_CONVERGENCE_DIST}px)")
                        print(f"        [ARROW FILTER]   Line1 closest to intersection: {conv_dist_1:.1f}px")
                        print(f"        [ARROW FILTER]   Line2 closest to intersection: {conv_dist_2:.1f}px")
                        continue

                    # Filter 5: Directional validation - ensure arrow points the correct way
                    # Normalize angles to 0-360° range
                    norm_angle1 = angle1 % 360
                    if norm_angle1 < 0:
                        norm_angle1 += 360
                    norm_angle2 = angle2 % 360
                    if norm_angle2 < 0:
                        norm_angle2 += 360

                    # Check if angles match expected arrow direction
                    if direction == 'left':
                        # LEFT arrow (<): requires top leg 120-165° and bottom leg 195-240°
                        has_top_leg = (120 <= norm_angle1 <= 165) or (120 <= norm_angle2 <= 165)
                        has_bottom_leg = (195 <= norm_angle1 <= 240) or (195 <= norm_angle2 <= 240)

                        if not (has_top_leg and has_bottom_leg):
                            print(f"        [ARROW FILTER] Rejected: angles {norm_angle1:.1f}° and {norm_angle2:.1f}° don't form LEFT arrow (<)")
                            print(f"        [ARROW FILTER]   Need: top leg 120-165° AND bottom leg 195-240°")
                            continue

                    elif direction == 'right':
                        # RIGHT arrow (>): requires top leg 300-345° or 0-45° and bottom leg 15-60°
                        # Top leg can wrap around 0° (315° to 45° range)
                        has_top_leg = (300 <= norm_angle1 <= 345) or (0 <= norm_angle1 <= 45) or \
                                      (300 <= norm_angle2 <= 345) or (0 <= norm_angle2 <= 45)
                        has_bottom_leg = (15 <= norm_angle1 <= 60) or (15 <= norm_angle2 <= 60)

                        if not (has_top_leg and has_bottom_leg):
                            print(f"        [ARROW FILTER] Rejected: angles {norm_angle1:.1f}° and {norm_angle2:.1f}° don't form RIGHT arrow (>)")
                            print(f"        [ARROW FILTER]   Need: top leg 300-45° AND bottom leg 15-60°")
                            continue

                    # Valid horizontal arrow (< or >) - passed all filters
                    print(f"        [ARROW DETECTED] {direction.upper()} arrow found: angles {angle1:.1f}° and {angle2:.1f}°")
                    print(f"        [ARROW ANGLES] Line1: horiz_dev={dev_horiz_1:.1f}°, vert_dev={dev_vert_1:.1f}°")
                    print(f"        [ARROW ANGLES] Line2: horiz_dev={dev_horiz_2:.1f}°, vert_dev={dev_vert_2:.1f}°")
                    print(f"        [ARROW VALIDATION] len1={length1:.1f}px, len2={length2:.1f}px, ratio={length_ratio:.2f}, conv_dist={conv_dist:.1f}px")

                    # Calculate the arrow tip position (convergence point of the two lines)
                    # For left arrow: use the leftmost X coordinate
                    # For right arrow: use the rightmost X coordinate
                    all_x = [lx1a, lx2a, lx1b, lx2b]
                    if direction == 'left':
                        arrow_x = min(all_x)  # Leftmost point is the arrow tip
                    else:  # right
                        arrow_x = max(all_x)  # Rightmost point is the arrow tip

                    return (True, arrow_x)

        return (False, None)

    elif direction in ['up', 'down']:
        # Detect lines in the FULL ROI
        lines = detect_lines_in_roi(roi_image, roi_x_offset, roi_y_offset, inverse_transform) if roi_image.size > 0 else []
        # For up/down arrows, just check if we have at least 2 lines
        return len(lines) >= 2

    return False


def calculate_rotated_corners(x1, y1, x2, y2, angle_degrees, center_point):
    """Calculate the 4 corners of a rotated rectangle.

    Args:
        x1, y1, x2, y2: Bounds of the axis-aligned rectangle
        angle_degrees: Rotation angle in degrees
        center_point: (cx, cy) center point for rotation

    Returns:
        List of 4 corner points [(x,y), ...] in clockwise order from top-left
    """
    # Define the 4 corners of the rectangle
    corners = np.array([
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2]   # Bottom-left
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
        rotated_corners.append((px_rot + cx, py_rot + cy))

    return rotated_corners


def extend_roi_and_get_full_extent(image, measurement, category, image_path=None):
    """
    After classification, search extended ROIs for arrows to find full line extent.
    Creates large search areas and scans for arrow patterns.

    Args:
        image: The full image
        measurement: The measurement dict with ROI info
        category: 'width' or 'height'
        image_path: Optional path to the image file (for debug output naming)

    Returns:
        Extent info dict with left/right (for width) or top/bottom (for height)
    """
    print(f"    Searching extended ROIs for arrows to calculate {category.upper()} extent...")

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

    # Store both angles for two-pass search
    classification_roi_angle = measurement.get('roi_rotation_angle', 0)  # Angle used in classification
    h_line_angle = measurement.get('h_line_angle', 0)  # Detected line angle (may be arrow line)
    print(f"      [EXTENT DEBUG] classification_roi_angle (roi_rotation_angle) = {classification_roi_angle:.1f}°")
    print(f"      [EXTENT DEBUG] h_line_angle = {h_line_angle:.1f}°")

    h_roi_height_multiplier = 1.0
    v_roi_width_multiplier = 1.0
    min_gap = 15

    if category == 'width':
        # Create extended search ROIs to look for arrows
        # Extend H-ROI maximum 2 inches (192px at 96 DPI) on each side for arrow search
        max_extension = 192  # 2 inches at 96 DPI

        # Calculate ROI bounds for arrow search zones
        # These will be used for visualization and arrow detection
        h_left_x2 = max(0, text_left - min_gap)
        h_left_x1 = max(0, h_left_x2 - max_extension)
        h_left_y1 = int(y - text_height * h_roi_height_multiplier)
        h_left_y2 = int(y + text_height * h_roi_height_multiplier)

        h_right_x1 = min(image.shape[1], text_right + min_gap)
        h_right_x2 = min(image.shape[1], h_right_x1 + max_extension)
        h_right_y1 = h_left_y1
        h_right_y2 = h_left_y2

        print(f"      Arrow search Left H-ROI: x=[{h_left_x1}, {h_left_x2}] (width: {h_left_x2-h_left_x1}px)")
        print(f"      Arrow search Right H-ROI: x=[{h_right_x1}, {h_right_x2}] (width: {h_right_x2-h_right_x1}px)")

        # === PASS 1: Try classification ROI angle first ===
        print(f"      [PASS 1] Searching with classification ROI angle ({classification_roi_angle:.1f}°)...")
        roi_rotation_angle_pass1 = classification_roi_angle
        h_left_M_inv_pass1 = None
        h_right_M_inv_pass1 = None
        if roi_rotation_angle_pass1 != 0:
            h_left_roi_pass1, h_left_M_inv_pass1 = extract_rotated_roi(image, h_left_x1, h_left_y1, h_left_x2, h_left_y2, roi_rotation_angle_pass1, (x, y))
            h_right_roi_pass1, h_right_M_inv_pass1 = extract_rotated_roi(image, h_right_x1, h_right_y1, h_right_x2, h_right_y2, roi_rotation_angle_pass1, (x, y))
        else:
            h_left_roi_pass1 = image[h_left_y1:h_left_y2, h_left_x1:h_left_x2]
            h_right_roi_pass1 = image[h_right_y1:h_right_y2, h_right_x1:h_right_x2]

        # Search for arrows in Pass 1 ROIs
        has_left_arrows_pass1, left_arrow_x_pass1 = check_for_arrows_at_line_ends(h_left_roi_pass1, h_left_x1, h_left_y1, None, 'left', h_left_M_inv_pass1)
        has_right_arrows_pass1, right_arrow_x_pass1 = check_for_arrows_at_line_ends(h_right_roi_pass1, h_right_x1, h_right_y1, None, 'right', h_right_M_inv_pass1)

        print(f"        Pass 1 Left ROI: arrows={has_left_arrows_pass1}, position={left_arrow_x_pass1}")
        print(f"        Pass 1 Right ROI: arrows={has_right_arrows_pass1}, position={right_arrow_x_pass1}")

        # If Pass 1 found arrows, use those results
        if has_left_arrows_pass1 or has_right_arrows_pass1:
            print(f"      [PASS 1 SUCCESS] Found arrows using classification angle, processing extent...")

            # Use Pass 1 results - set these as the main results
            roi_rotation_angle = roi_rotation_angle_pass1
            h_left_roi = h_left_roi_pass1
            h_right_roi = h_right_roi_pass1
            h_left_M_inv = h_left_M_inv_pass1
            h_right_M_inv = h_right_M_inv_pass1
            has_left_arrows = has_left_arrows_pass1
            left_arrow_x = left_arrow_x_pass1
            has_right_arrows = has_right_arrows_pass1
            right_arrow_x = right_arrow_x_pass1

            # Save Pass 1 debug images
            if image_path:
                import os
                output_dir = os.path.dirname(os.path.abspath(image_path))
                base_name = os.path.splitext(os.path.basename(image_path))[0]
            else:
                output_dir = "/tmp"
                base_name = "debug"
            m_id = measurement.get('id', 'unknown')
            m_text = measurement.get('text', 'unknown').replace('/', '-').replace(' ', '')
            left_roi_path = f"{output_dir}/{base_name}_M{m_id}_{m_text}_extent_left_roi_pass1.png"
            right_roi_path = f"{output_dir}/{base_name}_M{m_id}_{m_text}_extent_right_roi_pass1.png"
            cv2.imwrite(left_roi_path, h_left_roi)
            cv2.imwrite(right_roi_path, h_right_roi)
            print(f"      [DEBUG] Saved Pass 1 ROI images: {base_name}_M{m_id}_{m_text}_extent_left_roi_pass1.png, {base_name}_M{m_id}_{m_text}_extent_right_roi_pass1.png")

            # Skip Pass 2 and go directly to extent calculation
        else:
            print(f"      [PASS 1 FAILED] No arrows found with classification angle")
            print(f"      [PASS 2] Trying with h-line angle ({h_line_angle:.1f}°)...")

            # Pass 2: Use h-line angle (current behavior)
            roi_rotation_angle = h_line_angle

        # Extract ROIs for arrow detection (Pass 2 or if Pass 1 failed)
        if not (has_left_arrows_pass1 or has_right_arrows_pass1):
            h_left_M_inv = None
            h_right_M_inv = None
            if roi_rotation_angle != 0:
                h_left_roi, h_left_M_inv = extract_rotated_roi(image, h_left_x1, h_left_y1, h_left_x2, h_left_y2, roi_rotation_angle, (x, y))
                h_right_roi, h_right_M_inv = extract_rotated_roi(image, h_right_x1, h_right_y1, h_right_x2, h_right_y2, roi_rotation_angle, (x, y))
            else:
                h_left_roi = image[h_left_y1:h_left_y2, h_left_x1:h_left_x2]
                h_right_roi = image[h_right_y1:h_right_y2, h_right_x1:h_right_x2]

            # Save Pass 2 debug images of ROIs for inspection
            if image_path:
                import os
                output_dir = os.path.dirname(os.path.abspath(image_path))
                base_name = os.path.splitext(os.path.basename(image_path))[0]
            else:
                output_dir = "/tmp"
                base_name = "debug"

            # Use measurement ID (assigned in Phase 2)
            m_id = measurement.get('id', 'unknown')
            m_text = measurement.get('text', 'unknown').replace('/', '-').replace(' ', '')
            left_roi_path = f"{output_dir}/{base_name}_M{m_id}_{m_text}_extent_left_roi_pass2.png"
            right_roi_path = f"{output_dir}/{base_name}_M{m_id}_{m_text}_extent_right_roi_pass2.png"
            cv2.imwrite(left_roi_path, h_left_roi)
            cv2.imwrite(right_roi_path, h_right_roi)
            print(f"      [DEBUG] Saved Pass 2 ROI images: {base_name}_M{m_id}_{m_text}_extent_left_roi_pass2.png, {base_name}_M{m_id}_{m_text}_extent_right_roi_pass2.png")

            # Search the FULL extended ROIs for arrow patterns (Pass 2)
            print(f"      Searching complete ROIs for arrow patterns (Pass 2)...")

            # Search each ROI directly for arrows
            has_left_arrows, left_arrow_x = check_for_arrows_at_line_ends(h_left_roi, h_left_x1, h_left_y1, None, 'left', h_left_M_inv)
            has_right_arrows, right_arrow_x = check_for_arrows_at_line_ends(h_right_roi, h_right_x1, h_right_y1, None, 'right', h_right_M_inv)

            print(f"        Pass 2 Left ROI: arrows={has_left_arrows}, position={left_arrow_x}")
            print(f"        Pass 2 Right ROI: arrows={has_right_arrows}, position={right_arrow_x}")

        # Use the angle from whichever pass succeeded (roi_rotation_angle is set by Pass 1 or Pass 2)
        extent_angle = roi_rotation_angle
        print(f"      Using extent angle: {extent_angle:.1f}° from {'Pass 1' if has_left_arrows_pass1 or has_right_arrows_pass1 else 'Pass 2'}")

        # Extend from measurement center (x, y) to arrow positions at extent_angle
        # This ensures both sides follow the same angle consistently
        if has_left_arrows and has_right_arrows:
            # Convert ROI coords to image coords using inverse transform
            if h_left_M_inv is not None:
                # Rotated ROI - use inverse transform to map arrow position back to image coords
                roi_center_y = (h_left_y2 - h_left_y1) / 2  # Middle of ROI height
                arrow_pt = np.array([[[left_arrow_x, roi_center_y]]], dtype=np.float32)
                arrow_img = cv2.perspectiveTransform(arrow_pt, h_left_M_inv)
                extent_left_x, extent_left_y = arrow_img[0][0]
                print(f"        [TRANSFORM] Left arrow: ROI({left_arrow_x:.0f}, {roi_center_y:.0f}) → Image({extent_left_x:.0f}, {extent_left_y:.0f})")
            else:
                # Non-rotated ROI - simple offset
                extent_left_x = h_left_x1 + left_arrow_x
                extent_left_y = y

            if h_right_M_inv is not None:
                # Rotated ROI - use inverse transform to map arrow position back to image coords
                roi_center_y = (h_right_y2 - h_right_y1) / 2  # Middle of ROI height
                arrow_pt = np.array([[[right_arrow_x, roi_center_y]]], dtype=np.float32)
                arrow_img = cv2.perspectiveTransform(arrow_pt, h_right_M_inv)
                extent_right_x, extent_right_y = arrow_img[0][0]
                print(f"        [TRANSFORM] Right arrow: ROI({right_arrow_x:.0f}, {roi_center_y:.0f}) → Image({extent_right_x:.0f}, {extent_right_y:.0f})")
            else:
                # Non-rotated ROI - simple offset
                extent_right_x = h_right_x1 + right_arrow_x
                extent_right_y = y

            # Extend from measurement center (x, y) to arrow positions at extent_angle
            dx_left = extent_left_x - x
            dy_left = dx_left * np.tan(np.radians(extent_angle))
            extent_left_y = y + dy_left

            dx_right = extent_right_x - x
            dy_right = dx_right * np.tan(np.radians(extent_angle))
            extent_right_y = y + dy_right

            # Store debug_rois with rotated corners if ROI is rotated
            if roi_rotation_angle != 0:
                # Calculate rotated corners for visualization
                left_corners = calculate_rotated_corners(h_left_x1, h_left_y1, h_left_x2, h_left_y2, roi_rotation_angle, (x, y))
                right_corners = calculate_rotated_corners(h_right_x1, h_right_y1, h_right_x2, h_right_y2, roi_rotation_angle, (x, y))
                debug_rois_data = {
                    'left': {'corners': left_corners},
                    'right': {'corners': right_corners}
                }
            else:
                # Non-rotated ROIs - use old format
                debug_rois_data = {
                    'left': (h_left_x1, h_left_y1, h_left_x2, h_left_y2),
                    'right': (h_right_x1, h_right_y1, h_right_x2, h_right_y2)
                }

            extent = {
                'left': extent_left_x,
                'right': extent_right_x,
                'left_y': extent_left_y,
                'right_y': extent_right_y,
                'span': extent_right_x - extent_left_x,
                'angle': extent_angle,
                'hline_angle': measurement.get('h_line_angle', extent_angle),  # Actual H-line angle for visualization
                'debug_rois': debug_rois_data
            }
            print(f"      Found arrows on both sides: extent ({extent_left_x:.0f},{extent_left_y:.0f}) to ({extent_right_x:.0f},{extent_right_y:.0f}) (span: {extent['span']:.0f}px, angle: {extent_angle:.1f}°)")
            return extent
        elif has_left_arrows or has_right_arrows:
            # Only one arrow found - mirror the distance to the other side at same angle
            if has_left_arrows:
                # Have left arrow - transform ROI coords to image coords
                if h_left_M_inv is not None:
                    # Rotated ROI - use inverse transform
                    roi_center_y = (h_left_y2 - h_left_y1) / 2
                    arrow_pt = np.array([[[left_arrow_x, roi_center_y]]], dtype=np.float32)
                    arrow_img = cv2.perspectiveTransform(arrow_pt, h_left_M_inv)
                    extent_left_x, extent_left_y = arrow_img[0][0]
                    print(f"        [TRANSFORM] Left arrow: ROI({left_arrow_x:.0f}, {roi_center_y:.0f}) → Image({extent_left_x:.0f}, {extent_left_y:.0f})")
                else:
                    # Non-rotated ROI - simple offset
                    extent_left_x = h_left_x1 + left_arrow_x
                    extent_left_y = y

                # Mirror the distance to right side at same angle
                distance = abs(extent_left_x - x)
                extent_right_x = x + distance
                dx_right = extent_right_x - x
                dy_right = dx_right * np.tan(np.radians(extent_angle))
                extent_right_y = y + dy_right
                side = "left"
            else:
                # Have right arrow - transform ROI coords to image coords
                if h_right_M_inv is not None:
                    # Rotated ROI - use inverse transform
                    roi_center_y = (h_right_y2 - h_right_y1) / 2
                    arrow_pt = np.array([[[right_arrow_x, roi_center_y]]], dtype=np.float32)
                    arrow_img = cv2.perspectiveTransform(arrow_pt, h_right_M_inv)
                    extent_right_x, extent_right_y = arrow_img[0][0]
                    print(f"        [TRANSFORM] Right arrow: ROI({right_arrow_x:.0f}, {roi_center_y:.0f}) → Image({extent_right_x:.0f}, {extent_right_y:.0f})")
                else:
                    # Non-rotated ROI - simple offset
                    extent_right_x = h_right_x1 + right_arrow_x
                    extent_right_y = y

                # Mirror the distance to left side at same angle
                distance = abs(extent_right_x - x)
                extent_left_x = x - distance
                dx_left = extent_left_x - x
                dy_left = dx_left * np.tan(np.radians(extent_angle))
                extent_left_y = y + dy_left
                side = "right"

            # Store debug_rois with rotated corners if ROI is rotated
            if roi_rotation_angle != 0:
                # Calculate rotated corners for visualization
                left_corners = calculate_rotated_corners(h_left_x1, h_left_y1, h_left_x2, h_left_y2, roi_rotation_angle, (x, y))
                right_corners = calculate_rotated_corners(h_right_x1, h_right_y1, h_right_x2, h_right_y2, roi_rotation_angle, (x, y))
                debug_rois_data = {
                    'left': {'corners': left_corners},
                    'right': {'corners': right_corners}
                }
            else:
                # Non-rotated ROIs - use old format
                debug_rois_data = {
                    'left': (h_left_x1, h_left_y1, h_left_x2, h_left_y2),
                    'right': (h_right_x1, h_right_y1, h_right_x2, h_right_y2)
                }

            extent = {
                'left': extent_left_x,
                'right': extent_right_x,
                'left_y': extent_left_y,
                'right_y': extent_right_y,
                'span': extent_right_x - extent_left_x,
                'angle': extent_angle,
                'hline_angle': measurement.get('h_line_angle', extent_angle),  # Actual H-line angle for visualization
                'debug_rois': debug_rois_data
            }
            print(f"      Found arrow on {side} side only: mirrored extent ({extent_left_x:.0f},{extent_left_y:.0f}) to ({extent_right_x:.0f},{extent_right_y:.0f}) (span: {extent['span']:.0f}px, angle: {extent_angle:.1f}°)")
            return extent
        else:
            # No arrows found - use fallback extent OR return failed extent with debug_rois
            print(f"      No arrows found on either side - using fallback")
            if 'width_extent' in measurement:
                fallback_extent = measurement['width_extent'].copy()

                # Store debug_rois with rotated corners if ROI is rotated
                if roi_rotation_angle != 0:
                    # Calculate rotated corners for visualization
                    left_corners = calculate_rotated_corners(h_left_x1, h_left_y1, h_left_x2, h_left_y2, roi_rotation_angle, (x, y))
                    right_corners = calculate_rotated_corners(h_right_x1, h_right_y1, h_right_x2, h_right_y2, roi_rotation_angle, (x, y))
                    fallback_extent['debug_rois'] = {
                        'left': {'corners': left_corners},
                        'right': {'corners': right_corners}
                    }
                else:
                    # Non-rotated ROIs - use old format
                    fallback_extent['debug_rois'] = {
                        'left': (h_left_x1, h_left_y1, h_left_x2, h_left_y2),
                        'right': (h_right_x1, h_right_y1, h_right_x2, h_right_y2)
                    }

                print(f"      Using fallback extent: {fallback_extent['left']:.0f} to {fallback_extent['right']:.0f} (span: {fallback_extent['span']:.0f}px)")
                return fallback_extent
            else:
                # No fallback extent available - return failed extent with debug_rois for visualization
                print(f"      No fallback extent available - returning failed extent with debug_rois")

                # Store debug_rois for visualization even though extent failed
                if roi_rotation_angle != 0:
                    # Calculate rotated corners for visualization
                    left_corners = calculate_rotated_corners(h_left_x1, h_left_y1, h_left_x2, h_left_y2, roi_rotation_angle, (x, y))
                    right_corners = calculate_rotated_corners(h_right_x1, h_right_y1, h_right_x2, h_right_y2, roi_rotation_angle, (x, y))
                    debug_rois_data = {
                        'left': {'corners': left_corners},
                        'right': {'corners': right_corners}
                    }
                else:
                    # Non-rotated ROIs - use old format
                    debug_rois_data = {
                        'left': (h_left_x1, h_left_y1, h_left_x2, h_left_y2),
                        'right': (h_right_x1, h_right_y1, h_right_x2, h_right_y2)
                    }

                # Return a dict with failed flag and debug_rois
                return {
                    'failed': True,
                    'debug_rois': debug_rois_data,
                    'angle': extent_angle,
                    'hline_angle': measurement.get('h_line_angle', extent_angle)  # Actual H-line angle for visualization
                }

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
        v_top_M_inv = None
        v_bottom_M_inv = None
        if roi_rotation_angle != 0:
            v_top_roi, v_top_M_inv = extract_rotated_roi(image, v_top_x1, v_top_y1, v_top_x2, v_top_y2, roi_rotation_angle, (x, y))
            v_bottom_roi, v_bottom_M_inv = extract_rotated_roi(image, v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2, roi_rotation_angle, (x, y))
        else:
            v_top_roi = image[v_top_y1:v_top_y2, v_top_x1:v_top_x2]
            v_bottom_roi = image[v_bottom_y1:v_bottom_y2, v_bottom_x1:v_bottom_x2]

        # Detect lines in extended ROIs
        top_v_lines = detect_lines_in_roi(v_top_roi, v_top_x1, v_top_y1, v_top_M_inv)
        bottom_v_lines = detect_lines_in_roi(v_bottom_roi, v_bottom_x1, v_bottom_y1, v_bottom_M_inv)

        # Filter for vertical lines
        all_v_lines = []
        for line in top_v_lines + bottom_v_lines:
            lx1, ly1, lx2, ly2 = line['image_coords']
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


def classify_measurements_by_lines(image, measurements, image_path=None, non_measurement_text_exclusions=None, all_measurements=None):
    """
    Classify measurements as WIDTH, HEIGHT, or UNCLASSIFIED based on nearby dimension lines

    Args:
        image: Source image
        measurements: List of measurement dicts
        image_path: Path to image for debug output
        non_measurement_text_exclusions: List of items to exclude from arrow detection (OL text, room names)
        all_measurements: List of all measurements for text exclusion
        image_path: Path to image file (for debug image saving)

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

    # Add measurement IDs if not present (for debug filenames)
    for i, meas in enumerate(measurements):
        if 'id' not in meas:
            meas['id'] = i + 1

    for i, meas in enumerate(measurements):
        print(f"\n  Analyzing measurement {i+1}: '{meas['text']}' at ({meas['position'][0]:.0f}, {meas['position'][1]:.0f})")

        # Find lines near this measurement
        line_info = find_lines_near_measurement(image, meas, save_roi_debug=True, image_path=image_path, exclude_items=non_measurement_text_exclusions, all_measurements=all_measurements)

        # Check if it's actually classified or unclassified
        if line_info and not line_info.get('unclassified'):
            print(f"    Found {line_info['orientation']} at distance {line_info['distance']:.1f} pixels")
            # Horizontal line = WIDTH measurement
            # Vertical line = HEIGHT measurement
            if line_info['orientation'] == 'horizontal_line':
                classified['width'].append(meas['text'])
                measurement_categories.append('width')
                print(f"    → Classified as WIDTH")
                # NOTE: Extent calculation moved to Phase 4 (pairing)
                # Extent is only calculated for bottom widths after down arrow classification
            elif line_info['orientation'] == 'vertical_line':
                classified['height'].append(meas['text'])
                measurement_categories.append('height')
                print(f"    → Classified as HEIGHT")
                # NOTE: Heights do not need extent calculation
        else:
            # UNCLASSIFIED - try multiple rotations and pick the best one
            print(f"    UNCLASSIFIED - Trying rotation fallback with scoring...")

            # Try all rotations and score them
            rotation_angles = [+22.5, -22.5, +30.0, -30.0, +45.0, -45.0]
            rotation_results = []

            for angle in rotation_angles:
                print(f"    Attempting {angle:+.1f}° rotation...")
                meas_rot = meas.copy()
                meas_rot['roi_rotation_angle'] = angle
                line_info = find_lines_near_measurement(image, meas_rot, save_roi_debug=True, image_path=image_path, exclude_items=non_measurement_text_exclusions, all_measurements=all_measurements)

                if line_info and not line_info.get('unclassified'):
                    # Calculate score based on how horizontal the lines are
                    # For horizontal lines: deviation from expected angle
                    # For vertical lines: don't score (we want horizontal)
                    if line_info['orientation'] == 'horizontal_line':
                        line_angle = line_info.get('angle', 0)
                        # Expected angle for horizontal lines should match rotation angle
                        deviation = abs(line_angle - angle)
                        # Normalize to 0-180 range
                        if deviation > 90:
                            deviation = 180 - deviation

                        score = deviation  # Lower is better
                        print(f"      Found h-lines at angle={line_angle:.1f}°, deviation={deviation:.1f}° (score={score:.1f})")

                        rotation_results.append({
                            'angle': angle,
                            'line_info': line_info,
                            'meas_copy': meas_rot,
                            'score': score,
                            'orientation': 'horizontal_line',
                            'line_angle': line_angle  # Store the actual h-line angle for extent calculation
                        })
                    elif line_info['orientation'] == 'vertical_line':
                        # Store but with high score (we prefer horizontal for width)
                        print(f"      Found v-lines (not scoring for width)")
                        rotation_results.append({
                            'angle': angle,
                            'line_info': line_info,
                            'meas_copy': meas_rot,
                            'score': 999,  # High score = low priority
                            'orientation': 'vertical_line'
                        })

            # Pick the best rotation based on score
            if rotation_results:
                # Sort by score (lowest first)
                rotation_results.sort(key=lambda r: r['score'])
                best = rotation_results[0]

                print(f"    BEST ROTATION: {best['angle']:+.1f}° with score={best['score']:.1f}")
                print(f"    ROTATION FALLBACK SUCCESS: Found {best['orientation']} with {best['angle']:+.1f}° rotation!")

                # Apply the winning rotation
                if best['orientation'] == 'horizontal_line':
                    classified['width'].append(meas['text'])
                    measurement_categories.append('width')
                    meas['roi_rotation_angle'] = best['angle']
                    meas['actual_h_left_roi'] = best['meas_copy'].get('actual_h_left_roi')
                    meas['actual_h_right_roi'] = best['meas_copy'].get('actual_h_right_roi')
                    meas['actual_v_top_roi'] = best['meas_copy'].get('actual_v_top_roi')
                    meas['actual_v_bottom_roi'] = best['meas_copy'].get('actual_v_bottom_roi')
                    # Copy detected lines for visualization
                    meas['h_lines'] = best['meas_copy'].get('h_lines', [])
                    meas['v_lines'] = best['meas_copy'].get('v_lines', [])
                    # IMPORTANT: Use the actual h-line angle from scoring, not from measurement object
                    meas['h_line_angle'] = best.get('line_angle', best['angle'])  # Use line_angle if available, fallback to rotation angle
                    print(f"    → Classified as WIDTH (via {best['angle']:+.1f}° rotation)")
                    # NOTE: Extent calculation moved to Phase 4 (pairing)
                    # Extent is only calculated for bottom widths after down arrow classification
                elif best['orientation'] == 'vertical_line':
                    classified['height'].append(meas['text'])
                    measurement_categories.append('height')
                    meas['roi_rotation_angle'] = best['angle']
                    meas['actual_h_left_roi'] = best['meas_copy'].get('actual_h_left_roi')
                    meas['actual_h_right_roi'] = best['meas_copy'].get('actual_h_right_roi')
                    meas['actual_v_top_roi'] = best['meas_copy'].get('actual_v_top_roi')
                    meas['actual_v_bottom_roi'] = best['meas_copy'].get('actual_v_bottom_roi')
                    # Copy detected lines for visualization
                    meas['h_lines'] = best['meas_copy'].get('h_lines', [])
                    meas['v_lines'] = best['meas_copy'].get('v_lines', [])
                    print(f"    → Classified as HEIGHT (via {best['angle']:+.1f}° rotation)")
                    # NOTE: Heights do not need extent calculation
            else:
                # Still unclassified after all rotations - reset to 0° for viz
                meas['roi_rotation_angle'] = 0.0  # Reset to regular ROIs in viz
                meas['rotation_failed'] = True
                # Don't store the rotated ROI coords - keep original non-rotated ones
                print(f"    No dimension lines found even with rotation")
                classified['unclassified'].append(meas['text'])
                measurement_categories.append('unclassified')
                print(f"    → Classified as UNCLASSIFIED (rotation attempted but failed, ROIs reset to 0°)")

    return classified, measurement_categories
