#!/usr/bin/env python3
"""
Visualization module for cabinet measurement detection.
Handles drawing output images with markers, labels, and annotations.
"""

import cv2
import numpy as np
import os
from measurement_config import ZOOM_CONFIG
from measurement_pairing import add_fraction_to_measurement, find_clear_position_for_marker


def create_visualization(
    image_path,
    groups,
    measurement_texts,
    measurement_logic=None,
    save_viz=True,
    opencv_regions=None,
    measurement_categories=None,
    measurements_list=None,
    show_rois=False,
    paired_openings=None,
    show_groups=False,
    show_opencv=False,
    show_line_rois=True,
    show_panel=True,
    show_pairing=True,
    show_classification=True,
    room_name=None,
    overlay_info=None,
    unpaired_heights_info=None,
    page_number=None,
    start_opening_number=1
):
    """Create visualization showing groups and measurements side by side"""

    # Colors for different groups (BGR format - muted professional colors)
    COLORS = [
        (0, 0, 200),      # Dark Red
        (0, 100, 180),    # Dark Orange
        (180, 0, 0),     # Dark Blue
        (128, 0, 128),   # Purple
        (150, 0, 150),   # Dark Magenta
        (128, 64, 0),     # Dark blue
        (0, 128, 128),   # Dark olive/brown
        (100, 0, 100),   # Dark purple
        (0, 80, 160),     # Dark orange-red
        (80, 40, 120),   # Dark red-purple
        (150, 75, 0),     # Navy blue
        (75, 0, 130),     # Indigo
    ]

    # Convert Windows network paths to Unix-style for OpenCV
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')

    # Load original image
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Could not load image for visualization")
        return

    vis_image = image.copy()
    h, w = image.shape[:2]

    # Draw ROIs for line detection (part of classification visualization)
    # Line ROIs are shown when classification is enabled (unless explicitly disabled)
    if show_classification and show_line_rois and measurements_list and len(measurements_list) > 0:
        for i, meas in enumerate(measurements_list[:len(groups)]):  # Match measurements to groups
            if i >= len(groups):
                break

            # Get measurement position and bounds
            x = int(meas['position'][0])
            y = int(meas['position'][1])

            # Get bounds for text size estimation
            if 'bounds' in meas:
                bounds = meas['bounds']
                text_width = int(bounds['right'] - bounds['left'])
                text_height = int(bounds['bottom'] - bounds['top'])
                # Update x,y to be the actual center of the text bounds (same as detection logic)
                x = int((bounds['left'] + bounds['right']) / 2)
                y = int((bounds['top'] + bounds['bottom']) / 2)
            else:
                # Estimate
                text_height = 30
                text_width = len(meas.get('text', '')) * 15

            # Calculate ROIs EXACTLY same as in find_lines_near_measurement (UPDATED VERSION)
            # Horizontal ROIs (for WIDTH detection)
            # Use 0.75x text width but ensure minimum of 40 pixels for short text
            h_strip_extension = max(40, int(text_width * 0.75))

            # IMPROVED: Match the corrected H-ROI logic with proper gaps
            min_gap = 10  # Minimum 10px gap between text and H-ROI

            # Check if this measurement used rotation fallback
            viz_rotation_angle = meas.get('roi_rotation_angle', 0.0)

            # Use the ACTUAL ROI coordinates stored during detection
            if 'actual_h_left_roi' in meas:
                h_left_x1, h_left_y1, h_left_x2, h_left_y2 = meas['actual_h_left_roi']
            else:
                # Fallback to recalculation if not stored
                text_left_edge = int(x - text_width//2)
                h_left_x2 = max(0, text_left_edge - min_gap)
                h_left_x1 = max(0, h_left_x2 - h_strip_extension)
                h_left_y1 = int(y - text_height * 2.0)
                h_left_y2 = int(y + text_height * 2.0)

            if 'actual_h_right_roi' in meas:
                h_right_x1, h_right_y1, h_right_x2, h_right_y2 = meas['actual_h_right_roi']
            else:
                # Fallback to recalculation if not stored
                text_right_edge = int(x + text_width//2)
                h_right_x1 = min(w, text_right_edge + min_gap)
                h_right_x2 = min(w, h_right_x1 + h_strip_extension)
                h_right_y1 = h_left_y1
                h_right_y2 = h_left_y2

            # Add label if rotation was used
            if viz_rotation_angle != 0:
                rotation_failed = meas.get('rotation_failed', False)
                if rotation_failed:
                    rotation_label = f"ROI ROTATED {viz_rotation_angle:+.1f}° FAILED"
                    label_color = (0, 0, 255)  # Red for failed
                else:
                    rotation_label = f"ROI ROTATED {viz_rotation_angle:+.1f}° SUCCESS"
                    label_color = (0, 255, 0)  # Green for success

                label_x = x - 80
                label_y = y - text_height - 15
                # Draw label background
                cv2.rectangle(vis_image, (label_x - 5, label_y - 15), (label_x + 200, label_y + 5), (0, 0, 0), -1)
                cv2.rectangle(vis_image, (label_x - 5, label_y - 15), (label_x + 200, label_y + 5), label_color, 1)
                # Draw label text
                cv2.putText(vis_image, rotation_label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

            # Use the ACTUAL V-ROI coordinates stored during detection
            if 'actual_v_top_roi' in meas:
                v_top_x1, v_top_y1, v_top_x2, v_top_y2 = meas['actual_v_top_roi']
            else:
                # Fallback to recalculation if not stored
                v_strip_extension = int(text_height * 4)
                v_top_x1 = int(x - text_width//2)
                v_top_x2 = int(x + text_width//2)
                v_top_y1 = max(0, int(y - text_height//2 - v_strip_extension))
                v_top_y2 = int(y - text_height//2 - 5)

            if 'actual_v_bottom_roi' in meas:
                v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2 = meas['actual_v_bottom_roi']
            else:
                # Fallback to recalculation if not stored
                v_strip_extension = int(text_height * 4)
                v_bottom_x1 = v_top_x1 if 'actual_v_top_roi' in meas else int(x - text_width//2)
                v_bottom_x2 = v_top_x2 if 'actual_v_top_roi' in meas else int(x + text_width//2)
                v_bottom_y1 = int(y + text_height//2 + 5)
                v_bottom_y2 = min(h, int(y + text_height//2 + v_strip_extension))

            # Helper function to draw rotated rectangle
            def draw_rotated_rect(image, x1, y1, x2, y2, angle, color, center_point):
                """Draw a rotated rectangle given corners and rotation angle around center point"""
                if angle == 0:
                    # No rotation - draw normal rectangle with dotted lines
                    for px in range(x1, x2, 8):
                        cv2.line(image, (px, y1), (min(px+4, x2), y1), color, 1)
                        cv2.line(image, (px, y2), (min(px+4, x2), y2), color, 1)
                    for py in range(y1, y2, 8):
                        cv2.line(image, (x1, py), (x1, min(py+4, y2)), color, 1)
                        cv2.line(image, (x2, py), (x2, min(py+4, y2)), color, 1)
                else:
                    # Rotate the corners around the center point
                    cx, cy = center_point
                    angle_rad = np.radians(angle)
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)

                    # Original corners
                    corners = [
                        (x1, y1),  # Top-left
                        (x2, y1),  # Top-right
                        (x2, y2),  # Bottom-right
                        (x1, y2)   # Bottom-left
                    ]

                    # Rotate each corner around center
                    rotated_corners = []
                    for px, py in corners:
                        # Translate to origin
                        px_rel = px - cx
                        py_rel = py - cy
                        # Rotate
                        px_rot = px_rel * cos_a - py_rel * sin_a
                        py_rot = px_rel * sin_a + py_rel * cos_a
                        # Translate back
                        rotated_corners.append((int(px_rot + cx), int(py_rot + cy)))

                    # Draw the rotated rectangle with solid lines (easier than dotted for rotated)
                    pts = np.array(rotated_corners, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, color, 2)

            # Draw horizontal ROIs with dotted rectangles (for WIDTH detection) - RED
            # Left horizontal ROI
            if h_left_x2 > h_left_x1 and h_left_y2 > h_left_y1:
                draw_rotated_rect(vis_image, h_left_x1, h_left_y1, h_left_x2, h_left_y2,
                                 viz_rotation_angle, (0, 0, 200), (x, y))
                cv2.putText(vis_image, "W", (h_left_x1+2, h_left_y1-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)

            # Right horizontal ROI
            if h_right_x2 > h_right_x1 and h_right_y2 > h_right_y1:
                draw_rotated_rect(vis_image, h_right_x1, h_right_y1, h_right_x2, h_right_y2,
                                 viz_rotation_angle, (0, 0, 200), (x, y))
                cv2.putText(vis_image, "W", (h_right_x2-10, h_right_y1-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)

            # Draw vertical ROIs with dotted rectangles (for HEIGHT detection) - BLUE
            # Top vertical ROI
            if v_top_x2 > v_top_x1 and v_top_y2 > v_top_y1:
                draw_rotated_rect(vis_image, v_top_x1, v_top_y1, v_top_x2, v_top_y2,
                                 viz_rotation_angle, (200, 0, 0), (x, y))
                cv2.putText(vis_image, "H", (v_top_x1+2, v_top_y1+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 1)

            # Bottom vertical ROI
            if v_bottom_x2 > v_bottom_x1 and v_bottom_y2 > v_bottom_y1:
                draw_rotated_rect(vis_image, v_bottom_x1, v_bottom_y1, v_bottom_x2, v_bottom_y2,
                                 viz_rotation_angle, (200, 0, 0), (x, y))
                cv2.putText(vis_image, "H", (v_bottom_x1+2, v_bottom_y2-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 1)

    # Draw groups with different colors (just groups, no classification)
    if show_groups:
      for i, group in enumerate(groups):
        # Use cycling colors for groups (no classification here)
        color = COLORS[i % len(COLORS)]
        label_suffix = ""  # No classification labels in groups

        bounds = group['bounds']
        texts = group.get('texts', [])

        # Draw rectangle for group
        left = int(bounds['left'])
        right = int(bounds['right'])
        top = int(bounds['top'])
        bottom = int(bounds['bottom'])

        cv2.rectangle(vis_image, (left - 5, top - 5), (right + 5, bottom + 5), color, 3)

        # Draw group number with category
        group_label = f"G{i+1}{label_suffix}"
        label_x = max(10, left - 35)
        label_y = max(30, top - 10)
        cv2.putText(vis_image, group_label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw center point
        center = group['center']
        cv2.circle(vis_image, (int(center[0]), int(center[1])), 5, color, -1)

        # Draw zoom region as dashed rectangle
        # Calculate zoom bounds (using same padding as in verify_measurement_at_center_with_logic)
        padding = ZOOM_CONFIG['padding']  # Should be 150 based on config
        zoom_left = max(0, int(min(bounds['left'], bounds['right']) - padding))
        zoom_top = max(0, int(min(bounds['top'], bounds['bottom']) - padding))
        zoom_right = min(w, int(max(bounds['left'], bounds['right']) + padding))
        zoom_bottom = min(h, int(max(bounds['top'], bounds['bottom']) + padding))

        # Draw dashed rectangle for zoom region
        # Create a lighter version of the color for the zoom region
        lighter_color = tuple(min(255, c + 100) for c in color)

        # Draw dashed lines manually (OpenCV doesn't have built-in dashed lines)
        dash_length = 10
        gap_length = 5

        # Top edge
        x = zoom_left
        while x < zoom_right:
            x_end = min(x + dash_length, zoom_right)
            cv2.line(vis_image, (x, zoom_top), (x_end, zoom_top), lighter_color, 1)
            x += dash_length + gap_length

        # Bottom edge
        x = zoom_left
        while x < zoom_right:
            x_end = min(x + dash_length, zoom_right)
            cv2.line(vis_image, (x, zoom_bottom), (x_end, zoom_bottom), lighter_color, 1)
            x += dash_length + gap_length

        # Left edge
        y = zoom_top
        while y < zoom_bottom:
            y_end = min(y + dash_length, zoom_bottom)
            cv2.line(vis_image, (zoom_left, y), (zoom_left, y_end), lighter_color, 1)
            y += dash_length + gap_length

        # Right edge
        y = zoom_top
        while y < zoom_bottom:
            y_end = min(y + dash_length, zoom_bottom)
            cv2.line(vis_image, (zoom_right, y), (zoom_right, y_end), lighter_color, 1)
            y += dash_length + gap_length

    # Draw OpenCV additions if provided
    if show_opencv and opencv_regions:
        for region in opencv_regions:
            cx, cy = region['center']
            # Draw with thick red circle and "OCV+" label
            cv2.circle(vis_image, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            cv2.circle(vis_image, (int(cx), int(cy)), 8, (0, 0, 255), 2)
            cv2.putText(vis_image, "OCV+", (int(cx)-20, int(cy)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            # Special marking for potential left 23
            if 250 < cx < 400 and 700 < cy < 850:
                cv2.rectangle(vis_image, (int(cx)-15, int(cy)-15),
                            (int(cx)+15, int(cy)+15), (255, 0, 255), 2)

    # Create info panel if enabled (this is for grouping, not classification/pairing)
    if show_panel and show_groups:
        panel_width = 400  # Narrower panel
        panel_height = 150  # Fixed height for summary
        panel_y = h - panel_height - 20  # Position at bottom with 20px margin
        panel_x = 20  # Left margin

        # Create semi-transparent black background for panel
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0, vis_image)

        # Draw panel border
        cv2.rectangle(vis_image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                      (200, 200, 200), 2)

        # Add title
        y_pos = panel_y + 30
        x_pos = panel_x + 10
        cv2.putText(vis_image, "MEASUREMENT CLASSIFICATION", (x_pos, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 30

        # Show category counts if available and classification is enabled
        if show_classification and measurement_categories:
            width_count = sum(1 for cat in measurement_categories if cat == 'width')
            height_count = sum(1 for cat in measurement_categories if cat == 'height')
            unclass_count = sum(1 for cat in measurement_categories if cat == 'unclassified')

            cv2.putText(vis_image, f"Widths: {width_count}", (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis_image, f"Heights: {height_count}", (x_pos + 120, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if unclass_count > 0:
                cv2.putText(vis_image, f"Unclassified: {unclass_count}", (x_pos + 240, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            y_pos += 35

        # Show pairing summary if available and pairing is enabled
        if show_pairing and paired_openings:
            cv2.putText(vis_image, f"Cabinet Openings: {len(paired_openings)}", (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30

        # Simple line separator at bottom of panel
        cv2.line(vis_image, (panel_x + 10, panel_y + panel_height - 20),
                 (panel_x + panel_width - 10, panel_y + panel_height - 20), (200, 200, 200), 1)

    # Draw classification labels on measurements when classification is enabled
    with open('//vmware-host/Shared Folders/D/OneDrive/customers/raised panel/Measures-2025-09-09(14-18)/all_pages/viz_debug.txt', 'a') as f:
        f.write(f"show_classification={show_classification}, measurement_categories={'exists' if measurement_categories else 'None'}, measurements_list={'exists' if measurements_list else 'None'}\n")
        if measurement_categories:
            f.write(f"  len(measurement_categories)={len(measurement_categories)}\n")
        if measurements_list:
            f.write(f"  len(measurements_list)={len(measurements_list)}\n")

    if show_classification and measurement_categories and measurements_list:
        # Track if we've drawn the first width extent line
        first_width_drawn = False

        for i, (meas, category) in enumerate(zip(measurements_list, measurement_categories)):
            if i >= len(measurement_categories):
                break

            x, y = int(meas['position'][0]), int(meas['position'][1])
            print(f"[VIZ-LOOP] M{i+1} ({meas.get('text')}): cat={category}, has_width_extent={'width_extent' in meas}")
            # Debug to file
            with open('//vmware-host/Shared Folders/D/OneDrive/customers/raised panel/Measures-2025-09-09(14-18)/all_pages/viz_debug.txt', 'a') as f:
                f.write(f"M{i+1} ({meas.get('text')}): cat={category}, has_width_extent={'width_extent' in meas}")
                if 'width_extent' in meas:
                    f.write(f", has_debug_rois={'debug_rois' in meas['width_extent']}\n")
                else:
                    f.write("\n")

            # DEBUG: Draw OCR text with box below the actual text for comparison
            # DISABLED by default - set SHOW_DEBUG_TEXT_BOXES = True to enable
            SHOW_DEBUG_TEXT_BOXES = False  # Set to True to show debug text boxes and OCR text

            if SHOW_DEBUG_TEXT_BOXES and 'bounds' in meas and meas['bounds']:
                bounds = meas['bounds']
                text_left = int(bounds['left'])
                text_right = int(bounds['right'])
                text_top = int(bounds['top'])
                text_bottom = int(bounds['bottom'])
                text_width_px = text_right - text_left
                text_height_px = text_bottom - text_top

                # Draw bounding box around actual text (cyan color)
                cv2.rectangle(vis_image, (text_left, text_top), (text_right, text_bottom), (255, 255, 0), 2)

                # Get the RAW OCR text (not cleaned) - this is what the bounds were calculated from
                measurement_text = meas.get('raw_ocr_text', meas.get('text', ''))

                # Draw the OCR text below with same dimensions
                # Position it below the actual text with some spacing
                draw_y_offset = text_bottom + 50  # 50px below actual text

                # Draw box with same dimensions as calculated bounds
                draw_box_left = text_left
                draw_box_top = draw_y_offset
                draw_box_right = text_left + text_width_px
                draw_box_bottom = draw_y_offset + text_height_px

                # Draw the comparison box (cyan)
                cv2.rectangle(vis_image, (draw_box_left, draw_box_top), (draw_box_right, draw_box_bottom), (255, 255, 0), 2)

                # Draw the OCR text inside the box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                # Get text size to center it in the box
                (text_w, text_h), baseline = cv2.getTextSize(measurement_text, font, font_scale, thickness)

                # Center the text in the drawn box
                text_x = draw_box_left + (text_width_px - text_w) // 2
                text_y = draw_box_top + (text_height_px + text_h) // 2

                # Draw the text
                cv2.putText(vis_image, measurement_text, (text_x, text_y),
                           font, font_scale, (0, 255, 255), thickness)

            # Determine color based on category
            if category == 'width':
                color = (0, 0, 255)  # Red for WIDTH
                label = "WIDTH"
            elif category == 'height':
                color = (255, 0, 0)  # Blue for HEIGHT
                label = "HEIGHT"
            else:
                color = (128, 128, 128)  # Gray for UNCLASSIFIED
                label = "UNCLASS"

            # Find clear position for classification label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Collect avoid regions from measurement bounds
            avoid_regions = []
            if 'bounds' in meas and meas['bounds']:
                bounds = meas['bounds']
                avoid_regions.append({
                    'left': bounds['left'],
                    'right': bounds['right'],
                    'top': bounds['top'],
                    'bottom': bounds['bottom']
                })

            # Try different positions around the measurement
            test_offsets = [
                (15, -10),   # Upper right
                (-text_width - 15, -10),  # Upper left
                (15, 20),    # Lower right
                (-text_width - 15, 20),   # Lower left
                (0, -25),    # Above center
                (0, 30),     # Below center
            ]

            best_pos = (x + 12, y + 5)  # Default position
            min_overlap = float('inf')

            for offset_x, offset_y in test_offsets:
                label_x = x + offset_x
                label_y = y + offset_y

                # Check if label fits in image
                if label_x < 0 or label_x + text_width >= w:
                    continue
                if label_y - text_height < 0 or label_y >= h:
                    continue

                # Calculate label bounds
                label_left = label_x
                label_right = label_x + text_width
                label_top = label_y - text_height
                label_bottom = label_y

                # Check overlap with measurement bounds
                total_overlap = 0
                for region in avoid_regions:
                    if (label_left < region['right'] and label_right > region['left'] and
                        label_top < region['bottom'] and label_bottom > region['top']):
                        overlap_x = min(label_right, region['right']) - max(label_left, region['left'])
                        overlap_y = min(label_bottom, region['bottom']) - max(label_top, region['top'])
                        total_overlap += overlap_x * overlap_y

                if total_overlap < min_overlap:
                    min_overlap = total_overlap
                    best_pos = (label_x, label_y)
                    if total_overlap == 0:
                        break

            # Draw classification label at best position
            cv2.putText(vis_image, label, best_pos,
                       font, font_scale, color, thickness)

            # Draw extent lines ONLY for bottom width measurements (those with drawer_config)
            if category == 'width' and 'width_extent' in meas and 'drawer_config' in meas:
                extent = meas['width_extent']
                print(f"[VIZ] M{i+1} has width_extent, debug_rois present: {'debug_rois' in extent}")
                left_x = int(extent['left'])
                right_x = int(extent['right'])
                # Use actual Y coordinates if available, otherwise use measurement Y
                left_y = int(extent.get('left_y', y))
                right_y = int(extent.get('right_y', y))

                # Draw extended ROI rectangles (debug visualization) - DISABLED
                # if 'debug_rois' in extent:
                #     rois = extent['debug_rois']
                #     # Draw left ROI in magenta (bright pink)
                #     lx1, ly1, lx2, ly2 = rois['left']
                #     cv2.rectangle(vis_image, (lx1, ly1), (lx2, ly2), (255, 0, 255), 3)
                #     # Draw right ROI in orange
                #     rx1, ry1, rx2, ry2 = rois['right']
                #     cv2.rectangle(vis_image, (rx1, ry1), (rx2, ry2), (0, 165, 255), 3)

                # Draw thick RED line at the actual angle of the H-lines
                cv2.line(vis_image, (left_x, left_y), (right_x, right_y), (0, 0, 255), 5)

                # Draw RED tick marks perpendicular to the line angle
                angle = extent.get('angle', 0)
                angle_rad = np.radians(angle)
                # Perpendicular angle is 90 degrees offset
                perp_angle = angle_rad + np.pi/2
                tick_len = 15

                # Left tick mark
                tick_dx = int(tick_len * np.cos(perp_angle))
                tick_dy = int(tick_len * np.sin(perp_angle))
                cv2.line(vis_image,
                        (left_x - tick_dx, left_y - tick_dy),
                        (left_x + tick_dx, left_y + tick_dy),
                        (0, 0, 255), 5)

                # Right tick mark
                cv2.line(vis_image,
                        (right_x - tick_dx, right_y - tick_dy),
                        (right_x + tick_dx, right_y + tick_dy),
                        (0, 0, 255), 5)

                # Add label showing measurement number and span
                label_text = f"M{i+1} ({int(extent['span'])}px) {angle:.1f}°"
                cv2.putText(vis_image, label_text, (left_x - 50, left_y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"[VIZ DEBUG] Drew extent line for M{i+1} ({meas['text']}) at angle={angle:.1f}°, left=({left_x},{left_y}), right=({right_x},{right_y}), span={extent['span']}")

                # Draw drawer configuration label and scan ROI if available
                if 'drawer_config' in meas:
                    drawer_text = meas['drawer_config']

                    # Determine scan height and text position based on position class
                    position_class = 'bottom'  # default

                    # Extract position class from drawer_text
                    if 'top width' in drawer_text:
                        position_class = 'top'
                        scan_distance_px = 60  # 5/8" at 96 DPI (base height for top widths)
                        # Top widths: place text 1" ABOVE the width line (96px at 96 DPI)
                        text_y = left_y - 96
                    else:
                        position_class = 'bottom'
                        # Bottom widths: scan up to where top widths are + 1"
                        # Find the Y position of top widths in the measurements list
                        top_width_y = None
                        if measurements_list:
                            for other_meas in measurements_list:
                                if 'drawer_config' in other_meas:
                                    other_text = other_meas['drawer_config']
                                    if 'top width' in other_text:
                                        # Get the Y position of this top width from width_extent
                                        if 'width_extent' in other_meas:
                                            other_left_y = other_meas['width_extent'].get('left_y', None)
                                            if other_left_y is not None:
                                                if top_width_y is None or other_left_y < top_width_y:
                                                    top_width_y = other_left_y

                        # If we found a top width, scan to it + 1", otherwise use default
                        if top_width_y is not None:
                            # Distance from current bottom width to top width + 1"
                            scan_distance_px = abs(left_y - top_width_y) + 96  # +1" (96px)
                        else:
                            scan_distance_px = 60  # Default if no top width found

                        # Bottom widths: place text 3/4" BELOW the width line (72px at 96 DPI)
                        text_y = left_y + 72

                    # Draw text at appropriate position
                    cv2.putText(vis_image, drawer_text, (left_x - 50, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    # Calculate scan ROI bounds following the angle of the dimension line
                    # Bottom edge parallel to dimension line, sides perpendicular to it
                    offset_from_width = 12  # 1/8 inch at 96 DPI (bottom edge starts here)
                    additional_height = 96  # 1 inch at 96 DPI (extend above top widths)
                    horizontal_extension = 24  # 1/4 inch at 96 DPI (extend left and right)

                    # Get the angle of the dimension line
                    angle_rad = np.radians(angle)

                    # Calculate direction along the dimension line (for horizontal extension)
                    line_dx = np.cos(angle_rad)
                    line_dy = np.sin(angle_rad)

                    # Calculate perpendicular direction (upward from the line)
                    perp_angle = angle_rad + np.pi/2
                    perp_dx = np.cos(perp_angle)
                    perp_dy = np.sin(perp_angle)

                    # Ensure perpendicular points upward (negative Y)
                    if perp_dy > 0:
                        perp_dx = -perp_dx
                        perp_dy = -perp_dy

                    # Extend the dimension line endpoints by 1/4" on each side
                    extended_left_x = left_x - int(horizontal_extension * line_dx)
                    extended_left_y = left_y - int(horizontal_extension * line_dy)
                    extended_right_x = right_x + int(horizontal_extension * line_dx)
                    extended_right_y = right_y + int(horizontal_extension * line_dy)

                    # Bottom edge: starts 1/8" above extended dimension line, parallel to it
                    # First move perpendicular from the extended dimension line endpoints
                    bottom_left_x = extended_left_x + int(offset_from_width * perp_dx)
                    bottom_left_y = extended_left_y + int(offset_from_width * perp_dy)
                    bottom_right_x = extended_right_x + int(offset_from_width * perp_dx)
                    bottom_right_y = extended_right_y + int(offset_from_width * perp_dy)

                    # Top edge: scan_distance_px further in perpendicular direction
                    # For top widths, add additional_height; for bottom widths, it's already included in scan_distance_px
                    # Move perpendicular from bottom edge
                    if position_class == 'top':
                        total_height = scan_distance_px + additional_height
                    else:
                        total_height = scan_distance_px  # Already includes the +1" for bottom widths

                    top_left_x = bottom_left_x + int(total_height * perp_dx)
                    top_left_y = bottom_left_y + int(total_height * perp_dy)
                    top_right_x = bottom_right_x + int(total_height * perp_dx)
                    top_right_y = bottom_right_y + int(total_height * perp_dy)

                    # Draw scan ROI as a rotated rectangle in cyan (light blue)
                    scan_roi_pts = np.array([
                        [bottom_left_x, bottom_left_y],
                        [bottom_right_x, bottom_right_y],
                        [top_right_x, top_right_y],
                        [top_left_x, top_left_y]
                    ], np.int32)
                    cv2.polylines(vis_image, [scan_roi_pts], isClosed=True,
                                 color=(255, 255, 0), thickness=2)  # Cyan color

                    # Draw detected horizontal edges in yellow
                    if 'edge_viz_data' in meas:
                        edge_data = meas['edge_viz_data']
                        if 'horizontal_edges' in edge_data:
                            for x1, y1, x2, y2 in edge_data['horizontal_edges']:
                                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow color

            # Draw measurement number in a square box
            if True:
                meas_num = str(i + 1)
                meas_font = cv2.FONT_HERSHEY_SIMPLEX
                meas_font_scale = 0.5
                meas_thickness = 1
                (meas_text_w, meas_text_h), meas_baseline = cv2.getTextSize(meas_num, meas_font, meas_font_scale, meas_thickness)

                # Create square box (use max of width/height for square)
                box_size = max(meas_text_w, meas_text_h) + 8

                # Position the box near the measurement (top-left corner)
                box_x = x - 30
                box_y = y - 30

                # Draw white filled square
                cv2.rectangle(vis_image, (box_x, box_y), (box_x + box_size, box_y + box_size), (255, 255, 255), -1)
                # Draw black border
                cv2.rectangle(vis_image, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 0, 0), 2)

                # Center the number in the square
                text_x = box_x + (box_size - meas_text_w) // 2
                text_y = box_y + (box_size + meas_text_h) // 2 - meas_baseline // 2

                # Draw measurement number
                cv2.putText(vis_image, meas_num, (text_x, text_y),
                           meas_font, meas_font_scale, (0, 0, 0), meas_thickness)

    # Draw X-range visualization for unpaired heights
    if False and unpaired_heights_info:  # Disabled X-range visualization
        print(f"Drawing X-range zones for {len(unpaired_heights_info)} unpaired heights")
        for height_info in unpaired_heights_info:
            # Only draw for 9 1/4 measurements
            if '9 1/4' not in height_info.get('text', ''):
                continue

            height_x = height_info['x']
            height_y = height_info['y']
            x_tolerance = height_info.get('x_tolerance', 50)

            # Calculate X-range boundaries
            x_min = max(0, int(height_x - x_tolerance))
            x_max = min(w, int(height_x + x_tolerance))

            # Draw semi-transparent vertical band showing X-range
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (x_min, 0), (x_max, h),
                         (255, 200, 100), -1)  # Light blue color
            cv2.addWeighted(overlay, 0.15, vis_image, 0.85, 0, vis_image)

            # Draw vertical boundary lines
            cv2.line(vis_image, (x_min, 0), (x_min, h), (255, 200, 100), 2)
            cv2.line(vis_image, (x_max, 0), (x_max, h), (255, 200, 100), 2)

            # Draw horizontal line at height position
            cv2.line(vis_image, (x_min, int(height_y)), (x_max, int(height_y)),
                    (0, 255, 255), 2)

            # Add label
            label = f"X-range: {x_tolerance:.0f}px"
            cv2.putText(vis_image, label,
                       (x_min + 5, int(height_y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(vis_image, label,
                       (x_min + 5, int(height_y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw X-range for the 16 5/8 width above (at 940, 810)
    if False:  # Disabled width visualization
        # For the specific width at (940, 810)
        width_x = 940
        width_y = 810
        # Estimate text width for "16 5/8"
        width_text_width = 75  # Based on actual OCR bounds from output
        x_tolerance_width = width_text_width / 2 + 50  # Same formula as heights

        # Calculate X-range boundaries for this width
        x_min_w = max(0, int(width_x - x_tolerance_width))
        x_max_w = min(w, int(width_x + x_tolerance_width))

        # Draw semi-transparent vertical band for width's X-range (different color)
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (x_min_w, 0), (x_max_w, h),
                     (100, 255, 100), -1)  # Light green color for width
        cv2.addWeighted(overlay, 0.15, vis_image, 0.85, 0, vis_image)

        # Draw vertical boundary lines for width
        cv2.line(vis_image, (x_min_w, 0), (x_min_w, h), (100, 255, 100), 2)
        cv2.line(vis_image, (x_max_w, 0), (x_max_w, h), (100, 255, 100), 2)

        # Draw horizontal line at width position
        cv2.line(vis_image, (x_min_w, int(width_y)), (x_max_w, int(width_y)),
                (0, 255, 0), 2)

        # Add label for width
        label_w = f"Width X-range: {x_tolerance_width:.0f}px"
        cv2.putText(vis_image, label_w,
                   (x_min_w + 5, int(width_y) + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis_image, label_w,
                   (x_min_w + 5, int(width_y) + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw paired openings if provided
    if show_pairing and paired_openings and len(paired_openings) > 0:
        print(f"Drawing {len(paired_openings)} paired openings")

        # Define colors for different openings (muted professional colors)
        opening_colors = [
            (0, 0, 200),     # Dark Red
            (180, 0, 0),     # Dark Blue
            (0, 150, 0),     # Dark Green
            (128, 0, 128),   # Purple
            (0, 140, 140),   # Teal
            (150, 75, 0)     # Navy
        ]

        # Track placed markers to avoid overlaps
        placed_markers = []

        for idx, opening in enumerate(paired_openings):
            color = opening_colors[idx % len(opening_colors)]

            # Get positions
            width_x, width_y = opening['width_pos']
            height_x, height_y = opening['height_pos']

            # Calculate intersection point:
            # Start from height position, travel parallel to width's H-lines toward width's X position
            width_hline_angle = opening.get('width_angle', 0)

            # Calculate horizontal distance to travel from height to width's X position
            x_diff = width_x - height_x

            # Use the width's H-line angle to calculate Y offset
            # For a line at angle θ from horizontal: tan(θ) = y/x
            # We travel parallel to the H-line from height position
            angle_rad = np.radians(width_hline_angle)

            # Calculate Y offset based on the H-line angle
            # tan(angle) = opposite/adjacent = y_offset/x_diff
            # y_offset = x_diff * tan(angle)
            if np.abs(x_diff) > 0.01:  # Avoid near-zero horizontal distance
                y_offset = x_diff * np.tan(angle_rad)
            else:
                y_offset = 0  # No horizontal movement

            intersection_x = int(width_x)
            intersection_y = int(height_y + y_offset)

            # Draw lines from intersection to measurements (thin lines)
            cv2.line(vis_image, (intersection_x, intersection_y),
                    (int(width_x), int(width_y)), color, 1, cv2.LINE_AA)
            cv2.line(vis_image, (intersection_x, intersection_y),
                    (int(height_x), int(height_y)), color, 1, cv2.LINE_AA)

            # Calculate opening number text and radius BEFORE finding marker position
            text = f"#{start_opening_number + idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2

            # Get text size for proper centering
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Calculate dynamic circle radius based on text size (with proportional padding)
            # Use 30% padding around the text for a balanced appearance
            padding_factor = 1.30
            radius = int(max(text_width, text_height) * padding_factor / 2)

            # Find clear position for marker that avoids overlapping text and other markers
            marker_x, marker_y = find_clear_position_for_marker(
                intersection_x, intersection_y,
                (width_x, width_y), (height_x, height_y),
                measurements_list if measurements_list else [],
                opening, vis_image,
                placed_markers,  # Pass existing markers to avoid
                overlay_info,  # Pass overlay info for dimension text
                radius  # Pass the calculated radius
            )

            # Draw white filled circle with colored outline for opening number
            cv2.circle(vis_image, (marker_x, marker_y), radius, (255, 255, 255), -1)  # White fill
            cv2.circle(vis_image, (marker_x, marker_y), radius, color, 3)  # Colored outline

            # Calculate centered position (accounting for baseline)
            text_x = marker_x - text_width // 2
            text_y = marker_y + text_height // 2 - baseline // 2

            # Draw opening number
            cv2.putText(vis_image, text,
                       (text_x, text_y),
                       font, font_scale, color, thickness)

            # Draw line connecting marker to intersection if offset
            if marker_x != intersection_x or marker_y != intersection_y:
                cv2.line(vis_image, (intersection_x, intersection_y),
                        (marker_x, marker_y), color, 1, cv2.LINE_AA)

            # Add dimension label below marker
            # Add (F) suffix to finished dimensions
            width_text = opening['width']
            height_text = opening['height']

            width_is_finished = opening.get('width_is_finished', False)
            height_is_finished = opening.get('height_is_finished', False)

            if width_is_finished:
                width_text = f"{width_text}(F)"
            if height_is_finished:
                height_text = f"{height_text}(F)"

            dim_text = f"{width_text} W x {height_text} H"

            has_notation = 'notation' in opening and opening['notation']
            # Format notation with quotes and all caps, spell out abbreviations
            if has_notation:
                notation_value = opening["notation"].upper()
                # Spell out common abbreviations
                if notation_value == "NH":
                    notation_value = "NO HINGES"
                notation_text = f'"{notation_value}"'
            else:
                notation_text = None

            # Add finished size note if applicable (similar to NH notation)
            finished_note = None
            if width_is_finished and height_is_finished:
                finished_note = '"BOTH FINISHED SIZES"'
            elif width_is_finished:
                finished_note = '"WIDTH FINISHED SIZE"'
            elif height_is_finished:
                finished_note = '"HEIGHT FINISHED SIZE"'

            label_y = marker_y + 50

            # Calculate background size based on whether we have notation and finished note
            (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.7, 2)

            total_height = text_height
            total_width = text_width

            if has_notation:
                (notation_width, notation_height), _ = cv2.getTextSize(notation_text, font, 0.6, 2)
                total_width = max(total_width, notation_width)
                total_height += notation_height + 5

            if finished_note:
                (finished_width, finished_height), _ = cv2.getTextSize(finished_note, font, 0.6, 2)
                total_width = max(total_width, finished_width)
                total_height += finished_height + 5

            # Background for dimension text (sized to fit all text)
            cv2.rectangle(vis_image,
                         (marker_x - total_width//2 - 3, label_y - text_height - 3),
                         (marker_x + total_width//2 + 3, label_y + total_height - text_height + 3),
                         (255, 255, 255), -1)

            # Draw dimension text
            cv2.putText(vis_image, dim_text,
                       (marker_x - text_width//2, label_y),
                       font, 0.7, color, 2)

            current_y = label_y + text_height + 5

            # Draw notation below if present (in bright red for visibility)
            if has_notation:
                notation_color = (0, 0, 255)  # Bright red (BGR)
                cv2.putText(vis_image, notation_text,
                           (marker_x - notation_width//2, current_y),
                           font, 0.6, notation_color, 2)
                current_y += notation_height + 5

            # Draw finished size note if present (same style as NH notation - red, quotes, all caps)
            if finished_note:
                finished_color = (0, 0, 255)  # Bright red (BGR) - same as NH notation
                (finished_width, finished_height), _ = cv2.getTextSize(finished_note, font, 0.6, 2)
                cv2.putText(vis_image, finished_note,
                           (marker_x - finished_width//2, current_y),
                           font, 0.6, finished_color, 2)

            # Add this marker to the placed markers list for collision avoidance
            placed_markers.append({
                'position': (marker_x, marker_y),
                'opening': opening,
                'radius': radius
            })

        # Add opening list panel/legend
        if room_name or overlay_info:
            legend_height = 60 + (len(paired_openings) * 35)

            # Calculate required width based on text content
            title_text = f"{room_name} - Finish Sizes"
            if overlay_info:
                title_text = f"{room_name} - Finish Sizes (including {overlay_info})"

            # Get title width
            (title_width, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            # Calculate max width needed for opening specifications
            max_spec_width = 0
            for opening in paired_openings:
                # Check if dimensions are finished sizes
                width_is_finished = opening.get('width_is_finished', False)
                height_is_finished = opening.get('height_is_finished', False)

                if overlay_info:
                    # Only add overlay to dimensions that are NOT finished
                    if width_is_finished:
                        finish_width = f"{opening['width']} (F)"
                    else:
                        finish_width = add_fraction_to_measurement(opening['width'], overlay_info)

                    if height_is_finished:
                        finish_height = f"{opening['height']} (F)"
                    else:
                        finish_height = add_fraction_to_measurement(opening['height'], overlay_info)

                    spec_text = f"{finish_width} W x {finish_height} H"
                else:
                    spec_text = f"{opening['width']} W x {opening['height']} H"
                    if width_is_finished:
                        spec_text = spec_text.replace(opening['width'], f"{opening['width']} (F)", 1)
                    if height_is_finished:
                        spec_text = spec_text.replace(opening['height'], f"{opening['height']} (F)", 1)

                if 'notation' in opening and opening['notation']:
                    notation_value = opening["notation"].upper()
                    # Spell out common abbreviations
                    if notation_value == "NH":
                        notation_value = "NO HINGES"
                    spec_text += f' "{notation_value}"'

                # Add finished size note to width calculation
                if width_is_finished and height_is_finished:
                    spec_text += ' "BOTH FINISHED SIZES"'
                elif width_is_finished:
                    spec_text += ' "WIDTH FINISHED SIZE"'
                elif height_is_finished:
                    spec_text += ' "HEIGHT FINISHED SIZE"'

                (spec_width, _), _ = cv2.getTextSize(spec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                max_spec_width = max(max_spec_width, spec_width)

            # Legend width = max of (title width, spec width + circle space) + padding
            legend_width = max(title_width + 20, max_spec_width + 60) + 20

            legend_x = 10
            legend_y = h - legend_height - 20

            # White background with black border
            cv2.rectangle(vis_image, (legend_x, legend_y),
                         (legend_x + legend_width, legend_y + legend_height),
                         (255, 255, 255), -1)
            cv2.rectangle(vis_image, (legend_x, legend_y),
                         (legend_x + legend_width, legend_y + legend_height),
                         (0, 0, 0), 2)

            # Title
            title_y = legend_y + 30
            cv2.putText(vis_image, title_text,
                       (legend_x + 10, title_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Opening list
            for i, opening in enumerate(paired_openings):
                opening_y = title_y + 35 + (i * 35)
                color = opening_colors[i % len(opening_colors)]

                # Calculate text size first for dynamic circle sizing
                legend_text = f"#{start_opening_number + i}"
                legend_font = cv2.FONT_HERSHEY_SIMPLEX
                legend_font_scale = 0.5
                legend_thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(legend_text, legend_font, legend_font_scale, legend_thickness)

                # Calculate dynamic radius based on text size (with proportional padding)
                # Use 40% padding for legend (slightly tighter than main markers)
                legend_padding_factor = 1.40
                legend_radius = int(max(text_w, text_h) * legend_padding_factor / 2)

                # Draw circle with number (dynamic size for better fit)
                circle_center_x = legend_x + 25
                circle_center_y = opening_y - 5
                cv2.circle(vis_image, (circle_center_x, circle_center_y), legend_radius, (255, 255, 255), -1)
                cv2.circle(vis_image, (circle_center_x, circle_center_y), legend_radius, color, 2)
                text_x = circle_center_x - text_w // 2
                text_y = circle_center_y + text_h // 2 - baseline // 2

                cv2.putText(vis_image, legend_text,
                           (text_x, text_y),
                           legend_font, legend_font_scale, color, legend_thickness)

                # Opening specification (calculate finish sizes with overlay)
                # Check if dimensions are finished sizes
                width_is_finished = opening.get('width_is_finished', False)
                height_is_finished = opening.get('height_is_finished', False)

                # Build dimension text with (F) suffix
                if overlay_info:
                    # Only add overlay to dimensions that are NOT finished
                    if width_is_finished:
                        finish_width = f"{opening['width']}(F)"
                    else:
                        finish_width = add_fraction_to_measurement(opening['width'], overlay_info)

                    if height_is_finished:
                        finish_height = f"{opening['height']}(F)"
                    else:
                        finish_height = add_fraction_to_measurement(opening['height'], overlay_info)

                    spec_text = f"{finish_width} W x {finish_height} H"
                else:
                    width_text = f"{opening['width']}(F)" if width_is_finished else opening['width']
                    height_text = f"{opening['height']}(F)" if height_is_finished else opening['height']
                    spec_text = f"{width_text} W x {height_text} H"

                # Add notation if present (in quotes, all caps, red color)
                notation_part = None
                if 'notation' in opening and opening['notation']:
                    notation_value = opening["notation"].upper()
                    # Spell out common abbreviations
                    if notation_value == "NH":
                        notation_value = "NO HINGES"
                    notation_part = f' "{notation_value}"'
                    spec_text += notation_part

                # Add finished size note (similar wording to marker notation)
                finished_note = None
                if width_is_finished and height_is_finished:
                    finished_note = ' "BOTH FINISHED SIZES"'
                elif width_is_finished:
                    finished_note = ' "WIDTH FINISHED SIZE"'
                elif height_is_finished:
                    finished_note = ' "HEIGHT FINISHED SIZE"'

                if finished_note:
                    spec_text += finished_note

                # Draw main spec text (black)
                spec_without_extras = spec_text
                if notation_part:
                    spec_without_extras = spec_without_extras.replace(notation_part, '')
                if finished_note:
                    spec_without_extras = spec_without_extras.replace(finished_note, '')

                cv2.putText(vis_image, spec_without_extras,
                           (legend_x + 50, opening_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                current_x = legend_x + 50
                (base_width, _), _ = cv2.getTextSize(spec_without_extras, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                current_x += base_width

                # Draw notation part in red if present
                if notation_part:
                    cv2.putText(vis_image, notation_part,
                               (current_x, opening_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)  # Red color
                    (notation_width, _), _ = cv2.getTextSize(notation_part, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    current_x += notation_width

                # Draw finished note in red if present
                if finished_note:
                    cv2.putText(vis_image, finished_note,
                               (current_x, opening_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)  # Red color


    # No need to combine images since panel is overlaid
    combined = vis_image

    # Add page number and timestamp at bottom
    from datetime import datetime
    timestamp = datetime.now().strftime("%m-%d-%Y %I:%M:%S %p Central")

    # Extract page number from filename if not provided
    if page_number is None:
        import re
        filename = os.path.basename(image_path)
        match = re.search(r'page_(\d+)', filename)
        if match:
            page_number = int(match.group(1))

    # Create footer text
    if page_number is not None:
        footer_text = f"Page {page_number} - {timestamp}"
    else:
        footer_text = timestamp

    # Position at bottom right
    footer_font = cv2.FONT_HERSHEY_SIMPLEX
    footer_scale = 0.5
    footer_thickness = 1
    (footer_w, footer_h), footer_baseline = cv2.getTextSize(footer_text, footer_font, footer_scale, footer_thickness)

    footer_x = w - footer_w - 10
    footer_y = h - 10

    # Add white background behind text
    cv2.rectangle(combined, (footer_x - 5, footer_y - footer_h - 5),
                 (footer_x + footer_w + 5, footer_y + footer_baseline + 5),
                 (255, 255, 255), -1)

    # Add text
    cv2.putText(combined, footer_text,
               (footer_x, footer_y),
               footer_font, footer_scale, (0, 0, 0), footer_thickness)

    # Save visualization
    if save_viz:
        output_path = image_path.replace('.png', '_test_viz.png')
        cv2.imwrite(output_path, combined)
        # Convert to Windows path for display if needed
        display_path = output_path.replace('/', '\\') if output_path.startswith('//') else output_path
        print(f"\n[SAVED] Visualization: {display_path}")

    return combined
