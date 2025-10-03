#!/usr/bin/env python3
"""
Image preprocessing functions for cabinet measurement detection.
Includes HSV color filtering and OpenCV text region detection.
"""

import cv2
import numpy as np
from measurement_config import HSV_CONFIG, ZOOM_CONFIG


def apply_hsv_preprocessing(image):
    """Apply HSV preprocessing to isolate green text on brown backgrounds"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array(HSV_CONFIG['lower_green'])
    upper_green = np.array(HSV_CONFIG['upper_green'])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    return mask_inv


def find_opencv_supplemental_regions_better(hsv_image, all_vision_detections):
    """Find text regions using better text detection methods - not lines!"""

    # Apply threshold to get binary image
    _, binary = cv2.threshold(hsv_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological gradient to enhance text edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    additional_regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Basic size filter for text
        if not (8 < w < 150 and 8 < h < 60):
            continue

        # Aspect ratio for text
        aspect_ratio = w / h
        if not (0.2 < aspect_ratio < 8):
            continue

        # Skip very small areas
        if area < 80:
            continue

        center_x = x + w//2
        center_y = y + h//2

        # Check if overlaps with Vision API detections
        is_covered = False
        for vision_item in all_vision_detections:
            if not (x + w < vision_item['x_min'] or
                    x > vision_item['x_max'] or
                    y + h < vision_item['y_min'] or
                    y > vision_item['y_max']):
                is_covered = True
                break

        if not is_covered:
            # Extract the region
            roi = binary[y:y+h, x:x+w]

            # Text validation using stroke analysis
            # Apply distance transform to get stroke widths
            dist_transform = cv2.distanceTransform(roi, cv2.DIST_L2, 3)

            # Get stroke width statistics
            if dist_transform.size > 0:
                non_zero = dist_transform[dist_transform > 0]
                if len(non_zero) > 0:
                    mean_stroke = np.mean(non_zero)
                    std_stroke = np.std(non_zero)

                    # Text has consistent stroke width (low std/mean ratio)
                    # Lines have very uniform stroke (very low ratio) or very high ratio
                    stroke_consistency = std_stroke / (mean_stroke + 0.001)

                    # Text typically has stroke consistency between 0.3 and 0.8
                    if not (0.25 < stroke_consistency < 0.85):
                        continue

                    # Text strokes are typically 2-15 pixels thick
                    if not (1.5 < mean_stroke < 15):
                        continue

            # Check edge density - text has moderate edge density
            edges = cv2.Canny(hsv_image[y:y+h, x:x+w], 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)

            # Text has moderate edge density (not too low like solid areas, not too high like noise)
            if not (0.05 < edge_density < 0.4):
                continue

            # Count connected components in the region
            num_labels, _ = cv2.connectedComponents(roi, connectivity=8)

            # Text regions typically have 2-20 components (letters/numbers)
            # Single component = likely a line or solid shape
            # Too many = likely noise
            if not (2 <= num_labels <= 20):
                continue

            # Passed all checks - likely text!
            additional_regions.append({
                'text': 'OPENCV',
                'center': (center_x, center_y),
                'x': center_x,
                'y': center_y,
                'source': 'opencv'
            })

    return additional_regions


def find_opencv_supplemental_regions(hsv_image, all_vision_detections):
    """Find additional text regions using OpenCV in areas Vision API missed

    Focus: Detect text regions while excluding lines using multiple validation techniques
    """

    # Apply threshold to get binary image
    _, binary = cv2.threshold(hsv_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    additional_regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Basic size filter for text (measurements are typically this size)
        if not (10 < w < 120 and 10 < h < 45):
            continue

        # Skip if area too small
        if area < 100:
            continue

        center_x = x + w//2
        center_y = y + h//2

        # Check if overlaps with Vision API detections or their zoom areas
        is_covered = False
        # Use the same padding that will be used for zoom ROI areas
        zoom_padding_h = ZOOM_CONFIG['padding_horizontal']
        zoom_padding_v = ZOOM_CONFIG['padding']

        for vision_item in all_vision_detections:
            # Expand Vision API bounds by zoom padding to exclude zoom ROI areas
            vision_left = vision_item['x_min'] - zoom_padding_h
            vision_right = vision_item['x_max'] + zoom_padding_h
            vision_top = vision_item['y_min'] - zoom_padding_v
            vision_bottom = vision_item['y_max'] + zoom_padding_v

            # Check if OpenCV region overlaps with expanded Vision area (includes zoom ROI)
            if not (x + w < vision_left or
                    x > vision_right or
                    y + h < vision_top or
                    y > vision_bottom):
                is_covered = True
                break

        if is_covered:
            continue

        # Extract region for analysis
        roi = binary[y:y+h, x:x+w]

        # === ADVANCED LINE DETECTION ===

        # 1. Hough Line Detection - if we can fit a single strong line through most pixels, it's a line
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=min(w, h)//2,
                                minLineLength=min(w, h)*0.7, maxLineGap=5)

        if lines is not None and len(lines) > 0:
            # Check if a single line covers most of the region
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                diagonal = np.sqrt(w**2 + h**2)

                # If a single line spans most of the diagonal, it's likely just a line
                if line_length > diagonal * 0.75:
                    continue  # Skip this region, it's a line

        # 2. Profile Analysis - check intensity profile perpendicular to main axis
        aspect_ratio = w / h

        if aspect_ratio > 2.5:  # Horizontal orientation
            # Check vertical profile - lines have uniform profile
            vertical_profile = np.mean(roi, axis=1)
            profile_std = np.std(vertical_profile)

            # Lines have very low standard deviation in perpendicular profile
            if profile_std < 10:
                continue  # It's a horizontal line

        elif aspect_ratio < 0.4:  # Vertical orientation
            # Check horizontal profile
            horizontal_profile = np.mean(roi, axis=0)
            profile_std = np.std(horizontal_profile)

            if profile_std < 10:
                continue  # It's a vertical line

        # 3. Skeleton Analysis - lines have simple skeletons
        skeleton = cv2.ximgproc.thinning(roi)

        # Count skeleton endpoints and junctions
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
        filtered = cv2.filter2D(skeleton, cv2.CV_8U, kernel)

        # Endpoints have value 11 (1 neighbor + 10 center)
        # Junctions have value > 12 (2+ neighbors + 10 center)
        endpoints = np.sum(filtered == 11)
        junctions = np.sum(filtered > 12)

        # Lines have exactly 2 endpoints and 0 junctions
        # Text has multiple endpoints and junctions
        if endpoints <= 2 and junctions == 0:
            # Simple structure - likely a line
            skeleton_points = np.sum(skeleton > 0)
            if skeleton_points > 0:
                # Check if skeleton is mostly straight
                skeleton_coords = np.column_stack(np.where(skeleton > 0))
                if len(skeleton_coords) > 2:
                    # Fit a line to skeleton points
                    vx, vy, cx, cy = cv2.fitLine(skeleton_coords, cv2.DIST_L2, 0, 0.01, 0.01)

                    # Calculate how well points fit the line
                    distances = []
                    for point in skeleton_coords:
                        # Distance from point to line
                        d = abs((point[1] - cx) * vy[0] - (point[0] - cy) * vx[0]) / np.sqrt(vx[0]**2 + vy[0]**2)
                        distances.append(d)

                    mean_dist = np.mean(distances)

                    # If skeleton points are very close to a straight line, it's a line
                    if mean_dist < 2:
                        continue

        # 4. Contour Complexity - text contours are more complex than line contours
        perimeter = cv2.arcLength(contour, True)
        complexity = perimeter**2 / (4 * np.pi * area) if area > 0 else 0

        # Lines have low complexity (close to 1 for perfect rectangle)
        if complexity < 1.5:
            continue

        # 5. Connected Components Analysis
        num_labels, labels = cv2.connectedComponents(roi, connectivity=8)

        # Single component often means a line or solid shape
        if num_labels <= 2:  # Only background and 1 component
            # But check if it could be a single digit like "5"
            # Single digits are small and have moderate complexity
            if w > 30 or h > 30:  # Too large for single digit
                continue

        # 6. Pixel Distribution - text has more varied distribution than lines
        # Calculate histogram of row and column sums
        row_sums = np.sum(roi, axis=1)
        col_sums = np.sum(roi, axis=0)

        # Coefficient of variation (CV) - std/mean
        # Lines have low CV, text has higher CV
        if len(row_sums) > 0 and np.mean(row_sums) > 0:
            row_cv = np.std(row_sums) / np.mean(row_sums)
        else:
            row_cv = 0

        if len(col_sums) > 0 and np.mean(col_sums) > 0:
            col_cv = np.std(col_sums) / np.mean(col_sums)
        else:
            col_cv = 0

        # Lines have very low CV in perpendicular direction
        if aspect_ratio > 2 and row_cv < 0.2:  # Horizontal line
            continue
        if aspect_ratio < 0.5 and col_cv < 0.2:  # Vertical line
            continue

        # Passed all line detection checks - this is likely text!
        additional_regions.append({
            'text': 'OPENCV',
            'center': (center_x, center_y),
            'x': center_x,
            'y': center_y,
            'source': 'opencv'
        })

    return additional_regions
