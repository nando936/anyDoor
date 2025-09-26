"""
Mark Opening Intersections
Uses width X position and height Y position to find intersection points inside openings
"""
import json
import cv2
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

def load_data(base_name, image_dir):
    """Load measurement results and pairing results"""
    import os
    # Load measurement positions and classifications
    with open(os.path.join(image_dir, f'{base_name}_measurements_data.json'), 'r') as f:
        measurement_data = json.load(f)

    # Load proximity pairing results
    with open(os.path.join(image_dir, f'{base_name}_openings_data.json'), 'r') as f:
        pairing_data = json.load(f)

    return measurement_data, pairing_data

def find_measurement_positions(measurements, text):
    """Find the position of a specific measurement text"""
    for meas in measurements:
        if meas['text'] == text:
            return meas['position']
    return None

def find_clear_position_for_marker(intersection_x, intersection_y, width_pos, height_pos, all_measurement_positions, radius=45):
    """
    Find a clear position for the opening number marker that doesn't overlap with text.
    Try different offsets: left, right, above-left, above-right, below-left, below-right
    """
    # Collect all text/measurement positions to avoid
    avoid_positions = []

    # Add width and height positions
    if width_pos:
        avoid_positions.append(width_pos)
    if height_pos:
        avoid_positions.append(height_pos)

    # Add all other measurement positions
    for pos in all_measurement_positions:
        if pos:
            avoid_positions.append(pos)

    # Try different offset positions in order of preference
    offset_distance = 80  # Distance to offset the marker
    test_positions = [
        (intersection_x - offset_distance, intersection_y),  # Left
        (intersection_x + offset_distance, intersection_y),  # Right
        (intersection_x - offset_distance, intersection_y - offset_distance),  # Upper-left
        (intersection_x + offset_distance, intersection_y - offset_distance),  # Upper-right
        (intersection_x - offset_distance, intersection_y + offset_distance),  # Lower-left
        (intersection_x + offset_distance, intersection_y + offset_distance),  # Lower-right
        (intersection_x, intersection_y - offset_distance),  # Above
        (intersection_x, intersection_y + offset_distance),  # Below
    ]

    # Find the position with minimum overlap
    best_position = (intersection_x, intersection_y)  # Default to center if no better position
    min_overlap_score = float('inf')

    for test_x, test_y in test_positions:
        overlap_score = 0

        # Check distance to each position we want to avoid
        for avoid_x, avoid_y in avoid_positions:
            distance = ((test_x - avoid_x) ** 2 + (test_y - avoid_y) ** 2) ** 0.5

            # If too close (within the marker radius + buffer), add to overlap score
            if distance < radius + 30:  # 30 pixel buffer
                overlap_score += (radius + 30 - distance)

        # Keep the position with minimum overlap
        if overlap_score < min_overlap_score:
            min_overlap_score = overlap_score
            best_position = (test_x, test_y)

            # If we found a position with no overlap, use it immediately
            if overlap_score == 0:
                break

    return int(best_position[0]), int(best_position[1])

def mark_intersections(image_path, measurement_data, pairing_data):
    """
    Mark the intersection of width X and height Y positions
    This should land inside the actual opening
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        return

    annotated = image.copy()

    # Get measurements and openings
    measurements = measurement_data['measurements']
    openings = pairing_data['openings']

    print("=" * 80)
    print("MARKING WIDTH-HEIGHT INTERSECTIONS")
    print("-" * 60)
    print("Using: Width's X position × Height's Y position")
    print("=" * 80)

    # Use red for all markings (BGR format)
    color = (0, 0, 255)  # Red

    intersection_points = []

    for i, opening in enumerate(openings):

        # Get width and height measurement positions
        width_pos = find_measurement_positions(measurements, opening['width'])
        height_pos = find_measurement_positions(measurements, opening['height'])

        if width_pos and height_pos:
            # Take X from width, Y from height
            intersection_x = int(width_pos[0])  # Width's X position
            intersection_y = int(height_pos[1])  # Height's Y position

            print(f"\nOpening {opening['number']}: {opening['specification']}")
            print(f"  Width '{opening['width']}' position: ({width_pos[0]:.0f}, {width_pos[1]:.0f})")
            print(f"  Height '{opening['height']}' position: ({height_pos[0]:.0f}, {height_pos[1]:.0f})")
            print(f"  → Intersection (Width X, Height Y): ({intersection_x}, {intersection_y})")

            # Collect all measurement positions for this image
            all_measurement_positions = []
            for m in measurements:
                if 'position' in m:
                    all_measurement_positions.append(m['position'])

            intersection_points.append({
                'opening': opening,
                'intersection': (intersection_x, intersection_y),
                'width_pos': width_pos,
                'height_pos': height_pos
            })

            # Find a clear position for the opening number marker
            marker_x, marker_y = find_clear_position_for_marker(
                intersection_x, intersection_y,
                width_pos, height_pos,
                all_measurement_positions,
                radius=45
            )

            # Draw small cross at the actual intersection point
            cross_size = 10
            cv2.line(annotated, (intersection_x - cross_size, intersection_y),
                    (intersection_x + cross_size, intersection_y), color, 2)
            cv2.line(annotated, (intersection_x, intersection_y - cross_size),
                    (intersection_x, intersection_y + cross_size), color, 2)

            # Draw white filled circle with red outline at the clear position
            cv2.circle(annotated, (marker_x, marker_y), 45, (255, 255, 255), -1)  # White fill
            cv2.circle(annotated, (marker_x, marker_y), 45, color, 3)  # Red outline

            # Draw opening number at marker position (much bigger text with #)
            text = f"#{opening['number']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.8  # Much bigger text
            thickness = 3     # Thick text

            # Get text size for centering
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Draw red text
            cv2.putText(annotated, text,
                       (marker_x - text_width//2, marker_y + text_height//2),
                       font, font_scale, color, thickness)

            # If marker was moved, draw a line connecting it to the intersection
            if marker_x != intersection_x or marker_y != intersection_y:
                cv2.line(annotated, (intersection_x, intersection_y),
                        (marker_x, marker_y), color, 1, cv2.LINE_AA)

            # Draw lines from intersection to measurements (thin lines)
            # Line to width position
            cv2.line(annotated, (intersection_x, intersection_y),
                    (int(width_pos[0]), int(width_pos[1])), color, 1, cv2.LINE_AA)

            # Line to height position
            cv2.line(annotated, (intersection_x, intersection_y),
                    (int(height_pos[0]), int(height_pos[1])), color, 1, cv2.LINE_AA)

            # Mark measurement positions with small circles
            cv2.circle(annotated, (int(width_pos[0]), int(width_pos[1])), 5, color, 2)
            cv2.circle(annotated, (int(height_pos[0]), int(height_pos[1])), 5, color, 2)

            # Add dimension labels near the marker (use 'x' not '×' to avoid encoding issues)
            dim_text = f"{opening['width']} x {opening['height']}"
            label_y = marker_y + 55  # Below the marker circle

            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.5, 1)
            cv2.rectangle(annotated,
                         (marker_x - text_width//2 - 3, label_y - text_height - 3),
                         (marker_x + text_width//2 + 3, label_y + 3),
                         (255, 255, 255), -1)

            # Draw dimension text
            cv2.putText(annotated, dim_text,
                       (marker_x - text_width//2, label_y),
                       font, 0.5, color, 1)

            # Draw crosshair at intersection
            cross_size = 30
            cv2.line(annotated,
                    (intersection_x - cross_size, intersection_y),
                    (intersection_x + cross_size, intersection_y),
                    (0, 0, 0), 1)
            cv2.line(annotated,
                    (intersection_x, intersection_y - cross_size),
                    (intersection_x, intersection_y + cross_size),
                    (0, 0, 0), 1)

    # Look for hinge/overlay notations and room name in the measurement data
    overlay_info = measurement_data.get('overlay_info', '')
    room_name = measurement_data.get('room_name', '')

    # If not found, check individual measurements for OL notation
    if not overlay_info:
        for meas in measurement_data['measurements']:
            if 'OL' in meas['text'].upper():
                overlay_info = meas['text']
                break

    # Add legend with red indicators - LOWER LEFT with BIGGER TEXT
    image_height = annotated.shape[0]
    # Reduced height since overlay is now in title, not separate line
    legend_height = 50 + (len(openings) * 40)  # No extra space for overlay line
    # Leave about 1 inch (96 pixels) at the bottom of the page
    legend_y_start = image_height - legend_height - 100  # 100px from bottom for 1"+ margin
    legend_width = 650  # Wider to fit room name and overlay text

    # White background with black border
    legend_bottom = legend_y_start + legend_height
    cv2.rectangle(annotated, (10, legend_y_start), (10 + legend_width, legend_bottom), (255, 255, 255), -1)
    cv2.rectangle(annotated, (10, legend_y_start), (10 + legend_width, legend_bottom), (0, 0, 0), 3)

    # Title with overlay info and room name - BIGGER TEXT
    title_y = legend_y_start + 35
    title_text = "CABINET OPENINGS"
    if room_name:
        title_text = f"{room_name} - {title_text}"
    if overlay_info:
        title_text = f"{title_text} {overlay_info}"
    cv2.putText(annotated, title_text,
               (20, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Opening specifications - BIGGER TEXT
    for i, opening in enumerate(openings):
        y_pos = title_y + 40 + (i * 40)  # More spacing for bigger text

        # Draw red circle indicator with white fill (printer friendly) - BIGGER
        cv2.circle(annotated, (35, y_pos - 8), 12, (255, 255, 255), -1)  # White fill
        cv2.circle(annotated, (35, y_pos - 8), 12, color, 3)  # Red outline

        # Draw text (replace × with x to avoid encoding issues) - BIGGER
        spec_text = opening['specification'].replace('×', 'x')
        text = f"#{opening['number']}: {spec_text}"
        cv2.putText(annotated, text, (60, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


    # Get base name and directory from image path
    import os
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(os.path.abspath(image_path))

    # Save image with descriptive name including room name and range of openings
    num_openings = len(openings)
    if num_openings > 0:
        opening_range = f"1-{num_openings}" if num_openings > 1 else "1"
    else:
        opening_range = "0"

    # Add page number and timestamp at the bottom of the image
    from datetime import datetime
    import pytz

    # Get current time in CT (Central Time)
    ct_timezone = pytz.timezone('America/Chicago')
    current_time = datetime.now(ct_timezone)
    timestamp = current_time.strftime("%m-%d-%Y %I:%M %p CT")

    # Extract page number from base_name (e.g., "page_10" -> "Page 10")
    page_text = base_name.replace('_', ' ').title() if 'page' in base_name.lower() else base_name

    # Draw at bottom center of image
    bottom_text = f"{page_text} - {timestamp}"
    image_width = annotated.shape[1]
    image_height = annotated.shape[0]

    # Get text size for centering
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(bottom_text, font, 0.6, 1)
    text_x = (image_width - text_width) // 2
    text_y = image_height - 30  # 30 pixels from bottom

    # White background for text
    cv2.rectangle(annotated,
                 (text_x - 10, text_y - text_height - 5),
                 (text_x + text_width + 10, text_y + 5),
                 (255, 255, 255), -1)

    # Draw the text
    cv2.putText(annotated, bottom_text,
               (text_x, text_y),
               font, 0.6, (0, 0, 0), 1)

    # Include room name in filename if available
    if room_name:
        output_filename = f"{base_name}_{room_name}_openings_{opening_range}_marked.png"
    else:
        output_filename = f"{base_name}_openings_{opening_range}_marked.png"

    output_path = os.path.join(image_dir, output_filename)
    cv2.imwrite(output_path, annotated)
    print(f"\n[OK] Annotated image saved: {output_path}")

    return intersection_points

def main():
    import os
    print("=" * 80)
    print("OPENING INTERSECTION MARKER")
    print("Marks openings at Width X × Height Y intersections")
    print("=" * 80)
    print()

    # Get image path from command line or default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "page_16.png"

    # Get base name and directory
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(os.path.abspath(image_path))

    # Load data
    measurement_data, pairing_data = load_data(base_name, image_dir)

    # Mark intersections
    intersections = mark_intersections(image_path, measurement_data, pairing_data)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Intersection Points")
    print("-" * 80)
    for item in intersections:
        opening = item['opening']
        x, y = item['intersection']
        print(f"Opening {opening['number']}: Point at ({x}, {y})")
        print(f"  Should be inside: {opening['specification']} opening")

if __name__ == "__main__":
    main()