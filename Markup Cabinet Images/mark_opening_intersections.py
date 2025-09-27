"""
Mark Opening Intersections
Uses width X position and height Y position to find intersection points inside openings
"""
import json
import cv2
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

def expand_notation_abbreviation(notation_text):
    """Expand common abbreviations to full descriptions"""
    abbreviation_map = {
        'NH': 'NO HINGES',
        'SC': 'SOFT CLOSE',
        'PTO': 'PUSH TO OPEN',
        'FU': 'FLIP UP',
        'MW': 'MICROWAVE',
        'LS': 'LAZY SUSAN',
        'TPO': 'TRASH PULL OUT',
        'DB': 'DRAWER BOX',
        'RO': 'ROLL OUT',
    }

    # Check if it's an abbreviation we recognize (case insensitive)
    upper_notation = notation_text.upper()
    if upper_notation in abbreviation_map:
        return abbreviation_map[upper_notation]

    # If not an abbreviation, return as-is but capitalize properly
    return notation_text.upper()

def load_data(base_name, image_dir):
    """Load measurement results and pairing results"""
    import os
    # Load measurement positions and classifications
    with open(os.path.join(image_dir, f'{base_name}_measurements_data.json'), 'r') as f:
        measurement_data = json.load(f)

    # Load proximity pairing results
    with open(os.path.join(image_dir, f'{base_name}_openings_data.json'), 'r') as f:
        pairing_data = json.load(f)

    # Map finished_size flags from measurements to openings
    finished_sizes = {}
    for meas in measurement_data.get('measurements', []):
        if meas.get('finished_size', False):
            finished_sizes[meas['text']] = True

    # Add finished_size flags to each opening (track width and height separately)
    for opening in pairing_data.get('openings', []):
        opening['width_finished'] = finished_sizes.get(opening['width'], False)
        opening['height_finished'] = finished_sizes.get(opening['height'], False)
        # Overall flag for backward compatibility
        opening['finished_size'] = opening['width_finished'] or opening['height_finished']

    return measurement_data, pairing_data

def find_measurement_positions(measurements, text):
    """Find the position of a specific measurement text"""
    for meas in measurements:
        if meas['text'] == text:
            return meas['position']
    return None

def find_clear_position_for_marker(intersection_x, intersection_y, width_pos, height_pos, all_measurements, radius=45):
    """
    Find a clear position for the opening number marker that doesn't overlap with text.
    Try different offsets: left, right, above-left, above-right, below-left, below-right
    """
    # Collect all text boundaries to avoid
    avoid_regions = []

    # Helper to add a measurement's bounds to avoid regions
    def add_measurement_bounds(meas_text, pos):
        if pos:
            # Find the measurement in all_measurements to get its bounds
            for m in all_measurements:
                if m['text'] == meas_text and 'bounds' in m:
                    left_bound, right_bound = m['bounds']
                    # Create a region from left to right bound, with vertical padding
                    avoid_regions.append({
                        'left': left_bound,
                        'right': right_bound,
                        'top': pos[1] - 20,
                        'bottom': pos[1] + 20,
                        'center': pos
                    })
                    return
            # Fallback if bounds not found
            avoid_regions.append({
                'left': pos[0] - 70,
                'right': pos[0] + 70,
                'top': pos[1] - 20,
                'bottom': pos[1] + 20,
                'center': pos
            })

    # Extract width and height text from the opening specification if provided
    width_text = None
    height_text = None

    # Check if we have opening info passed in (will have 'specification' key)
    opening_info = None
    for item in all_measurements:
        if 'specification' in item:
            opening_info = item
            # Extract width and height from specification
            parts = item['specification'].replace('×', 'x').split(' x ')
            if len(parts) == 2:
                width_text = item.get('width', parts[0].strip())
                height_text = item.get('height', parts[1].strip())
            break

    # Add measurement regions to avoid
    if width_text and width_pos:
        add_measurement_bounds(width_text, width_pos)
    if height_text and height_pos:
        add_measurement_bounds(height_text, height_pos)

    # Also add all other measurements to avoid
    for m in all_measurements:
        if 'text' in m and 'position' in m:
            # Skip the width and height of current opening
            if m['text'] != width_text and m['text'] != height_text:
                add_measurement_bounds(m['text'], m['position'])

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

        # Check if marker would overlap with any text region
        marker_left = test_x - radius
        marker_right = test_x + radius
        marker_top = test_y - radius
        marker_bottom = test_y + radius

        for region in avoid_regions:
            # Check for overlap between marker and text region
            if (marker_left < region['right'] and marker_right > region['left'] and
                marker_top < region['bottom'] and marker_bottom > region['top']):
                # Calculate overlap amount
                overlap_x = min(marker_right, region['right']) - max(marker_left, region['left'])
                overlap_y = min(marker_bottom, region['bottom']) - max(marker_top, region['top'])
                overlap_score += overlap_x * overlap_y

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

        # Use the positions saved in the openings data (from proximity pairing)
        # This ensures we use the exact positions that were paired, not the first match
        if 'width_pos' in opening and 'height_pos' in opening:
            width_pos = opening['width_pos']
            height_pos = opening['height_pos']
        else:
            # Fallback to old method if positions not saved (backward compatibility)
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

            intersection_points.append({
                'opening': opening,
                'intersection': (intersection_x, intersection_y),
                'width_pos': width_pos,
                'height_pos': height_pos
            })

            # Find a clear position for the opening number marker
            # Pass the opening info along with all measurements
            current_opening_info = {
                'specification': opening['specification'],
                'width': opening['width'],
                'height': opening['height']
            }
            all_measurements_with_opening = measurements + [current_opening_info]

            marker_x, marker_y = find_clear_position_for_marker(
                intersection_x, intersection_y,
                width_pos, height_pos,
                all_measurements_with_opening,
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
            # Include F suffix if present
            width_text = opening['width']
            if opening.get('width_finished', False):
                width_text += "F"
            height_text = opening['height']
            if opening.get('height_finished', False):
                height_text += "F"
            dim_text = f"{width_text} x {height_text}"

            # Add customer notations if present
            notation_text = ""
            if 'notations' in opening and opening['notations']:
                # Expand abbreviations to full text
                expanded_notations = [expand_notation_abbreviation(n) for n in opening['notations']]
                notation_text = ", ".join(expanded_notations)

            label_y = marker_y + 55  # Below the marker circle

            # Background for dimension text
            (text_width, text_height), _ = cv2.getTextSize(dim_text, font, 0.5, 1)
            cv2.rectangle(annotated,
                         (marker_x - text_width//2 - 3, label_y - text_height - 3),
                         (marker_x + text_width//2 + 3, label_y + 3),
                         (255, 255, 255), -1)

            # Draw dimension text
            cv2.putText(annotated, dim_text,
                       (marker_x - text_width//2, label_y),
                       font, 0.5, color, 1)

            # Draw notation text below dimensions if present
            if notation_text:
                notation_y = label_y + 20  # Below the dimensions

                # Background for notation text
                (notation_width, notation_height), _ = cv2.getTextSize(notation_text, font, 0.45, 1)
                cv2.rectangle(annotated,
                             (marker_x - notation_width//2 - 3, notation_y - notation_height - 3),
                             (marker_x + notation_width//2 + 3, notation_y + 3),
                             (255, 255, 255), -1)

                # Draw notation text in red
                cv2.putText(annotated, notation_text,
                           (marker_x - notation_width//2, notation_y),
                           font, 0.45, color, 1)

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

    # Parse overlay amount if found (e.g., "5/8 OL" -> 0.625)
    overlay_amount = 0
    if overlay_info:
        import re
        # Look for fraction pattern before OL
        match = re.search(r'(\d+)/(\d+)\s+OL', overlay_info)
        if match:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            overlay_amount = numerator / denominator

    # Add legend with red indicators - LOWER LEFT with BIGGER TEXT
    image_height = annotated.shape[0]
    # Reduced height since overlay is now in title, not separate line
    legend_height = 50 + (len(openings) * 40)  # Base height
    # Leave about 1 inch (96 pixels) at the bottom of the page
    legend_y_start = image_height - legend_height - 100  # 100px from bottom for 1"+ margin

    # Check if we need more width for notations and F note
    max_text_width = 750
    for opening in openings:
        # Build the text as it will appear
        width_text = opening['width']
        if opening.get('width_finished', False):
            width_text += "F"  # No space before F
        height_text = opening['height']
        if opening.get('height_finished', False):
            height_text += "F"  # No space before F
        test_text = f"#{opening['number']}: {width_text} W x {height_text} H"

        # Add inline F note if needed
        if opening.get('finished_size', False):
            test_text += " - F = Finished size, no overlay added"

        # Add customer notations
        if 'notations' in opening and opening['notations']:
            expanded_notations = [expand_notation_abbreviation(n) for n in opening['notations']]
            test_text += f" ({', '.join(expanded_notations)})"

        # Estimate text width (approximately 12 pixels per character at scale 0.7)
        text_width_estimate = len(test_text) * 12 + 100  # Add margin for circle
        if text_width_estimate > max_text_width:
            max_text_width = text_width_estimate

    legend_width = max_text_width  # Use calculated width

    # White background with black border
    legend_bottom = legend_y_start + legend_height
    cv2.rectangle(annotated, (10, legend_y_start), (10 + legend_width, legend_bottom), (255, 255, 255), -1)
    cv2.rectangle(annotated, (10, legend_y_start), (10 + legend_width, legend_bottom), (0, 0, 0), 3)

    # Title with overlay info and room name - BIGGER TEXT
    title_y = legend_y_start + 35
    if overlay_info:
        # When overlay is found, change title to "Cabinet Finish Sizes"
        title_text = f"CABINET FINISH SIZES for {overlay_info}"
        if room_name:
            title_text = f"{room_name} - {title_text}"
    else:
        # No overlay, keep original title
        title_text = "CABINET OPENINGS"
        if room_name:
            title_text = f"{room_name} - {title_text}"
    cv2.putText(annotated, title_text,
               (20, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Helper function to parse and add overlay to dimension
    def add_overlay_to_dimension(dim_text, overlay_amt):
        """Add overlay amount to a dimension like '33 3/8' """
        if overlay_amt == 0:
            return dim_text

        # Parse the dimension
        import re
        match = re.match(r'(\d+)\s*(\d+/\d+)?', dim_text)
        if match:
            whole = int(match.group(1))
            fraction = match.group(2)

            # Convert to decimal
            total = float(whole)
            if fraction:
                parts = fraction.split('/')
                total += float(parts[0]) / float(parts[1])

            # Add overlay
            total += overlay_amt

            # Convert back to fraction format
            whole_part = int(total)
            decimal_part = total - whole_part

            # Find closest common fraction
            if decimal_part < 0.0625:  # Less than 1/16
                return str(whole_part)
            elif decimal_part < 0.1875:  # 1/8
                return f"{whole_part} 1/8"
            elif decimal_part < 0.3125:  # 1/4
                return f"{whole_part} 1/4"
            elif decimal_part < 0.4375:  # 3/8
                return f"{whole_part} 3/8"
            elif decimal_part < 0.5625:  # 1/2
                return f"{whole_part} 1/2"
            elif decimal_part < 0.6875:  # 5/8
                return f"{whole_part} 5/8"
            elif decimal_part < 0.8125:  # 3/4
                return f"{whole_part} 3/4"
            elif decimal_part < 0.9375:  # 7/8
                return f"{whole_part} 7/8"
            else:
                return str(whole_part + 1)

        return dim_text

    # Opening specifications - BIGGER TEXT
    for i, opening in enumerate(openings):
        y_pos = title_y + 40 + (i * 40)  # More spacing for bigger text

        # Draw red circle indicator with white fill (printer friendly) - BIGGER
        cv2.circle(annotated, (35, y_pos - 8), 12, (255, 255, 255), -1)  # White fill
        cv2.circle(annotated, (35, y_pos - 8), 12, color, 3)  # Red outline

        # Process specification - add overlay if found and not a finished size
        if overlay_amount > 0 and not opening.get('finished_size', False):
            # Add overlay to both width and height
            width_with_overlay = add_overlay_to_dimension(opening['width'], overlay_amount)
            height_with_overlay = add_overlay_to_dimension(opening['height'], overlay_amount)
            spec_text = f"{width_with_overlay} W x {height_with_overlay} H"
        else:
            # No overlay (either none specified or it's a finished size)
            # Build spec with (F) indicators inline
            width_text = opening['width']
            if opening.get('width_finished', False):
                width_text += "F"  # No space before F
            height_text = opening['height']
            if opening.get('height_finished', False):
                height_text += "F"  # No space before F
            spec_text = f"{width_text} W x {height_text} H"

        text = f"#{opening['number']}: {spec_text}"

        # If this opening has F, add the note inline
        if opening.get('finished_size', False):
            text += " - F = Finished size, no overlay added"

        # Add customer notations if present
        if 'notations' in opening and opening['notations']:
            # Expand abbreviations to full text
            expanded_notations = [expand_notation_abbreviation(n) for n in opening['notations']]
            notations_text = ", ".join(expanded_notations)
            text += f" ({notations_text})"

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