#!/usr/bin/env python3
"""
Main entry point for cabinet measurement detection.
Orchestrates the full detection pipeline.
"""

import sys
import os
import cv2
from dotenv import load_dotenv

# Fix Unicode issues on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()


class Tee:
    """Redirect stdout/stderr to both console and file"""
    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode, encoding='utf-8')
        self.terminal = sys.stdout if mode == 'w' else sys.stderr

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

# Add parent directory to path for shared utilities
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from shared_utils import (
    fraction_to_decimal,
    convert_opening_to_finished,
    parse_overlay_spec,
    calculate_sqft,
    calculate_summary,
    save_unified_json,
    get_current_date
)

# Import local modules
from text_detection import find_interest_areas, merge_close_centers
from measurement_verification import verify_measurement_at_center_with_logic
from line_detection import classify_measurements_by_lines
from measurement_pairing_v2 import pair_measurements_by_proximity
from visualization import create_visualization
from claude_verification import is_suspicious_measurement, verify_measurements_with_claude, apply_claude_corrections
from prompt_user_info import prompt_for_order_info
import json


def main(start_opening_number=1):
    """
    Main detection pipeline

    This is a STUB - Full implementation in original measurement_detector_test.py
    lines 3576-3854 (278 lines)

    The full pipeline:
        1. Find interest areas (text regions) using OCR
        2. Merge close centers to avoid duplicates
        3. Zoom and verify each measurement
        4. Classify measurements as width/height
        5. Pair measurements into cabinet openings
        6. Create visualization with markers
    """

    # Get image path from command line first (before logging setup)
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [start_opening_number]")
        print("Example: python main.py page_1.png")
        print("Example: python main.py page_2.png 5  # Start numbering at opening 5")
        return

    image_path = sys.argv[1]

    # Set up logging to file
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    log_dir = os.path.dirname(os.path.abspath(image_path))
    log_file = os.path.join(log_dir, f"{base_name}_debug.txt")

    # Redirect stdout to both console and file
    tee = Tee(log_file, 'w')
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        # Get API key
        api_key = os.getenv('GOOGLE_VISION_API_KEY')
        if not api_key:
            print("[ERROR] GOOGLE_VISION_API_KEY not found in environment")
            return

        if len(sys.argv) >= 3:
            try:
                start_opening_number = int(sys.argv[2])
                print(f"Starting opening numbers at: {start_opening_number}")
            except ValueError:
                print(f"[WARNING] Invalid start_opening_number '{sys.argv[2]}', using default of 1")
                start_opening_number = 1

        # Convert Windows network paths to Unix-style for OpenCV
        if image_path.startswith('\\\\'):
            image_path = image_path.replace('\\', '/')

        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return

        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # Phase 1: Find interest areas
        interest_areas, room_name, overlay_info, opencv_regions = find_interest_areas(image_path, api_key)

        if not interest_areas:
            print("[WARNING] No interest areas found")
            return
    
        print(f"\nFound {len(interest_areas)} initial interest areas")
    
        # Merge close centers
        merged_areas = merge_close_centers(interest_areas)
    
        # Phase 2: Verify measurements at each center
        print("\n=== PHASE 2: Zoom Verification ===")
        measurements_list = []
        measurement_texts = []
        measurement_logic = []
    
        for i, area in enumerate(merged_areas):
            print(f"\nCenter {i+1}/{len(merged_areas)}: ({area['center'][0]:.0f}, {area['center'][1]:.0f})")
            result = verify_measurement_at_center_with_logic(
                image_path,
                area['center'],
                area['bounds'],
                area['texts'],
                api_key,
                center_index=i,
                save_debug=True
            )
    
            if result and result[0] and result[0][0]:  # Check if measurement found
                measurement_value, bounds, notation, raw_ocr, is_finished = result[0]
                logic = result[1]
    
                measurement_texts.append(measurement_value)
                measurement_logic.append(logic)
    
                measurements_list.append({
                    'text': measurement_value,
                    'position': area['center'],
                    'bounds': bounds if bounds else area['bounds'],
                    'notation': notation if notation else None,
                    'is_finished_size': is_finished
                })
    
                print(f"  ✓ Verified: '{measurement_value}'")
                if notation:
                    print(f"    Special notation: {notation}")
                if is_finished:
                    print(f"    Finished size (F)")
            else:
                print(f"  ✗ No valid measurement")
    
        print(f"\n{len(measurement_texts)} measurements verified")
    
        # Recalculate bounds based on verified text dimensions
        print("\nRecalculating measurement bounds based on verified text...")
        for i, meas in enumerate(measurements_list):
            verified_text = meas['text']
            center_x, center_y = meas['position']
    
            # Use OpenCV to get actual text dimensions
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(verified_text, font, font_scale, thickness)
    
            # Calculate new bounds centered on existing position
            # Add some padding (20% of text dimensions)
            padding_w = int(text_w * 0.2)
            padding_h = int(text_h * 0.2)
    
            new_bounds = {
                'left': center_x - (text_w + padding_w) / 2,
                'right': center_x + (text_w + padding_w) / 2,
                'top': center_y - (text_h + padding_h) / 2,
                'bottom': center_y + (text_h + padding_h) / 2
            }
    
            old_width = meas['bounds']['right'] - meas['bounds']['left']
            new_width = new_bounds['right'] - new_bounds['left']
    
            if abs(old_width - new_width) > 20:  # Significant difference
                print(f"  M{i+1} '{verified_text}': {old_width:.0f}px → {new_width:.0f}px")
    
            meas['bounds'] = new_bounds
    
        # Phase 2.5: Claude verification of suspicious measurements
        if measurements_list:
            print("\n=== PHASE 2.5: Claude Verification ===")
            print("Checking for suspicious OCR readings...")
    
            # Collect suspicious measurements
            suspicious_measurements = []
            for i, meas in enumerate(measurements_list):
                is_suspicious, reason = is_suspicious_measurement(meas['text'])
                if is_suspicious:
                    # Find the debug image for this measurement by matching position
                    # Debug images are saved as: DEBUG_page{N}_M{i}_pos{x}x{y}_{text}_zoom3x.png
                    import re
                    import glob
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    # Remove underscore from page name (debug files use "page2" not "page_2")
                    base_name_clean = base_name.replace('_', '')
    
                    # Find debug image in same directory as source image
                    # Normalize path for glob (use forward slashes)
                    img_path_norm = image_path.replace('\\', '/')
                    image_dir = os.path.dirname(img_path_norm)
    
                    # Match by position (x, y) not by index, since some measurements get filtered
                    pos_x = int(meas['position'][0])
                    pos_y = int(meas['position'][1])
                    debug_pattern = f"{image_dir}/DEBUG_{base_name_clean}_M*_pos{pos_x}x{pos_y}_*_zoom3x.png"
                    debug_files = glob.glob(debug_pattern)
    
                    if debug_files:
                        # Filter out HSV images, prefer regular zoom
                        non_hsv_files = [f for f in debug_files if 'hsv' not in f.lower()]
                        debug_image = non_hsv_files[0] if non_hsv_files else debug_files[0]
                        suspicious_measurements.append({
                            'text': meas['text'],
                            'debug_image': debug_image,
                            'index': i,
                            'reason': reason
                        })
                        print(f"  [{i}] '{meas['text']}' at ({pos_x}, {pos_y}) - {reason}")
                    else:
                        print(f"  [{i}] '{meas['text']}' at ({pos_x}, {pos_y}) - {reason} (no debug image found)")
    
            # Verify with Claude if any suspicious measurements found
            if suspicious_measurements:
                corrections = verify_measurements_with_claude(suspicious_measurements, image_dir)
                if corrections:
                    applied = apply_claude_corrections(measurements_list, corrections)
                    print(f"\n{applied} measurements corrected by Claude")
            else:
                print("  No suspicious measurements found - all OCR looks good!")
    
        # Phase 3: Classify measurements (WIDTH/HEIGHT/UNCLASSIFIED)
        measurement_categories = None
        classified = None
        if measurements_list:
            print("\n=== PHASE 3: Classifying Measurements ===")
            print("Finding dimension lines near each measurement...")
    
            # Convert Windows network paths to Unix-style for OpenCV
            img_path = image_path
            if img_path.startswith('\\\\'):
                img_path = img_path.replace('\\', '/')
    
            # Load image for line detection
            image_cv = cv2.imread(img_path)
            if image_cv is not None:
                classified, measurement_categories = classify_measurements_by_lines(image_cv, measurements_list)
    
                print(f"\nClassification Results:")
                print(f"  WIDTH measurements: {len(classified['width'])}")
                for w in classified['width']:
                    print(f"    - {w}")
                print(f"  HEIGHT measurements: {len(classified['height'])}")
                for h in classified['height']:
                    print(f"    - {h}")
                if classified['unclassified']:
                    print(f"  UNCLASSIFIED: {len(classified['unclassified'])}")
                    for u in classified['unclassified']:
                        print(f"    - {u}")
    
        # Phase 4: Pair measurements into cabinet openings
        paired_openings = []
        unpaired_heights_info = []
        if measurement_categories and measurements_list:
            print("\n=== PHASE 4: Pairing Measurements into Cabinet Openings ===")
            paired_openings, unpaired_heights_info = pair_measurements_by_proximity(measurement_categories, measurements_list, image_cv)
    
            if paired_openings:
                print("\nCABINET OPENING SPECIFICATIONS:")
                print("-" * 60)
                for i, opening in enumerate(paired_openings, start_opening_number):
                    print(f"Opening {i}: {opening['width']} W × {opening['height']} H")
                    print(f"  Width at: ({opening['width_pos'][0]:.0f}, {opening['width_pos'][1]:.0f})")
                    print(f"  Height at: ({opening['height_pos'][0]:.0f}, {opening['height_pos'][1]:.0f})")
                    print(f"  Pairing distance: {opening['distance']:.0f}px")
            else:
                print("\nNo openings could be paired from the measurements")
    
        # Summary
        print("\n" + "=" * 80)
        print("RESULTS:")
        print("-" * 60)
    
        if room_name:
            print(f"Room: {room_name}")
        if overlay_info:
            print(f"Overlay: {overlay_info}")
    
        # Count measurements
        from collections import Counter
        counts = Counter(m['text'] for m in measurements_list)
    
        print(f"\nTOTAL MEASUREMENTS: {len(measurements_list)}")
    
        # Show categorized counts if available
        if classified:
            print(f"  Widths: {len(classified['width'])}")
            print(f"  Heights: {len(classified['height'])}")
            print(f"  Unclassified: {len(classified['unclassified'])}")
    
        print("\nMeasurement Counts:")
        for meas, count in sorted(counts.items()):
            # Find category for this measurement
            category = ""
            if classified:
                if meas in classified['width']:
                    category = " [WIDTH]"
                elif meas in classified['height']:
                    category = " [HEIGHT]"
                else:
                    category = " [UNCLASS]"
            print(f"  {count}x {meas}{category}")
    
        # Phase 5: Create visualization
        print("\n=== PHASE 5: Creating Visualization ===")
        create_visualization(
            image_path, merged_areas, measurement_texts, measurement_logic,
            save_viz=True,
            opencv_regions=opencv_regions,
            measurement_categories=measurement_categories,
            measurements_list=measurements_list,
            paired_openings=paired_openings,
            show_groups=False,  # Don't show initial groups
            show_opencv=False,  # Don't show OpenCV regions
            show_line_rois=True,  # Show line detection ROIs
            show_panel=True,  # Show info panel
            show_pairing=True,  # Show paired openings
            show_classification=True,  # Show WIDTH/HEIGHT labels
            room_name=room_name,
            overlay_info=overlay_info,
            unpaired_heights_info=unpaired_heights_info,
            start_opening_number=start_opening_number
        )
    
        # Save pairing results to JSON
        if paired_openings:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(os.path.abspath(image_path))
    
            results_data = {
                'room_name': room_name if room_name else "",
                'overlay_info': overlay_info if overlay_info else "",
                'total_measurements': len(measurements_list),
                'total_openings': len(paired_openings),
                'measurements': {
                    'widths': classified['width'] if classified else [],
                    'heights': classified['height'] if classified else [],
                    'unclassified': classified['unclassified'] if classified else []
                },
                'openings': [
                    {
                        'number': i,
                        'width': opening['width'],
                        'height': opening['height'],
                        'specification': f"{opening['width']} W × {opening['height']} H" +
                                       (" NO HINGES" if opening.get('notation') == 'NH' else ""),
                        'width_position': opening['width_pos'],
                        'height_position': opening['height_pos'],
                        'pairing_distance': opening['distance'],
                        'width_is_finished': opening.get('width_is_finished', False),
                        'height_is_finished': opening.get('height_is_finished', False),
                        **({'notation': opening['notation']} if 'notation' in opening else {})
                    }
                    for i, opening in enumerate(paired_openings, start_opening_number)
                ]
            }
    
            output_json = os.path.join(output_dir, f"{base_name}_cabinet_openings.json")
            with open(output_json, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\n[SAVED] Cabinet openings data: {output_json}")
    
            # Also save in unified format
            unified_data = convert_inhouse_to_unified(
                results_data,
                image_path,
                room_name,
                overlay_info
            )
            unified_json_path = os.path.join(output_dir, f"{base_name}_unified_door_order.json")
            save_unified_json(unified_data, unified_json_path)
    
            print("\n" + "=" * 80)
            print("PIPELINE COMPLETE")
            print("=" * 80)

    finally:
        # Restore stdout and close log file
        sys.stdout = old_stdout
        tee.close()
        print(f"\n[SAVED] Debug log: {log_file}")


def convert_inhouse_to_unified(inhouse_data, original_file_path, room_name, overlay_info):
    """
    Convert Inhouse cabinet openings format to Unified Door Order format

    Args:
        inhouse_data: Dictionary in Inhouse format
        original_file_path: Path to original image file
        room_name: Room name from detection
        overlay_info: Overlay specification (e.g., "5/8 OL")

    Returns:
        dict: Data in unified format
    """
    # Prompt user for missing order information
    order_info_user, specifications, _ = prompt_for_order_info(original_file_path)

    # Parse overlay to get decimal value
    overlay_decimal = parse_overlay_spec(overlay_info) if overlay_info else 0.625  # Default 5/8"

    # Load config for drawer height threshold
    config_path = os.path.join(os.path.dirname(__file__), 'inhouse_defaults.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            drawer_threshold = config.get('drawer_height_threshold', 10.0)
    else:
        drawer_threshold = 10.0

    # Convert openings to doors/drawers with finished sizes
    doors = []
    drawers = []

    for opening in inhouse_data.get('openings', []):
        width_opening = opening.get('width', '')
        height_opening = opening.get('height', '')

        # Skip invalid measurements (OCR errors like "or 2")
        try:
            # Convert opening sizes to finished sizes
            width_finished, width_dec = convert_opening_to_finished(width_opening, overlay_decimal)
            height_finished, height_dec = convert_opening_to_finished(height_opening, overlay_decimal)
        except (ValueError, AttributeError) as e:
            print(f"[WARNING] Skipping invalid opening #{opening.get('number', '?')}: {width_opening} x {height_opening} - {e}")
            continue

        # Create item structure
        item = {
            "marker": f"#{opening.get('number', '')}",
            "qty": 1,  # Inhouse openings are individual
            "width": width_finished,
            "height": height_finished,
            "width_decimal": width_dec,
            "height_decimal": height_dec,
            "location": room_name if room_name else ""
        }

        # Classify as door or drawer based on height
        if height_dec <= drawer_threshold:
            drawers.append(item)
        else:
            item["sqft"] = calculate_sqft(width_dec, height_dec)
            doors.append(item)

    # Build unified format
    unified_data = {
        "schema_version": "1.0",
        "source": {
            "type": "inhouse",
            "original_file": os.path.basename(original_file_path),
            "extraction_date": get_current_date(),
            "extractor_version": "1.0"
        },
        "order_info": {
            "customer_company": order_info_user.get('customer_company', ''),
            "jobsite": order_info_user.get('jobsite', ''),
            "room": room_name if room_name else "",
            "submitted_by": order_info_user.get('submitted_by', ''),
            "order_date": get_current_date()
        },
        "specifications": specifications,
        "size_info": {
            "all_sizes_are_finished": True,
            "conversion_notes": f"Converted from opening sizes using {overlay_info if overlay_info else '5/8'} overlay"
        },
        "doors": doors,
        "drawers": drawers,
        "summary": calculate_summary(doors, drawers)
    }

    return unified_data


if __name__ == "__main__":
    main()
