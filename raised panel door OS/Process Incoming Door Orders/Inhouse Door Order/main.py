#!/usr/bin/env python3
"""
Main entry point for cabinet measurement detection.
Orchestrates the full detection pipeline.
"""

import sys
import os
from dotenv import load_dotenv

# Fix Unicode issues on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Import modules
from text_detection import find_interest_areas, merge_close_centers
from measurement_verification import verify_measurement_at_center_with_logic
from line_detection import classify_measurements_by_lines
from measurement_pairing import pair_measurements_by_proximity
from visualization import create_visualization
from claude_verification import is_suspicious_measurement, verify_measurements_with_claude, apply_claude_corrections


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

    # Get API key
    api_key = os.getenv('GOOGLE_VISION_API_KEY')
    if not api_key:
        print("[ERROR] GOOGLE_VISION_API_KEY not found in environment")
        return

    # Get image path from command line
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [start_opening_number]")
        print("Example: python main.py page_1.png")
        print("Example: python main.py page_2.png 5  # Start numbering at opening 5")
        return

    image_path = sys.argv[1]
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
        import cv2
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
        paired_openings, unpaired_heights_info = pair_measurements_by_proximity(measurement_categories, measurements_list)

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
        import json
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

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
