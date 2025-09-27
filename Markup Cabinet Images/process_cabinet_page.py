"""
Master Cabinet Page Processor
Runs all three scripts in sequence to process a cabinet drawing page
"""
import sys
import os
import subprocess

def process_page(image_path):
    """
    Process a cabinet page image through all three stages
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return False

    print("=" * 80)
    print("CABINET PAGE PROCESSOR")
    print("=" * 80)
    print(f"Processing: {image_path}\n")

    # Get the base name and directory for output files
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(os.path.abspath(image_path))

    # Step 1: Run measurement_based_detector.py
    print("[1/3] Running measurement detection and classification...")
    print("-" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run([sys.executable, os.path.join(script_dir, "measurement_based_detector.py"), image_path],
                          capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Measurement detection failed:\n{result.stderr}")
        return False

    print("[OK] Measurements detected and classified")
    print()

    # Step 2: Run proximity_pairing_detector.py
    print("[2/3] Running proximity pairing to find actual openings...")
    print("-" * 60)
    result = subprocess.run([sys.executable, os.path.join(script_dir, "proximity_pairing_detector.py"), image_path],
                          capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Proximity pairing failed:\n{result.stderr}")
        return False

    print("[OK] Openings paired successfully")
    print()

    # Step 3: Run mark_opening_intersections.py
    print("[3/3] Creating final marked drawing...")
    print("-" * 60)
    result = subprocess.run([sys.executable, os.path.join(script_dir, "mark_opening_intersections.py"), image_path],
                          capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Marking intersections failed:\n{result.stderr}")
        return False

    print("[OK] Final marked drawing created")
    print()

    # List output files
    print("=" * 80)
    print("OUTPUT FILES CREATED:")
    print("-" * 60)

    # Look for files that match the pattern
    import glob
    pattern_files = [
        f"{base_name}_measurements_detected.png",
        f"{base_name}_measurements_data.json",
        f"{base_name}_*_paired.png",
        f"{base_name}_openings_data.json",
        f"{base_name}_*_marked.png"
    ]

    final_output = None
    marked_files = []
    all_created_files = []
    for pattern in pattern_files:
        matches = glob.glob(os.path.join(image_dir, pattern))
        for file_path in matches:
            file_name = os.path.basename(file_path)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"  [OK] {file_name} ({size:,} bytes)")
                all_created_files.append(file_name)
                if "_marked.png" in file_name:
                    marked_files.append(file_name)

    # The last marked file created is the final output
    if marked_files:
        # Sort to get the last one created (highest number)
        marked_files.sort()
        final_output = marked_files[-1]

    print("\n" + "=" * 80)
    if final_output:
        print(f"FINAL OUTPUT: {final_output}")
    else:
        print("FINAL OUTPUT: No marked file found")
    print("=" * 80)

    # Clean up temporary files - keep only the original and final marked image
    print("\n" + "=" * 80)
    print("CLEANING UP TEMPORARY FILES")
    print("-" * 60)
    print(f"Keeping: {os.path.basename(image_path)} (original)")
    if final_output:
        print(f"Keeping: {final_output} (final marked output)")
    print("-" * 60)

    deleted_count = 0
    for file_name in all_created_files:
        # Keep only the final marked output, delete everything else
        if file_name != final_output:
            file_path = os.path.join(image_dir, file_name)
            try:
                os.remove(file_path)
                print(f"  [DELETED] {file_name}")
                deleted_count += 1
            except Exception as e:
                print(f"  [ERROR] Could not delete {file_name}: {e}")

    if deleted_count > 0:
        print(f"\n[OK] Cleaned up {deleted_count} temporary files")
        if final_output:
            print(f"[OK] Kept original image and final output: {final_output}")
    else:
        print("[OK] No temporary files to clean up")

    print("=" * 80)

    return True

def main():
    # Default to page_16.png if no argument provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for any .png file in current directory
        png_files = [f for f in os.listdir('.') if f.endswith('.png') and not f.endswith('_annotated.png')
                     and not f.endswith('_marked.png') and not f.endswith('_paired.png')]

        if png_files:
            image_path = png_files[0]
            print(f"No image specified, using: {image_path}")
        else:
            print("Usage: python process_cabinet_page.py <image_file.png>")
            print("   or: python process_cabinet_page.py")
            print("        (will use first .png file found in current directory)")
            return

    # Process the page
    success = process_page(image_path)

    if success:
        print("\n[SUCCESS] Page processed successfully!")
    else:
        print("\n[FAILED] Page processing failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()