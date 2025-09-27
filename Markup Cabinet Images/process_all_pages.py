"""
Process all extracted pages with the cabinet detection pipeline
"""
import os
import sys
import subprocess
import glob

# Get working directory
work_folder = os.getcwd()

# Find all page PNG files
page_files = glob.glob(os.path.join(work_folder, "page_*.png"))
page_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # Sort by page number

if not page_files:
    print("[ERROR] No page files found in current directory")
    sys.exit(1)

print(f"[INFO] Found {len(page_files)} pages to process")
print("=" * 80)

# Process each page
successful = []
failed = []

for i, page_file in enumerate(page_files, 1):
    page_name = os.path.basename(page_file)
    page_num = page_name.split("_")[1].split(".")[0]

    print(f"\n[{i}/{len(page_files)}] Processing {page_name}...")
    print("-" * 60)

    # Run the process_cabinet_page.py script
    script_path = "C:/Users/nando/Projects/anyDoor/Markup Cabinet Images/process_cabinet_page.py"

    result = subprocess.run(
        [sys.executable, script_path, page_file],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    if result.returncode == 0:
        # Check if marked file was created
        marked_files = glob.glob(os.path.join(work_folder, f"page_{page_num}_*_marked.png"))
        if marked_files:
            successful.append(page_name)
            print(f"[OK] {page_name} processed successfully")
        else:
            failed.append(page_name)
            print(f"[WARNING] {page_name} processed but no openings detected")
    else:
        failed.append(page_name)
        print(f"[ERROR] {page_name} failed to process")
        if "No measurements found" in result.stdout:
            print("  -> No measurements detected in image")

print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)
print(f"Successfully processed: {len(successful)} pages")
print(f"Failed/No openings: {len(failed)} pages")

if successful:
    print("\nSuccessful pages:")
    for page in successful:
        print(f"  [OK] {page}")

if failed:
    print("\nFailed/No openings pages:")
    for page in failed:
        print(f"  [X] {page}")

# List all marked output files
marked_outputs = glob.glob(os.path.join(work_folder, "*_marked.png"))
if marked_outputs:
    print("\n" + "=" * 80)
    print("MARKED OUTPUT FILES:")
    print("-" * 80)
    for output in marked_outputs:
        print(f"  {os.path.basename(output)}")

print("\n[DONE] All processing complete!")