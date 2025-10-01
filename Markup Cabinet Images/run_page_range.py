#!/usr/bin/env python3
"""
Run measurement_detector_test.py on a range of pages
Usage: python run_page_range.py 1-10
       python run_page_range.py 1,3,5
       python run_page_range.py 1-5,10,15-20
"""

import sys
import subprocess
import os

def parse_range(range_str):
    """Parse range string like '1-10' or '1,3,5' or '1-5,10,15-20' into list of page numbers"""
    pages = []
    parts = range_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range like '1-10'
            start, end = part.split('-')
            pages.extend(range(int(start), int(end) + 1))
        else:
            # Single page like '5'
            pages.append(int(part))

    return sorted(set(pages))  # Remove duplicates and sort

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_page_range.py <range>")
        print("Examples:")
        print("  python run_page_range.py 1-10")
        print("  python run_page_range.py 1,3,5")
        print("  python run_page_range.py 1-5,10,15-20")
        sys.exit(1)

    range_str = sys.argv[1]
    pages = parse_range(range_str)

    base_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages"

    print(f"Processing {len(pages)} pages: {pages}")
    print()

    current_opening_num = 1

    for page_num in pages:
        page_file = f"{base_path}/page_{page_num}.png"

        print(f"{'='*80}")
        print(f"Processing page {page_num} (opening numbers start at {current_opening_num})")
        print(f"{'='*80}")

        # Run measurement_detector_test.py with start number
        result = subprocess.run(
            [sys.executable, "measurement_detector_test.py", page_file, "--start-num", str(current_opening_num)],
            capture_output=True,
            text=True
        )

        # Show summary from output (last few lines)
        lines = result.stdout.strip().split('\n')
        summary_lines = [line for line in lines[-10:] if 'SAVED' in line or 'Drawing' in line or 'TOTAL' in line or 'openings' in line.lower()]
        for line in summary_lines:
            print(line)

        # Count openings from this page to update current_opening_num
        for line in lines:
            if 'Drawing' in line and 'paired openings' in line:
                # Extract number like "Drawing 9 paired openings"
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    openings_count = int(parts[1])
                    current_opening_num += openings_count
                    break

        print()

if __name__ == "__main__":
    main()
