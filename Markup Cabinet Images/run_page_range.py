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
import tkinter as tk
from tkinter import scrolledtext
import threading
import time

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

class ProgressDialog:
    def __init__(self, pages):
        self.root = tk.Tk()
        self.root.title(f"Processing {len(pages)} Pages")
        self.root.geometry("600x450")

        # Timer label at top
        self.timer_label = tk.Label(self.root, text="Elapsed Time: 00:00:00", font=("Arial", 11, "bold"))
        self.timer_label.pack(pady=5)

        # Text widget for output
        self.text = scrolledtext.ScrolledText(self.root, width=70, height=20, font=("Courier", 10))
        self.text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Status label
        self.status_label = tk.Label(self.root, text="Starting...", font=("Arial", 10, "bold"))
        self.status_label.pack(pady=5)

        # OK button (hidden initially)
        self.ok_button = tk.Button(self.root, text="OK", command=self.on_close, font=("Arial", 10), width=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.closed = False
        self.timer_running = True
        self.start_time = time.time()
        self.update_timer()

    def append_text(self, text):
        if not self.closed:
            self.text.insert(tk.END, text + "\n")
            self.text.see(tk.END)
            self.root.update()

    def update_status(self, text):
        if not self.closed:
            self.status_label.config(text=text)
            self.root.update()

    def update_timer(self):
        if not self.closed and self.timer_running:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.timer_label.config(text=f"Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)  # Update every second

    def stop_timer(self):
        self.timer_running = False
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        self.timer_label.config(text=f"Total Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        # Show OK button when complete
        self.ok_button.pack(pady=10)

    def on_close(self):
        self.closed = True
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()

def process_pages(dialog, pages, base_path):
    current_opening_num = 1

    for page_num in pages:
        if dialog.closed:
            break
        page_file = f"{base_path}/page_{page_num}.png"

        # Update dialog status
        dialog.update_status(f"Processing page {page_num}...")
        dialog.append_text(f"Processing page {page_num}...")

        # Run measurement_detector_test.py with start number
        process = subprocess.Popen(
            [sys.executable, "measurement_detector_test.py", page_file, "--start-num", str(current_opening_num)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Collect output while streaming
        lines = []
        for line in process.stdout:
            lines.append(line.rstrip())

        process.wait()

        # Count openings from this page
        openings_count = 0
        for line in lines:
            if 'Drawing' in line and 'paired openings' in line:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    openings_count = int(parts[1])
                    break

        # Display page summary
        if openings_count > 0:
            end_opening_num = current_opening_num + openings_count - 1
            msg = f"  [OK] Page {page_num}: Openings #{current_opening_num} - #{end_opening_num} found ({openings_count} total)"
        else:
            msg = f"  [OK] Page {page_num}: No openings found"

        dialog.append_text(msg)
        print(msg)

        # Check for issues/warnings
        issues = []
        for line in lines:
            if 'WARNING' in line.upper() or 'ERROR' in line.upper():
                issues.append(line.strip())
            elif 'No openings could be paired' in line:
                issues.append(line.strip())
            elif 'Cannot pair' in line:
                issues.append(line.strip())

        if issues:
            dialog.append_text("    Issues found:")
            print("    Issues found:")
            for issue in issues:
                dialog.append_text(f"      - {issue}")
                print(f"      - {issue}")
        else:
            dialog.append_text("    No issues found")
            print("    No issues found")

        # Update opening counter
        current_opening_num += openings_count
        dialog.append_text("")
        print()

    dialog.stop_timer()
    dialog.update_status("Complete!")
    dialog.append_text("=" * 50)
    dialog.append_text("Processing complete!")

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

    # Create progress dialog
    dialog = ProgressDialog(pages)
    dialog.append_text(f"Processing {len(pages)} pages: {pages}")
    dialog.append_text("")

    # Start processing in a separate thread
    thread = threading.Thread(target=process_pages, args=(dialog, pages, base_path))
    thread.daemon = True
    thread.start()

    # Run dialog
    dialog.mainloop()

if __name__ == "__main__":
    main()
