"""
Open HTML file in browser for manual PDF printing
This matches the original PDF creation method
"""

import sys
import os
import webbrowser
from pathlib import Path

if len(sys.argv) > 1:
    html_file = sys.argv[1]
else:
    html_file = "../need to process/door_103_order_form.html"

# Get absolute path
abs_path = os.path.abspath(html_file)
file_url = Path(abs_path).as_uri()

print("Opening HTML in browser for PDF printing...")
print(f"File: {file_url}")
print("\nTo create PDF:")
print("1. Press Ctrl+P (or Cmd+P on Mac)")
print("2. Select 'Save as PDF' or 'Microsoft Print to PDF'")
print("3. Ensure margins are set to 'Default' or 'None'")
print("4. Save the PDF file")

webbrowser.open(file_url)