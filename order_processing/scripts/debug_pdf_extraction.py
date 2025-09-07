import pdfplumber
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = "1-302_door_order.pdf"

print("Debugging PDF extraction to see actual text format...")
print("=" * 60)

try:
    with pdfplumber.open(pdf_path) as pdf:
        # Look at first page
        for page_num, page in enumerate(pdf.pages[:2]):  # First 2 pages
            print(f"\n--- Page {page_num + 1} ---")
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                for i, line in enumerate(lines[:50]):  # First 50 lines
                    # Show lines with dimensions or type indicators
                    if any(x in line for x in [' x ', ' X ', 'DF', ' D ', 'FF', '16 5/8', '24 3/4']):
                        print(f"Line {i:3}: [{line}]")
                        
except Exception as e:
    print(f"[ERROR] {e}")