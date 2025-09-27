"""
Universal page extraction script - extract any page number from any PDF
Usage: python extract_any_page.py <pdf_path> <page_number>
"""
import fitz  # PyMuPDF
import os
import sys

if len(sys.argv) < 3:
    print("[ERROR] Please specify PDF path and page number")
    print("Usage: python extract_any_page.py <pdf_path> <page_number>")
    print("Example: python extract_any_page.py 'path/to/document.pdf' 11")
    sys.exit(1)

# Get arguments from command line
pdf_path = sys.argv[1]
page_num = int(sys.argv[2])
page_index = page_num - 1  # PDF uses 0-based indexing

# Use current directory as work folder (agent will have already created and cd'd to the subfolder)
work_folder = os.getcwd()

# Open PDF
doc = fitz.open(pdf_path)

# Check if page number is valid
if page_index >= len(doc) or page_index < 0:
    print(f"[ERROR] Page {page_num} does not exist in PDF")
    print(f"[INFO] PDF has {len(doc)} pages (1-{len(doc)})")
    doc.close()
    sys.exit(1)

# Extract specified page
page = doc[page_index]

# Increase resolution for better quality
mat = fitz.Matrix(3, 3)  # 3x scale for higher resolution
pix = page.get_pixmap(matrix=mat)

# Save as PNG with page number in filename
output_file = os.path.join(work_folder, f"page_{page_num}.png")

# Save the page
pix.save(output_file)
print(f"[OK] Extracted page {page_num} to: {output_file}")

# Get page dimensions
page_rect = page.rect
print(f"[INFO] Page dimensions: {page_rect.width:.0f} x {page_rect.height:.0f}")
print(f"[INFO] Extracted image dimensions: {pix.width} x {pix.height} (3x scale)")

doc.close()
print(f"[DONE] Page {page_num} extracted successfully")