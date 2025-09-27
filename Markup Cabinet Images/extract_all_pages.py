"""
Extract all pages from a PDF
"""
import fitz  # PyMuPDF
import os
import sys

if len(sys.argv) < 2:
    print("[ERROR] Please specify PDF path")
    print("Usage: python extract_all_pages.py <pdf_path>")
    sys.exit(1)

pdf_path = sys.argv[1]
work_folder = os.getcwd()

# Open PDF
doc = fitz.open(pdf_path)
total_pages = len(doc)

print(f"[INFO] Extracting {total_pages} pages from PDF...")
print("=" * 60)

# Extract each page
for page_num in range(1, total_pages + 1):
    page_index = page_num - 1
    page = doc[page_index]

    # Increase resolution for better quality
    mat = fitz.Matrix(3, 3)  # 3x scale for higher resolution
    pix = page.get_pixmap(matrix=mat)

    # Save as PNG with page number in filename
    output_file = os.path.join(work_folder, f"page_{page_num}.png")
    pix.save(output_file)

    print(f"[OK] Extracted page {page_num} -> page_{page_num}.png")

doc.close()
print("=" * 60)
print(f"[SUCCESS] All {total_pages} pages extracted successfully!")