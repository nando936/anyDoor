"""
Check total number of pages in a PDF
"""
import fitz  # PyMuPDF
import sys

if len(sys.argv) < 2:
    print("[ERROR] Please specify PDF path")
    print("Usage: python check_pdf_pages.py <pdf_path>")
    sys.exit(1)

pdf_path = sys.argv[1]

# Open PDF
doc = fitz.open(pdf_path)
total_pages = len(doc)
doc.close()

print(f"[INFO] PDF has {total_pages} pages")