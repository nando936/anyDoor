#!/usr/bin/env python3
"""Test Docling's basic capabilities for measurements"""

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

# Initialize converter
converter = DocumentConverter(
    allowed_formats=[InputFormat.IMAGE]
)

# Path to image
image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-09-09(14-18)/all_pages/page_8.png"

print("Converting document...")
result = converter.convert(image_path)

doc = result.document
print(f"\n=== DOCLING FOUND {len(doc.texts)} TEXT ITEMS ===\n")

# Show all text items with coordinates
for i, text in enumerate(doc.texts):
    bbox = text.prov[0].bbox if text.prov else None
    if bbox:
        print(f"{i+1:2d}. '{text.text}' at ({int(bbox.l):4d}, {int(bbox.t):4d})")
    else:
        print(f"{i+1:2d}. '{text.text}'")

print("\n=== MEASUREMENT ANALYSIS ===")
print("Docling provides:")
print("- Raw OCR text extraction")
print("- Bounding box coordinates for each text item")
print("- Document structure (tables, sections, etc.)")
print("\nDocling does NOT provide:")
print("- Automatic measurement parsing (e.g., converting '18 1/16' to decimal)")
print("- Measurement unit extraction")
print("- Measurement classification (width/height/etc.)")
print("\nFor structured extraction, you would need to:")
print("1. Use the experimental extract() API with a schema (requires VLM model)")
print("2. Post-process the raw text yourself with Python")
