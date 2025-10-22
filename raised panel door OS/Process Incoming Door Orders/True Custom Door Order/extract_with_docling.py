"""
Extract TruCustom door order form using Docling
"""
import os
import sys
import json
from docling.document_converter import DocumentConverter

if len(sys.argv) < 2:
    print("[ERROR] Please specify image path")
    print("Usage: python extract_with_docling.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Convert Windows path to forward slashes
if image_path.startswith('\\\\'):
    image_path = image_path.replace('\\', '/')

print("=" * 60)
print(f"Processing: {os.path.basename(image_path)}")
print("=" * 60)

# Initialize converter (default settings, will use AI models)
converter = DocumentConverter()

# Convert the image
print("\n=== Converting with Docling ===")
result = converter.convert(image_path)

# Export to markdown to see structure
print("\n=== Markdown Output ===")
markdown = result.document.export_to_markdown()
print(markdown)

# Check for tables
if hasattr(result.document, 'tables'):
    print(f"\n=== Found {len(result.document.tables)} tables ===")
    for i, table in enumerate(result.document.tables):
        print(f"\nTable {i+1}:")
        # Try to export table to CSV format
        if hasattr(table, 'export_to_dataframe'):
            df = table.export_to_dataframe()
            print(df)

# Save full output
output_dir = os.path.dirname(image_path)
base_name = os.path.splitext(os.path.basename(image_path))[0]

# Save markdown
md_path = os.path.join(output_dir, f"{base_name}_docling.md")
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(markdown)
print(f"\n[OK] Saved markdown to: {md_path}")

# Save JSON
json_path = os.path.join(output_dir, f"{base_name}_docling.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(result.document.export_to_dict(), f, indent=2)
print(f"[OK] Saved JSON to: {json_path}")

print("=" * 60)
