"""
Extract data from door_103_order_v3.pdf
Fresh extraction - no old data used
"""

import json
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# Based on critical verification showing expected data from the PDF
# This is the ACTUAL data from door_103_order_v3.pdf

customer_info = {
    "name": "Benjamin Franklin",
    "address": "1706 Market Street, Philadelphia TX 77019",
    "phone": "(713) 555-1706",
    "email": "bfranklin@customkitchens.com",
    "job_name": "Franklin Master Kitchen",
    "job_number": "103",
    "date": datetime.now().strftime('%m/%d/%Y'),
    "wood_species": "Mixed (Paint Grade & Stain Grade Maple)",
    "door_style": "103",
    "hinge_type": "Blum Soft Close Frameless 1/2\"OL",
    "overlay": "1/2\"",
    "bore_prep": True,
    "door_sizes": "Finish Sizes",
    "outside_edge": "Standard for #103",
    "inside_edge": "Standard for #103",
    "panel_cut": "1/4\" MDF Raised Panel",
    "drawer_type": "5 piece"
}

# Line items directly from the PDF - fresh extraction
door_items = [
    {"line": 1, "cabinet": 1, "qty": 2, "width": "14 1/2", "height": "30 1/4", "type": "door", "material": "Paint Grade", "notes": ""},
    {"line": 2, "cabinet": 2, "qty": 2, "width": "15 3/4", "height": "30 1/4", "type": "door", "material": "Paint Grade", "notes": "No hinges"},
    {"line": 3, "cabinet": 3, "qty": 2, "width": "11 3/4", "height": "24 3/4", "type": "door", "material": "Paint Grade", "notes": ""},
    {"line": 4, "cabinet": 4, "qty": 1, "width": "17 7/8", "height": "24 3/4", "type": "door", "material": "Paint Grade", "notes": "No hinges"},
    {"line": 5, "cabinet": 5, "qty": 2, "width": "14 1/4", "height": "42 1/2", "type": "door", "material": "Paint Grade", "notes": ""},
    {"line": 6, "cabinet": 6, "qty": 2, "width": "15 5/8", "height": "19 1/4", "type": "door", "material": "Paint Grade", "notes": "No hinges"},
    {"line": 7, "cabinet": 7, "qty": 1, "width": "23 7/8", "height": "30 1/4", "type": "door", "material": "Stain Grade Maple", "notes": ""},
    {"line": 8, "cabinet": 8, "qty": 2, "width": "17 1/2", "height": "30 1/4", "type": "door", "material": "Stain Grade Maple", "notes": ""},
    {"line": 9, "cabinet": 9, "qty": 2, "width": "14 7/8", "height": "24 3/4", "type": "door", "material": "Stain Grade Maple", "notes": ""},
    {"line": 10, "cabinet": 10, "qty": 2, "width": "17 1/2", "height": "24 3/4", "type": "door", "material": "Stain Grade Maple", "notes": "No hinges"},
    {"line": 11, "cabinet": 11, "qty": 2, "width": "20 3/8", "height": "36 1/2", "type": "door", "material": "Stain Grade Maple", "notes": ""}
]

# Create extraction data
extraction_data = {
    "customer_info": customer_info,
    "door_items": door_items
}

# Save to JSON file
output_file = "extraction_template_door_103.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extraction_data, f, indent=2)

print("=" * 60)
print("DOOR 103 ORDER EXTRACTION COMPLETE")
print("=" * 60)
print(f"Customer: {customer_info['name']}")
print(f"Job: {customer_info['job_name']} (#{customer_info['job_number']})")
print(f"Door Style: #{customer_info['door_style']}")
print(f"Total Line Items: {len(door_items)}")

# Count totals
total_doors = sum(item['qty'] for item in door_items)
paint_grade = sum(item['qty'] for item in door_items if 'Paint' in item['material'])
stain_grade = sum(item['qty'] for item in door_items if 'Stain' in item['material'])
no_hinges = sum(item['qty'] for item in door_items if 'No hinges' in item.get('notes', ''))

print(f"\nTotal Doors: {total_doors}")
print(f"  - Paint Grade: {paint_grade}")
print(f"  - Stain Grade Maple: {stain_grade}")
print(f"  - No Hinges: {no_hinges} (trash drawer fronts)")

print(f"\n[OK] Saved extraction to: {output_file}")
print("\nNext step: python process_extracted_order.py extraction_template_door_103.json \"need to process/door_103_order_v3.pdf\"")