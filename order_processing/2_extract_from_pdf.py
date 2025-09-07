"""
PDF Data Extraction Helper
Helps extract door order data from user-submitted PDFs
Since PDFs vary in format, this provides tools for manual extraction
"""

import json
from datetime import datetime

def create_extraction_template():
    """Create a template for manual data extraction"""
    
    template = {
        "customer_info": {
            "name": "",
            "address": "",
            "phone": "",
            "email": "",
            "job_name": "",
            "job_number": "",
            "date": datetime.now().strftime('%m/%d/%Y'),
            "wood_species": "",
            "door_style": "",
            "hinge_type": "",
            "overlay": "",
            "bore_prep": True,
            "panel_cut": "3/8\" Plywood (Flat Panel ONLY)",
            "outside_edge": "",
            "inside_edge": ""
        },
        "door_items": [
            {
                "line": 1,
                "cabinet": 1,
                "qty": 0,
                "width": "",
                "height": "",
                "type": "door",  # door, drawer, false_front
                "material": "",
                "notes": ""
            }
        ]
    }
    
    return template

def save_extraction_data(data, output_file):
    """Save extracted data to JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Saved extraction data to: {output_file}")

def load_extraction_data(input_file):
    """Load previously extracted data"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_text_checklist(pdf_name):
    """Create a text checklist for manual extraction"""
    
    checklist = f"""
MANUAL DATA EXTRACTION CHECKLIST
PDF: {pdf_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'=' * 60}

CUSTOMER INFORMATION:
[ ] Name: ________________________________
[ ] Address: _____________________________
[ ] Phone: _______________________________
[ ] Email: _______________________________

JOB INFORMATION:
[ ] Job Name: ____________________________
[ ] Job Number: __________________________
[ ] Date: ________________________________

DOOR SPECIFICATIONS:
[ ] Wood Species: ________________________
[ ] Door Style #: ________________________
[ ] Outside Edge: ________________________
[ ] Inside Edge/Sticky: __________________
[ ] Panel Cut: ___________________________
[ ] Bore/Door Prep: [ ] Yes  [ ] No
[ ] Hinge Type: __________________________
[ ] Overlay: _____________________________
[ ] Door Sizes: [ ] Opening  [ ] Finish

LINE ITEMS:
{'=' * 60}
Line # | Cab # | Qty | Width | Height | Type | Notes
-------|-------|-----|-------|--------|------|-------
1      |       |     |       |        |      |
2      |       |     |       |        |      |
3      |       |     |       |        |      |
4      |       |     |       |        |      |
5      |       |     |       |        |      |
6      |       |     |       |        |      |
7      |       |     |       |        |      |
8      |       |     |       |        |      |
9      |       |     |       |        |      |
10     |       |     |       |        |      |

SPECIAL NOTES:
{'=' * 60}
Line # | Special Instructions
-------|---------------------



IMPORTANT REMINDERS:
- Line numbers from PDF become Cabinet numbers in all reports
- Multiple items can have same cabinet number
- Special notes must stay with their line items
- Verify all dimensions are in fraction format (e.g., 14 3/8)
- Type can be: door, drawer, or false_front
"""
    
    return checklist

def validate_extraction(data):
    """Validate extracted data for completeness"""
    
    errors = []
    warnings = []
    
    # Check customer info
    customer = data.get('customer_info', {})
    required_fields = ['name', 'job_name', 'wood_species', 'door_style']
    
    for field in required_fields:
        if not customer.get(field):
            errors.append(f"Missing required field: {field}")
    
    # Check door items
    items = data.get('door_items', [])
    if not items:
        errors.append("No door items found")
    else:
        for idx, item in enumerate(items, 1):
            if not item.get('qty'):
                warnings.append(f"Line {idx}: Missing quantity")
            if not item.get('width'):
                warnings.append(f"Line {idx}: Missing width")
            if not item.get('height'):
                warnings.append(f"Line {idx}: Missing height")
            if not item.get('type'):
                warnings.append(f"Line {idx}: Missing type")
    
    return errors, warnings

def print_extraction_summary(data):
    """Print a summary of extracted data"""
    
    customer = data.get('customer_info', {})
    items = data.get('door_items', [])
    
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Customer: {customer.get('name', 'Unknown')}")
    print(f"Job: {customer.get('job_name', 'Unknown')} (#{customer.get('job_number', 'N/A')})")
    print(f"Door Style: #{customer.get('door_style', 'Unknown')}")
    print(f"Wood Species: {customer.get('wood_species', 'Unknown')}")
    print(f"\nTotal Line Items: {len(items)}")
    
    # Count by type
    door_count = sum(item['qty'] for item in items if item['type'] == 'door')
    drawer_count = sum(item['qty'] for item in items if item['type'] == 'drawer')
    
    print(f"Total Doors: {door_count}")
    print(f"Total Drawers: {drawer_count}")
    
    # Validate
    errors, warnings = validate_extraction(data)
    
    if errors:
        print(f"\n[ERROR] Found {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print(f"\n[WARNING] Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("\n[OK] Data extraction looks complete!")

# Example usage
if __name__ == "__main__":
    print("PDF Data Extraction Helper")
    print("=" * 60)
    
    # Create a template
    template = create_extraction_template()
    
    # Save template for manual editing
    save_extraction_data(template, "extraction_template.json")
    
    # Create text checklist
    checklist = create_text_checklist("user_submitted_order.pdf")
    with open("extraction_checklist.txt", 'w') as f:
        f.write(checklist)
    print("[OK] Created extraction_checklist.txt for manual data entry")
    
    print("\nInstructions:")
    print("1. Open the PDF and extraction_checklist.txt side by side")
    print("2. Fill in the checklist with data from the PDF")
    print("3. Transfer the data to extraction_template.json")
    print("4. Run this script to validate the extraction")
    print("5. Use 1_process_new_order.py to generate reports")