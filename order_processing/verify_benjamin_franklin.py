"""
Critical Verification for Benjamin Franklin Order
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("CRITICAL VERIFICATION - Benjamin Franklin Order")
print("=" * 60)
print("\nVerifying door quantities and specifications...\n")

# Expected data from original order
expected_doors = [
    {'cabinet': 1, 'qty': 2, 'width': '14 1/2', 'height': '30 1/4', 'material': 'Paint Grade'},
    {'cabinet': 2, 'qty': 2, 'width': '15 3/4', 'height': '30 1/4', 'material': 'Paint Grade'},
    {'cabinet': 3, 'qty': 1, 'width': '23 7/8', 'height': '30 1/4', 'material': 'Stain Grade Maple'},
    {'cabinet': 4, 'qty': 2, 'width': '17 1/2', 'height': '30 1/4', 'material': 'Stain Grade Maple'},
    {'cabinet': 5, 'qty': 2, 'width': '11 3/4', 'height': '24 3/4', 'material': 'Paint Grade'},
    {'cabinet': 6, 'qty': 1, 'width': '17 7/8', 'height': '24 3/4', 'material': 'Paint Grade'},
    {'cabinet': 7, 'qty': 2, 'width': '14 7/8', 'height': '24 3/4', 'material': 'Stain Grade Maple'},
    {'cabinet': 8, 'qty': 2, 'width': '17 1/2', 'height': '24 3/4', 'material': 'Stain Grade Maple'},
    {'cabinet': 9, 'qty': 2, 'width': '20 3/8', 'height': '36 1/2', 'material': 'Stain Grade Maple'},
    {'cabinet': 10, 'qty': 2, 'width': '14 1/4', 'height': '42 1/2', 'material': 'Paint Grade'},
    {'cabinet': 11, 'qty': 1, 'width': '35 1/2', 'height': '18 3/4', 'material': 'Stain Grade Maple'},
    {'cabinet': 12, 'qty': 2, 'width': '15 5/8', 'height': '19 1/4', 'material': 'Paint Grade'}
]

# Load and parse the generated door list
import json
with open('benjamin_franklin_103_data.json', 'r') as f:
    data = json.load(f)

print("VERIFICATION CHECKLIST:")
print("-" * 40)

errors = []
warnings = []

# Check customer info
if data['customer_info']['name'] != 'Benjamin Franklin':
    errors.append(f"Customer name mismatch: {data['customer_info']['name']}")
else:
    print("[OK] Customer name: Benjamin Franklin")

if data['customer_info']['job_number'] != '103':
    errors.append(f"Job number mismatch: {data['customer_info']['job_number']}")
else:
    print("[OK] Job number: 103")

# Check door style
if data['door_style'] != '103':
    errors.append(f"Door style mismatch: {data['door_style']}")
else:
    print("[OK] Door style: 103")

# Check door counts
total_expected = sum(d['qty'] for d in expected_doors)
total_generated = sum(item['qty'] for item in data['door_items'])

if total_expected != total_generated:
    errors.append(f"Total door count mismatch: Expected {total_expected}, Got {total_generated}")
else:
    print(f"[OK] Total doors: {total_generated}")

# Check each door item
print("\nDoor-by-door verification:")
print("-" * 40)

for i, expected in enumerate(expected_doors, 1):
    if i <= len(data['door_items']):
        actual = data['door_items'][i-1]
        
        # Check quantity
        if actual['qty'] != expected['qty']:
            errors.append(f"Cabinet #{expected['cabinet']}: Qty mismatch - Expected {expected['qty']}, Got {actual['qty']}")
        
        # Check dimensions
        if actual['width'] != expected['width']:
            errors.append(f"Cabinet #{expected['cabinet']}: Width mismatch - Expected {expected['width']}, Got {actual['width']}")
        
        if actual['height'] != expected['height']:
            errors.append(f"Cabinet #{expected['cabinet']}: Height mismatch - Expected {expected['height']}, Got {actual['height']}")
        
        # Check material
        actual_material = actual.get('material', 'Unknown')
        if actual_material != expected['material']:
            errors.append(f"Cabinet #{expected['cabinet']}: Material mismatch - Expected {expected['material']}, Got {actual_material}")
        
        print(f"[OK] Cabinet #{expected['cabinet']}: {expected['qty']} door(s) @ {expected['width']} x {expected['height']} - {expected['material']}")
    else:
        errors.append(f"Missing door item for Cabinet #{expected['cabinet']}")

# Check materials summary
print("\nMaterial Summary:")
print("-" * 40)

paint_grade_count = sum(d['qty'] for d in expected_doors if d['material'] == 'Paint Grade')
maple_count = sum(d['qty'] for d in expected_doors if d['material'] == 'Stain Grade Maple')

print(f"[OK] Paint Grade: {paint_grade_count} doors")
print(f"[OK] Stain Grade Maple: {maple_count} doors")

# Check specifications
print("\nSpecifications Check:")
print("-" * 40)

if data['customer_info']['hinge_type'] == 'Blum Soft Close Frameless 1/2"OL':
    print("[OK] Hinge type: Blum Soft Close Frameless 1/2\"OL")
else:
    errors.append(f"Hinge type mismatch: {data['customer_info']['hinge_type']}")

if data['customer_info']['door_sizes'] == 'Finish Sizes':
    print("[OK] Door sizes: Finish Sizes")
else:
    warnings.append(f"Door sizes type: {data['customer_info']['door_sizes']}")

if data['customer_info'].get('bore_prep', False):
    print("[OK] Bore prep: Yes")
else:
    warnings.append("Bore prep: No")

# Final summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

if errors:
    print(f"\n[ERROR] Found {len(errors)} critical errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("\n[SUCCESS] All critical checks passed!")

if warnings:
    print(f"\n[WARNING] Found {len(warnings)} warnings:")
    for warning in warnings:
        print(f"  - {warning}")

print("\n" + "=" * 60)
if not errors:
    print("ORDER IS READY FOR PRODUCTION")
else:
    print("ORDER NEEDS CORRECTION BEFORE PRODUCTION")
print("=" * 60)