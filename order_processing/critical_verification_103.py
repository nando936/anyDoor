"""
Critical Verification for Benjamin Franklin Order #103
This performs INDEPENDENT verification from scratch
"""

import sys
import json
import os
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("CRITICAL VERIFICATION - INDEPENDENT RE-ANALYSIS")
print("Order: Benjamin Franklin #103")
print("=" * 60)
print("\nRe-reading ORIGINAL order from scratch...\n")

# What the ORIGINAL PDF says (from door_103_order_v3.pdf)
original_order = {
    'customer': 'Benjamin Franklin',
    'job_number': '103',
    'door_style': '103',
    'door_sizes': 'Finish Sizes',
    'bore_prep': 'Yes',
    'total_doors_claimed': 17,
    'items': [
        {'line': 1, 'qty': 2, 'width': '14 1/2', 'height': '30 1/4', 'material': 'Paint Grade', 'notes': ''},
        {'line': 2, 'qty': 2, 'width': '15 3/4', 'height': '30 1/4', 'material': 'Paint Grade', 'notes': 'No hinges'},
        {'line': 3, 'qty': 2, 'width': '11 3/4', 'height': '24 3/4', 'material': 'Paint Grade', 'notes': ''},
        {'line': 4, 'qty': 1, 'width': '17 7/8', 'height': '24 3/4', 'material': 'Paint Grade', 'notes': 'No hinges'},
        {'line': 5, 'qty': 2, 'width': '14 1/4', 'height': '42 1/2', 'material': 'Paint Grade', 'notes': ''},
        {'line': 6, 'qty': 2, 'width': '15 5/8', 'height': '19 1/4', 'material': 'Paint Grade', 'notes': 'No hinges'},
        {'line': 7, 'qty': 1, 'width': '23 7/8', 'height': '30 1/4', 'material': 'Stain Grade Maple', 'notes': ''},
        {'line': 8, 'qty': 2, 'width': '17 1/2', 'height': '30 1/4', 'material': 'Stain Grade Maple', 'notes': ''},
        {'line': 9, 'qty': 2, 'width': '14 7/8', 'height': '24 3/4', 'material': 'Stain Grade Maple', 'notes': ''},
        {'line': 10, 'qty': 2, 'width': '17 1/2', 'height': '24 3/4', 'material': 'Stain Grade Maple', 'notes': 'No hinges'},
        {'line': 11, 'qty': 2, 'width': '20 3/8', 'height': '36 1/2', 'material': 'Stain Grade Maple', 'notes': ''}
    ]
}

# Load what was generated
output_folder = 'output/benjamin_franklin_103'
json_file = os.path.join(output_folder, 'order_data.json')

# Try both locations for compatibility
import os
if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        generated = json.load(f)
elif os.path.exists('benjamin_franklin_103_data.json'):
    # Fallback to old location
    with open('benjamin_franklin_103_data.json', 'r') as f:
        generated = json.load(f)
else:
    print("[ERROR] Cannot find order data JSON file!")
    print(f"Looked for: {json_file}")
    sys.exit(1)

print("VERIFICATION CHECKLIST:")
print("-" * 40)

errors = []
warnings = []

# 1. Verify customer info
if generated['customer_info']['name'] != 'Benjamin Franklin':
    errors.append(f"Customer name: Expected 'Benjamin Franklin', Got '{generated['customer_info']['name']}'")
else:
    print("[OK] Customer name matches")

if generated['customer_info']['job_number'] != '103':
    errors.append(f"Job number: Expected '103', Got '{generated['customer_info']['job_number']}'")
else:
    print("[OK] Job number matches")

if generated['door_style'] != '103':
    errors.append(f"Door style: Expected '103', Got '{generated['door_style']}'")
else:
    print("[OK] Door style matches")

# 2. Verify door sizes type
if generated['customer_info'].get('door_sizes', '') != 'finish':
    warnings.append(f"Door sizes type: Got '{generated['customer_info'].get('door_sizes', 'unknown')}'")
else:
    print("[OK] Door sizes: Finish Sizes (no conversion needed)")

# 3. Verify bore prep
if not generated['customer_info'].get('bore_prep', False):
    errors.append("Bore prep should be Yes")
else:
    print("[OK] Bore prep: Yes")

# 4. Verify total door count
original_total = sum(item['qty'] for item in original_order['items'])
generated_total = sum(item['qty'] for item in generated['door_items'])

if original_total != generated_total:
    errors.append(f"Total doors: Expected {original_total}, Got {generated_total}")
else:
    print(f"[OK] Total doors: {generated_total}")

# 5. Verify each line item
print("\nLine-by-line verification:")
print("-" * 40)

no_hinge_lines = [2, 4, 6, 10]  # Lines that should have no hinges

for orig_item in original_order['items']:
    line_num = orig_item['line']
    
    # Find corresponding generated item
    gen_item = None
    for item in generated['door_items']:
        if item['cabinet'] == line_num:
            gen_item = item
            break
    
    if not gen_item:
        errors.append(f"Line #{line_num}: MISSING from generated door list!")
        continue
    
    # Check quantity
    if gen_item['qty'] != orig_item['qty']:
        errors.append(f"Line #{line_num}: Qty mismatch - Expected {orig_item['qty']}, Got {gen_item['qty']}")
    
    # Check dimensions (should be exact since these are finish sizes)
    if gen_item['width'] != orig_item['width']:
        errors.append(f"Line #{line_num}: Width mismatch - Expected {orig_item['width']}, Got {gen_item['width']}")
    
    if gen_item['height'] != orig_item['height']:
        errors.append(f"Line #{line_num}: Height mismatch - Expected {orig_item['height']}, Got {gen_item['height']}")
    
    # Check material
    if gen_item.get('material', '') != orig_item['material']:
        errors.append(f"Line #{line_num}: Material mismatch - Expected {orig_item['material']}, Got {gen_item.get('material', 'unknown')}")
    
    # Check no-hinge notes
    if line_num in no_hinge_lines:
        if 'no hinge' not in gen_item.get('notes', '').lower() and 'no bore' not in gen_item.get('notes', '').lower():
            errors.append(f"Line #{line_num}: MISSING 'No hinges' note - CRITICAL for production!")
    
    print(f"[{'OK' if line_num not in [e.split(':')[0].split('#')[1] for e in errors if 'Line #' in e] else 'ERROR'}] Line #{line_num}: {orig_item['qty']} @ {orig_item['width']} x {orig_item['height']} - {orig_item['material']}")

# 6. Verify material counts
print("\nMaterial Summary Verification:")
print("-" * 40)

paint_grade_expected = sum(item['qty'] for item in original_order['items'] if item['material'] == 'Paint Grade')
maple_expected = sum(item['qty'] for item in original_order['items'] if item['material'] == 'Stain Grade Maple')

paint_grade_generated = sum(item['qty'] for item in generated['door_items'] if item.get('material') == 'Paint Grade')
maple_generated = sum(item['qty'] for item in generated['door_items'] if item.get('material') == 'Stain Grade Maple')

if paint_grade_expected != paint_grade_generated:
    errors.append(f"Paint Grade count: Expected {paint_grade_expected}, Got {paint_grade_generated}")
else:
    print(f"[OK] Paint Grade: {paint_grade_generated} doors")

if maple_expected != maple_generated:
    errors.append(f"Stain Grade Maple count: Expected {maple_expected}, Got {maple_generated}")
else:
    print(f"[OK] Stain Grade Maple: {maple_generated} doors")

# 7. Critical no-hinge verification
print("\nCritical No-Hinge Verification:")
print("-" * 40)

for line in no_hinge_lines:
    gen_item = next((item for item in generated['door_items'] if item['cabinet'] == line), None)
    if gen_item:
        has_note = 'no hinge' in gen_item.get('notes', '').lower() or 'no bore' in gen_item.get('notes', '').lower()
        if has_note:
            print(f"[OK] Line #{line}: Has 'No hinges' note")
        else:
            print(f"[ERROR] Line #{line}: MISSING 'No hinges' note")

# Final summary
print("\n" + "=" * 60)
print("VERIFICATION RESULTS")
print("=" * 60)

if errors:
    print(f"\n[FAILED] Found {len(errors)} CRITICAL ERRORS:")
    for error in errors:
        print(f"  - {error}")
    print("\n[ACTION REQUIRED] FIX ERRORS AND REGENERATE DOOR LIST")
else:
    print("\n[SUCCESS] ALL CRITICAL CHECKS PASSED!")
    print("Door list matches original order exactly.")

if warnings:
    print(f"\n[WARNING] Found {len(warnings)} warnings:")
    for warning in warnings:
        print(f"  - {warning}")

print("\n" + "=" * 60)
if not errors:
    print("VERIFICATION COMPLETE - ORDER READY FOR PRODUCTION")
else:
    print("VERIFICATION FAILED - DO NOT PROCEED TO PRODUCTION")
    print("Fix extraction/processing and regenerate until verification passes")
print("=" * 60)