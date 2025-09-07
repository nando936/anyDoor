"""
CRITICAL REDUNDANT VERIFICATION
Independent re-analysis of original order vs generated door list
This script does NOT trust any previous processing - starts from scratch
"""

import sys
import os
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def perform_critical_verification(original_order_path, generated_door_list_path):
    """
    Completely independent verification - re-reads everything from scratch
    Does NOT use any data from the processing scripts
    """
    
    print("=" * 80)
    print("[CRITICAL VERIFICATION] Starting Independent Analysis")
    print("=" * 80)
    print("[!] This is a REDUNDANT check - analyzing everything from scratch")
    print("[!] Not trusting any previous processing")
    print()
    
    # Step 1: Fresh analysis of original order
    print("[STEP 1] Re-analyzing Original Order from Scratch...")
    print("-" * 60)
    
    with open(original_order_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Parse original order manually
    original_data = {
        'customer': None,
        'job': None,
        'door_style': None,
        'size_type': None,  # Opening or Finish
        'overlay': None,
        'items': []
    }
    
    # Extract customer info
    if 'Paul Revere' in original_content:
        original_data['customer'] = 'Paul Revere'
    
    if 'Revere Kitchen Remodel' in original_content:
        original_data['job'] = 'Revere Kitchen Remodel'
    
    # Find door style
    if '#231' in original_content or 'Door #:</td>' in original_content:
        original_data['door_style'] = '231'
    
    # Critical: Determine if Opening or Finish sizes
    if 'checkbox checked' in original_content:
        if 'Opening Sizes' in original_content and 'checkbox checked"></span> Opening' in original_content:
            original_data['size_type'] = 'Opening Sizes'
            print("   [!] Original order uses OPENING SIZES - must add 2 x overlay")
        elif 'Finish Sizes' in original_content and 'checkbox checked"></span> Finish' in original_content:
            original_data['size_type'] = 'Finish Sizes'
            print("   [!] Original order uses FINISH SIZES - no conversion needed")
    
    # Extract overlay
    if '1/2" Overlay' in original_content or '1/2"OL' in original_content:
        original_data['overlay'] = '1/2"'
        print(f"   Overlay detected: 1/2\" (will add 1\" total to opening sizes)")
    
    # Extract line items - parse the table rows
    print("\n   Extracting line items from original order:")
    
    # Look for table rows with door data
    import re
    
    # Pattern to find table rows with door data - updated to handle empty cells
    row_pattern = r'<tr>.*?<td>(\d+)</td>.*?<td>(.*?)</td>.*?<td>(.*?)</td>.*?<td>(.*?)</td>.*?<td>(.*?)</td>.*?<td[^>]*>(.*?)</td>.*?</tr>'
    
    rows = re.findall(row_pattern, original_content, re.DOTALL)
    
    for row in rows:
        line_num = row[0]
        qty = row[1].strip()
        width = row[2].strip()
        height = row[3].strip()
        item_type = row[4].strip()
        notes = row[5].strip()
        
        # Skip empty rows - must have quantity, width, and height
        if not qty or not width or not height:
            continue
            
        # Clean up notes - remove HTML
        notes = re.sub('<[^>]+>', '', notes)
        
        item = {
            'line': line_num,
            'qty': qty,
            'width': width,
            'height': height,
            'type': item_type,
            'notes': notes
        }
        original_data['items'].append(item)
        print(f"      Line #{line_num}: {qty} @ {width} x {height} - {notes[:30]}")
    
    print(f"\n   Total items found in original: {len(original_data['items'])}")
    
    # Step 2: Calculate expected finish sizes
    print("\n[STEP 2] Calculating Expected Finish Sizes...")
    print("-" * 60)
    
    for item in original_data['items']:
        if original_data['size_type'] == 'Opening Sizes':
            # Must add 2 x overlay to get finish size
            width_val = parse_dimension_fresh(item['width'])
            height_val = parse_dimension_fresh(item['height'])
            
            # Add 1" (2 x 1/2" overlay)
            finish_width = width_val + 1.0
            finish_height = height_val + 1.0
            
            item['expected_width'] = format_dimension_fresh(finish_width)
            item['expected_height'] = format_dimension_fresh(finish_height)
            
            print(f"   Line #{item['line']}: {item['width']} + 1\" = {item['expected_width']} (W), {item['height']} + 1\" = {item['expected_height']} (H)")
        else:
            # Finish sizes - no conversion
            item['expected_width'] = item['width']
            item['expected_height'] = item['height']
            print(f"   Line #{item['line']}: {item['width']} x {item['height']} (no conversion - finish sizes)")
    
    # Step 3: Fresh analysis of generated door list
    print("\n[STEP 3] Re-analyzing Generated Door List from Scratch...")
    print("-" * 60)
    
    with open(generated_door_list_path, 'r', encoding='utf-8') as f:
        generated_content = f.read()
    
    generated_items = []
    
    # Parse generated door list - look for input fields including material
    # Updated pattern to capture all fields including material
    rows_pattern = r'<td>#(\d+)</td>(.*?)</tr>'
    row_matches = re.findall(rows_pattern, generated_content, re.DOTALL)
    
    for row_match in row_matches:
        cabinet = row_match[0]
        row_content = row_match[1]
        
        # Extract all input values from this row
        input_values = re.findall(r'value="([^"]*)"', row_content)
        
        if len(input_values) >= 6:
            qty = input_values[0]
            width = input_values[1]
            height = input_values[2]
            item_type = input_values[3]
            material = input_values[4]
            style = input_values[5]
            notes = input_values[6] if len(input_values) > 6 else ''
            
            generated_items.append({
                'cabinet': cabinet,
                'qty': qty,
                'width': width,
                'height': height,
                'type': item_type,
                'material': material,
                'style': style,
                'notes': notes
            })
            print(f"   Cabinet #{cabinet}: {qty} @ {width} x {height} - {material} - {notes[:30] if notes else 'No notes'}")
    
    print(f"\n   Total items in door list: {len(generated_items)}")
    
    # Step 4: Line-by-line comparison
    print("\n[STEP 4] LINE-BY-LINE VERIFICATION...")
    print("=" * 80)
    
    errors = []
    warnings = []
    
    # Check count first
    if len(original_data['items']) != len(generated_items):
        errors.append(f"ITEM COUNT MISMATCH: Original has {len(original_data['items'])}, Door list has {len(generated_items)}")
    
    # Check each line
    for i, orig_item in enumerate(original_data['items']):
        print(f"\nLine #{orig_item['line']} Verification:")
        print("-" * 40)
        
        if i >= len(generated_items):
            errors.append(f"Line #{orig_item['line']}: MISSING from door list!")
            continue
        
        gen_item = generated_items[i]
        
        # Verify cabinet number matches line number
        if orig_item['line'] != gen_item['cabinet']:
            errors.append(f"Line #{orig_item['line']}: Cabinet number is {gen_item['cabinet']} (should be {orig_item['line']})")
            print(f"   [X] Cabinet number mismatch")
        else:
            print(f"   [OK] Cabinet number: #{gen_item['cabinet']}")
        
        # Verify quantity
        if orig_item['qty'] != gen_item['qty']:
            errors.append(f"Line #{orig_item['line']}: Quantity is {gen_item['qty']} (should be {orig_item['qty']})")
            print(f"   [X] Quantity: Expected {orig_item['qty']}, Got {gen_item['qty']}")
        else:
            print(f"   [OK] Quantity: {gen_item['qty']}")
        
        # Verify width
        if orig_item['expected_width'] != gen_item['width']:
            errors.append(f"Line #{orig_item['line']}: Width is {gen_item['width']} (should be {orig_item['expected_width']})")
            print(f"   [X] Width: Expected {orig_item['expected_width']}, Got {gen_item['width']}")
            print(f"       Original: {orig_item['width']} + 1\" = {orig_item['expected_width']}")
        else:
            print(f"   [OK] Width: {gen_item['width']} (converted correctly)")
        
        # Verify height
        if orig_item['expected_height'] != gen_item['height']:
            errors.append(f"Line #{orig_item['line']}: Height is {gen_item['height']} (should be {orig_item['expected_height']})")
            print(f"   [X] Height: Expected {orig_item['expected_height']}, Got {gen_item['height']}")
            print(f"       Original: {orig_item['height']} + 1\" = {orig_item['expected_height']}")
        else:
            print(f"   [OK] Height: {gen_item['height']} (converted correctly)")
        
        # Verify material/wood species
        orig_material = ''
        if 'Paint Grade' in orig_item['notes']:
            orig_material = 'Paint Grade'
        elif 'White Oak' in orig_item['notes']:
            orig_material = 'White Oak'
        
        if orig_material and 'material' in gen_item:
            if orig_material != gen_item.get('material', ''):
                errors.append(f"Line #{orig_item['line']}: Material is {gen_item.get('material')} (should be {orig_material})")
                print(f"   [X] Material: Expected {orig_material}, Got {gen_item.get('material')}")
            else:
                print(f"   [OK] Material: {gen_item.get('material')}")
        
        # Check for critical notes
        if 'no hinge' in orig_item['notes'].lower() or 'no bore' in orig_item['notes'].lower():
            if 'no' not in gen_item['notes'].lower():
                errors.append(f"Line #{orig_item['line']}: CRITICAL 'No bore/hinge' instruction NOT transferred!")
                print(f"   [X] CRITICAL ERROR: 'No bore/hinge' instruction NOT in notes")
                print(f"       Original: {orig_item['notes']}")
                print(f"       Generated: {gen_item['notes']}")
            else:
                print(f"   [OK] 'No bore' instruction preserved")
        
        # Verify cabinet number in notes
        if f"Cabinet #{orig_item['line']}" in orig_item['notes']:
            if f"Cabinet #{orig_item['line']}" not in gen_item['notes']:
                warnings.append(f"Line #{orig_item['line']}: Cabinet number not in notes")
                print(f"   [!] Warning: Cabinet number reference missing from notes")
        
        # Check if trash drawer note transferred
        if 'trash' in orig_item['notes'].lower():
            if 'trash' not in gen_item['notes'].lower():
                errors.append(f"Line #{orig_item['line']}: 'Trash drawer' note NOT transferred!")
                print(f"   [X] ERROR: 'Trash drawer' note missing")
    
    # Step 5: Final verdict
    print("\n" + "=" * 80)
    print("[VERIFICATION RESULTS]")
    print("=" * 80)
    
    if errors:
        print("\n[X] CRITICAL ERRORS FOUND - DO NOT PROCEED TO PRODUCTION")
        print("-" * 60)
        for error in errors:
            print(f"   ERROR: {error}")
        print("\n[!] FIX ALL ERRORS BEFORE PROCEEDING")
    else:
        print("\n[OK] ALL CRITICAL CHECKS PASSED")
        print("   - Item count matches")
        print("   - All quantities verified")
        print("   - All dimensions converted correctly")
        print("   - Cabinet numbers match line numbers")
    
    if warnings:
        print("\n[!] WARNINGS TO REVIEW:")
        for warning in warnings:
            print(f"   - {warning}")
    
    # Summary
    print("\n" + "=" * 80)
    print("[VERIFICATION SUMMARY]")
    print(f"   Original Order: {len(original_data['items'])} items")
    print(f"   Door List: {len(generated_items)} items")
    print(f"   Size Type: {original_data['size_type']}")
    print(f"   Overlay: {original_data['overlay']}")
    print(f"   Errors Found: {len(errors)}")
    print(f"   Warnings: {len(warnings)}")
    
    if not errors:
        print("\n[OK] VERIFICATION COMPLETE - SAFE TO PROCEED")
    else:
        print("\n[X] VERIFICATION FAILED - DO NOT PROCEED")
    
    return len(errors) == 0

def parse_dimension_fresh(dim_str):
    """Parse dimension string - independent implementation"""
    if not dim_str:
        return 0
    
    # Remove extra spaces
    dim_str = ' '.join(dim_str.split())
    
    parts = dim_str.split()
    total = 0.0
    
    for part in parts:
        if '/' in part:
            # It's a fraction
            try:
                num, den = part.split('/')
                total += float(num) / float(den)
            except:
                pass
        else:
            # It's a whole number
            try:
                total += float(part)
            except:
                pass
    
    return total

def format_dimension_fresh(value):
    """Format dimension - independent implementation"""
    whole = int(value)
    remainder = value - whole
    
    # Common fractions
    fractions = [
        (0.0625, "1/16"), (0.125, "1/8"), (0.1875, "3/16"), (0.25, "1/4"),
        (0.3125, "5/16"), (0.375, "3/8"), (0.4375, "7/16"), (0.5, "1/2"),
        (0.5625, "9/16"), (0.625, "5/8"), (0.6875, "11/16"), (0.75, "3/4"),
        (0.8125, "13/16"), (0.875, "7/8"), (0.9375, "15/16")
    ]
    
    if remainder < 0.03125:  # Less than 1/32
        return str(whole) if whole > 0 else "0"
    
    # Find closest fraction
    closest_frac = min(fractions, key=lambda x: abs(x[0] - remainder))
    
    if whole > 0:
        return f"{whole} {closest_frac[1]}"
    else:
        return closest_frac[1]

if __name__ == "__main__":
    print("\nSTARTING CRITICAL REDUNDANT VERIFICATION")
    print("This performs a completely independent check")
    print("Does NOT trust any previous processing\n")
    
    original = "archive/door_231_order_form.html"
    generated = "output/paul_revere_231/finish_door_list.html"
    
    if os.path.exists(original) and os.path.exists(generated):
        result = perform_critical_verification(original, generated)
    else:
        print("[ERROR] Files not found")
        print(f"   Original: {original} - {'EXISTS' if os.path.exists(original) else 'NOT FOUND'}")
        print(f"   Generated: {generated} - {'EXISTS' if os.path.exists(generated) else 'NOT FOUND'}")