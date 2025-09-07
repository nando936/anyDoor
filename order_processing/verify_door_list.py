"""
CRITICAL VERIFICATION SCRIPT
Compares original order against generated door list to prevent expensive mistakes
"""

import sys
import os
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

def verify_door_list(original_html_path, generated_html_path):
    """
    Critical verification of door list against original order
    """
    print("=" * 60)
    print("[CRITICAL VERIFICATION] Door List vs Original Order")
    print("=" * 60)
    print("This verification is CRITICAL to prevent expensive production mistakes")
    print()
    
    # Read original order
    with open(original_html_path, 'r', encoding='utf-8') as f:
        original_soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Read generated door list
    with open(generated_html_path, 'r', encoding='utf-8') as f:
        generated_soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Extract original order data
    print("[1] Analyzing Original Order...")
    original_items = []
    
    # Find the order table in original
    order_table = original_soup.find('table', class_='order-table')
    if order_table:
        rows = order_table.find_all('tr')[1:]  # Skip header
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 6 and cells[1].text.strip():  # Has quantity
                item = {
                    'line': cells[0].text.strip(),
                    'qty': cells[1].text.strip(),
                    'width': cells[2].text.strip(),
                    'height': cells[3].text.strip(),
                    'type': cells[4].text.strip(),
                    'notes': cells[5].text.strip() if len(cells) > 5 else ''
                }
                if item['qty']:  # Only add if has quantity
                    original_items.append(item)
                    print(f"   Line #{item['line']}: {item['qty']} @ {item['width']} x {item['height']} - {item['notes']}")
    
    # Check if original is Opening or Finish sizes
    door_sizes = "Finish Sizes"  # Default
    checkboxes = original_soup.find_all('span', class_='checkbox')
    for checkbox in checkboxes:
        if 'checked' in checkbox.get('class', []):
            parent_text = checkbox.parent.text
            if 'Opening' in parent_text:
                door_sizes = "Opening Sizes"
                break
    
    print(f"\n   Original Order Type: {door_sizes}")
    
    # Extract overlay value
    overlay_text = "1/2\""  # Default
    for td in original_soup.find_all('td'):
        if 'Overlay' in td.text:
            next_td = td.find_next_sibling('td')
            if next_td:
                overlay_text = next_td.text.strip()
                break
    
    print(f"   Overlay: {overlay_text}")
    
    # Calculate expected finish sizes
    print("\n[2] Calculating Expected Finish Sizes...")
    overlay_value = 0.5  # Default 1/2"
    if '1/2' in overlay_text:
        overlay_value = 0.5
    elif '3/4' in overlay_text:
        overlay_value = 0.75
    elif '1/4' in overlay_text:
        overlay_value = 0.25
    
    for item in original_items:
        if door_sizes == "Opening Sizes":
            # Convert to finish size
            width_value = parse_dimension(item['width'])
            height_value = parse_dimension(item['height'])
            finish_width = width_value + (2 * overlay_value)
            finish_height = height_value + (2 * overlay_value)
            item['expected_width'] = format_dimension(finish_width)
            item['expected_height'] = format_dimension(finish_height)
            print(f"   Cabinet #{item['line']}: {item['width']} + 1\" = {item['expected_width']} (width)")
        else:
            # Finish sizes stay the same
            item['expected_width'] = item['width']
            item['expected_height'] = item['height']
    
    # Extract generated door list data
    print("\n[3] Analyzing Generated Door List...")
    generated_items = []
    
    # Find the main table in generated list
    tables = generated_soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')[1:]  # Skip header
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 4:
                # Extract values from input fields
                inputs = row.find_all('input')
                if inputs and len(inputs) >= 4:
                    cabinet_num = cells[0].text.strip().replace('#', '')
                    item = {
                        'cabinet': cabinet_num,
                        'qty': inputs[0].get('value', ''),
                        'width': inputs[1].get('value', ''),
                        'height': inputs[2].get('value', ''),
                        'type': inputs[3].get('value', '') if len(inputs) > 3 else '',
                        'notes': inputs[-1].get('value', '') if len(inputs) > 4 else ''
                    }
                    generated_items.append(item)
                    print(f"   Cabinet #{item['cabinet']}: {item['qty']} @ {item['width']} x {item['height']}")
    
    # Verify counts match
    print("\n[4] VERIFICATION RESULTS:")
    print("-" * 40)
    
    errors = []
    warnings = []
    
    # Check item count
    if len(original_items) != len(generated_items):
        errors.append(f"[ERROR] Item count mismatch! Original: {len(original_items)}, Generated: {len(generated_items)}")
    
    # Check each item
    for i, orig in enumerate(original_items):
        if i < len(generated_items):
            gen = generated_items[i]
            
            # Check quantity
            if orig['qty'] != gen['qty']:
                errors.append(f"[ERROR] Cabinet #{orig['line']} quantity: Expected {orig['qty']}, Got {gen['qty']}")
            
            # Check dimensions
            if orig.get('expected_width') != gen['width']:
                errors.append(f"[ERROR] Cabinet #{orig['line']} width: Expected {orig.get('expected_width')}, Got {gen['width']}")
            
            if orig.get('expected_height') != gen['height']:
                errors.append(f"[ERROR] Cabinet #{orig['line']} height: Expected {orig.get('expected_height')}, Got {gen['height']}")
            
            # Check special notes
            if 'no bore' in orig['notes'].lower() and 'no' not in gen['notes'].lower():
                warnings.append(f"[WARNING] Cabinet #{orig['line']}: 'No bore' instruction may be missing")
    
    # Print results
    if errors:
        print("\n[X] CRITICAL ERRORS FOUND:")
        for error in errors:
            print(f"   {error}")
        print("\n   [!] DO NOT PROCEED - FIX THESE ERRORS FIRST")
    else:
        print("\n[OK] All quantities and dimensions verified correctly")
    
    if warnings:
        print("\n[!] WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
    
    print("\n" + "=" * 60)
    return len(errors) == 0

def parse_dimension(dim_str):
    """Parse dimension string to decimal"""
    if not dim_str:
        return 0
    parts = dim_str.strip().split()
    total = 0
    for part in parts:
        if '/' in part:
            num, den = part.split('/')
            total += float(num) / float(den)
        else:
            try:
                total += float(part)
            except:
                pass
    return total

def format_dimension(value):
    """Convert decimal back to fractional format"""
    whole = int(value)
    fraction = value - whole
    
    if fraction < 0.0625:  # Less than 1/16
        return str(whole) if whole > 0 else "0"
    
    fractions = [
        (1/16, "1/16"), (1/8, "1/8"), (3/16, "3/16"), (1/4, "1/4"),
        (5/16, "5/16"), (3/8, "3/8"), (7/16, "7/16"), (1/2, "1/2"),
        (9/16, "9/16"), (5/8, "5/8"), (11/16, "11/16"), (3/4, "3/4"),
        (13/16, "13/16"), (7/8, "7/8"), (15/16, "15/16")
    ]
    
    closest = min(fractions, key=lambda x: abs(x[0] - fraction))
    
    if whole > 0:
        return f"{whole} {closest[1]}"
    else:
        return closest[1]

if __name__ == "__main__":
    # Test with current files
    original = "archive/door_231_order_form.html"
    generated = "output/door_list_test/finish_door_list_test.html"
    
    if os.path.exists(original) and os.path.exists(generated):
        result = verify_door_list(original, generated)
        if result:
            print("[OK] Verification passed - safe to proceed")
        else:
            print("[X] Verification failed - DO NOT proceed to production")
    else:
        print("[ERROR] Files not found. Please specify paths to original and generated door list.")