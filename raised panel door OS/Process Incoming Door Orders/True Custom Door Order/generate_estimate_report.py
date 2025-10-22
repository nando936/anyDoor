"""
Generate Estimate Report from Door Order
Reads extracted door order data and calculates cost estimate
Doors: $30.50 each
Drawer Fronts: $25.50 each
"""
import os
import sys
import json
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP

def fraction_to_decimal(measurement):
    """Convert measurement string like '23 15/16' to decimal"""
    if not measurement or measurement == '':
        return 0.0

    parts = measurement.strip().split()

    if len(parts) == 0:
        return 0.0

    # Check if it's just a number
    if len(parts) == 1:
        if '/' in parts[0]:
            # Just a fraction like "15/16"
            frac = Fraction(parts[0])
            return float(frac)
        else:
            # Just a whole number
            return float(parts[0])

    # Whole number + fraction like "23 15/16"
    whole = int(parts[0])
    frac = Fraction(parts[1])
    return whole + float(frac)

def generate_report(order_json_path, output_path):
    """Generate estimate report"""

    # Load order data
    with open(order_json_path, 'r') as f:
        order_data = json.load(f)

    # Check which size type to use
    check1 = order_data.get('check1', {})
    finished_checked = check1.get('finished_door_size', {}).get('checked', False)
    opening_checked = check1.get('opening_size', {}).get('checked', False)
    opening_add_value = check1.get('opening_size', {}).get('add_value', 'EMPTY')

    # Determine size adjustment
    size_adjustment = 0.0
    size_type = "FINISHED"

    if not finished_checked and opening_checked and opening_add_value != 'EMPTY':
        # Use opening size - add the adjustment to each side
        try:
            add_per_side = fraction_to_decimal(opening_add_value)
            size_adjustment = add_per_side * 2  # Both sides
            size_type = f"OPENING (ADD {opening_add_value} TO EACH SIDE)"
        except:
            size_adjustment = 0.0
            size_type = "FINISHED (DEFAULT)"

    # Get jobsite and material info
    jobsite = order_data.get('fields', {}).get('jobsite', 'Unknown')
    wood_type = order_data.get('fields', {}).get('wood_type', 'Unknown')

    # Format material line centered
    material_text = f"*** MATERIAL: {wood_type.upper()} ***"
    material_line = material_text.center(80)

    print("=" * 80)
    print("ESTIMATE REPORT - TRUE CUSTOM DOORS")
    print("=" * 80)
    print(f"Jobsite: {jobsite}")
    print(f"Size Type: {size_type}")
    print(f"Size Adjustment: {size_adjustment} inches")
    print("=" * 80)
    print(material_line)
    print("=" * 80)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ESTIMATE REPORT - TRUE CUSTOM DOORS")
    report_lines.append("=" * 80)
    report_lines.append(f"Jobsite: {jobsite}")
    report_lines.append(f"Size Type: {size_type}")
    report_lines.append(f"Size Adjustment: {size_adjustment} inches")
    report_lines.append("=" * 80)
    report_lines.append(material_line)
    report_lines.append("=" * 80)
    report_lines.append("")

    # Process doors
    doors = order_data['doors_table']['rows']
    door_items = []
    total_doors = 0

    for door in doors:
        marker = door['marker']
        qty = int(door['qty'])
        width_str = door['width']
        height_str = door['height']
        note = door['note']

        # Convert to decimal
        width = fraction_to_decimal(width_str)
        height = fraction_to_decimal(height_str)

        # Apply size adjustment if needed
        actual_width = width + size_adjustment
        actual_height = height + size_adjustment

        door_items.append({
            'marker': marker,
            'qty': qty,
            'width': width_str,
            'height': height_str,
            'note': note,
            'price_each': 30.50,
            'subtotal': qty * 30.50
        })
        total_doors += qty

    # Process drawer fronts
    drawers = order_data['drawer_fronts_table']['rows']
    drawer_items = []
    total_drawers = 0

    for drawer in drawers:
        marker = drawer['marker']
        qty = int(drawer['qty'])
        width_str = drawer['width']
        height_str = drawer['height']

        # Convert to decimal
        width = fraction_to_decimal(width_str)
        height = fraction_to_decimal(height_str)

        # Apply size adjustment if needed
        actual_width = width + size_adjustment
        actual_height = height + size_adjustment

        drawer_items.append({
            'marker': marker,
            'qty': qty,
            'width': width_str,
            'height': height_str,
            'price_each': 25.50,
            'subtotal': qty * 25.50
        })
        total_drawers += qty

    # Print DOORS section
    if door_items:
        header_line = "\033[7mMARKER\033[0m | \033[7mQTY\033[0m | \033[7mWIDTH x HEIGHT     \033[0m | \033[7mPRICE EA\033[0m | \033[7mSUBTOTAL\033[0m | \033[7mNOTE\033[0m"
        header_line_plain = "MARKER | QTY | WIDTH x HEIGHT      | PRICE EA | SUBTOTAL | NOTE"

        print("")
        print("\033[7m*** DOORS ***\033[0m")
        print(header_line)
        print("-" * 80)

        report_lines.append("")
        report_lines.append("*** DOORS ***")
        report_lines.append(header_line_plain)
        report_lines.append("-" * 80)

        for item in door_items:
            size = f"{item['width']} x {item['height']}"
            line = f"{item['marker']:6s} | {item['qty']:3d} | {size:19s} | ${item['price_each']:7.2f} | ${item['subtotal']:8.2f} | {item['note']}"
            print(line)
            report_lines.append(line)

        print("-" * 80)
        door_total = sum(item['subtotal'] for item in door_items)
        total_line = f"DOORS TOTAL: {total_doors} units | ${door_total:.2f}"
        print(total_line)
        print("-" * 80)

        report_lines.append("-" * 80)
        report_lines.append(total_line)
        report_lines.append("-" * 80)

    # Print DRAWER FRONTS section
    if drawer_items:
        header_line = "\033[7mMARKER\033[0m | \033[7mQTY\033[0m | \033[7mWIDTH x HEIGHT     \033[0m | \033[7mPRICE EA\033[0m | \033[7mSUBTOTAL\033[0m"
        header_line_plain = "MARKER | QTY | WIDTH x HEIGHT      | PRICE EA | SUBTOTAL"

        print("")
        print("\033[7m*** DRAWER FRONTS ***\033[0m")
        print(header_line)
        print("-" * 80)

        report_lines.append("")
        report_lines.append("*** DRAWER FRONTS ***")
        report_lines.append(header_line_plain)
        report_lines.append("-" * 80)

        for item in drawer_items:
            size = f"{item['width']} x {item['height']}"
            line = f"{item['marker']:6s} | {item['qty']:3d} | {size:19s} | ${item['price_each']:7.2f} | ${item['subtotal']:8.2f}"
            print(line)
            report_lines.append(line)

        print("-" * 80)
        drawer_total = sum(item['subtotal'] for item in drawer_items)
        total_line = f"DRAWER FRONTS TOTAL: {total_drawers} units | ${drawer_total:.2f}"
        print(total_line)
        print("-" * 80)

        report_lines.append("-" * 80)
        report_lines.append(total_line)
        report_lines.append("-" * 80)

    # Print GRAND TOTAL
    grand_total = sum(item['subtotal'] for item in door_items) + sum(item['subtotal'] for item in drawer_items)
    total_units = total_doors + total_drawers

    print("")
    print("=" * 80)
    print(f"\033[7mGRAND TOTAL: {total_units} units | ${grand_total:.2f}\033[0m")
    print("=" * 80)

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append(f"GRAND TOTAL: {total_units} units | ${grand_total:.2f}")
    report_lines.append("=" * 80)

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print("\n" + "=" * 80)
    print(f"[OK] Report saved to: {output_path}")
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify order JSON path")
        print("Usage: python generate_estimate_report.py <order_json_path>")
        sys.exit(1)

    order_json_path = sys.argv[1]

    # Derive paths
    output_dir = os.path.dirname(order_json_path)
    base_name = os.path.splitext(os.path.basename(order_json_path))[0]
    base_name = base_name.replace('_two_pass', '')  # Remove _two_pass suffix

    output_path = os.path.join(output_dir, f"{base_name}_estimate_report.txt")

    if not os.path.exists(order_json_path):
        print(f"[ERROR] Order JSON not found: {order_json_path}")
        sys.exit(1)

    generate_report(order_json_path, output_path)
