"""
Generate Center Panel Report from Door Order
Reads extracted door order data and door style specification to create a panel cutting list
Works with unified door order format
"""
import os
import sys
import json
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP

# Import shared utilities
from shared_utils import load_unified_json, fraction_to_decimal

def decimal_to_fraction(decimal_value):
    """Convert decimal to fraction string like '23 15/16'"""
    whole = int(decimal_value)
    remainder = decimal_value - whole

    if remainder < 0.001:  # Essentially zero
        return str(whole)

    # Convert to fraction (assume 16ths for wood working)
    sixteenths = round(remainder * 16)

    if sixteenths == 0:
        return str(whole)
    elif sixteenths == 16:
        return str(whole + 1)

    # Simplify fraction
    frac = Fraction(sixteenths, 16)

    if whole == 0:
        return str(frac)
    else:
        return f"{whole} {frac}"

def calculate_panel(finish_width, finish_height, door_style):
    """Calculate panel dimensions from finish size and door style"""

    # Get constants from door style
    stile_width = door_style['dimensions']['stile_width_decimal']
    tongue_length = door_style['constants']['tongue_length']
    panel_clearance = door_style['constants']['panel_clearance']

    # Calculate panel dimensions from formulas
    panel_width = finish_width - (stile_width * 2) + tongue_length - panel_clearance
    panel_length = finish_height - (stile_width * 2 + tongue_length) - panel_clearance

    return {
        'width': panel_width,
        'length': panel_length,
        'width_frac': decimal_to_fraction(panel_width),
        'length_frac': decimal_to_fraction(panel_length)
    }

def generate_report(order_json_path, door_style_path, output_path):
    """Generate panel report"""

    # Load unified order data
    order_data = load_unified_json(order_json_path)
    if not order_data:
        print(f"[ERROR] Failed to load unified order data from {order_json_path}")
        return

    # Load door style
    with open(door_style_path, 'r') as f:
        door_style = json.load(f)

    # In unified format, sizes are already finished - no adjustment needed
    size_adjustment = 0.0
    size_info = order_data.get('size_info', {})
    if size_info.get('all_sizes_are_finished'):
        size_type = "FINISHED"
    else:
        size_type = "FINISHED (DEFAULT)"

    # Get jobsite and material info from unified format
    jobsite = order_data.get('order_info', {}).get('jobsite', 'Unknown')
    panel_cut = order_data.get('specifications', {}).get('panel_cut', 'Unknown')

    # Format material line centered
    material_text = f"*** MATERIAL: {panel_cut.upper()} ***"
    material_line = material_text.center(80)

    print("=" * 80)
    print(f"CENTER PANEL REPORT - Door Style {door_style['style_number']} ({door_style['style_name']})")
    print("=" * 80)
    print(f"Jobsite: {jobsite}")
    print(f"Size Type: {size_type}")
    print(f"Size Adjustment: {size_adjustment} inches")
    print("=" * 80)
    print(material_line)
    print("=" * 80)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"CENTER PANEL REPORT - Door Style {door_style['style_number']} ({door_style['style_name']})")
    report_lines.append("=" * 80)
    report_lines.append(f"Jobsite: {jobsite}")
    report_lines.append(f"Size Type: {size_type}")
    report_lines.append(f"Size Adjustment: {size_adjustment} inches")
    report_lines.append("=" * 80)
    report_lines.append(material_line)
    report_lines.append("=" * 80)
    report_lines.append("")

    # Collect all panels
    all_panels = []

    # Process doors
    doors = order_data.get('doors', [])

    for door in doors:
        marker = door['marker']
        qty = int(door['qty'])

        # Use decimal values directly from unified format
        width = door['width_decimal']
        height = door['height_decimal']

        # Apply size adjustment if needed
        actual_width = width + size_adjustment
        actual_height = height + size_adjustment

        # Calculate panel
        panel = calculate_panel(actual_width, actual_height, door_style)

        # Add panel to list
        all_panels.append({
            'qty': qty,
            'width': panel['width'],
            'length': panel['length'],
            'width_frac': panel['width_frac'],
            'length_frac': panel['length_frac'],
            'row': marker
        })

    # Process drawer fronts
    drawers = order_data.get('drawers', [])

    for drawer in drawers:
        marker = drawer['marker']
        qty = int(drawer['qty'])

        # Use decimal values directly from unified format
        width = drawer['width_decimal']
        height = drawer['height_decimal']

        # Apply size adjustment if needed
        actual_width = width + size_adjustment
        actual_height = height + size_adjustment

        # Calculate panel
        panel = calculate_panel(actual_width, actual_height, door_style)

        # Add panel to list
        all_panels.append({
            'qty': qty,
            'width': panel['width'],
            'length': panel['length'],
            'width_frac': panel['width_frac'],
            'length_frac': panel['length_frac'],
            'row': marker
        })

    # Group panels by width and length
    from collections import defaultdict
    grouped = defaultdict(list)

    for panel in all_panels:
        key = (panel['width'], panel['length'], panel['width_frac'], panel['length_frac'])
        grouped[key].append({'qty': panel['qty'], 'row': panel['row']})

    # Convert to list and sort by length descending
    grouped_list = []
    for key, rows in grouped.items():
        width, length, width_frac, length_frac = key
        total_qty = sum(r['qty'] for r in rows)
        grouped_list.append({
            'width': width,
            'length': length,
            'width_frac': width_frac,
            'length_frac': length_frac,
            'total_qty': total_qty,
            'rows': rows
        })

    grouped_list.sort(key=lambda x: x['length'], reverse=True)

    # Print report
    print("\nTOTAL | WIDTH x LENGTH      | ROWS")
    print("-" * 80)
    report_lines.append("\nTOTAL | WIDTH x LENGTH      | ROWS")
    report_lines.append("-" * 80)

    for item in grouped_list:
        # Build rows string like "(4) #5, (4) #8, (2) #11"
        rows_str = ", ".join([f"({r['qty']}) {r['row']}" for r in item['rows']])

        # Format line
        line = f"{item['total_qty']:5d} | {item['width_frac']:6s} x {item['length_frac']:10s} | {rows_str}"
        print(line)
        report_lines.append(line)

        # Add separator after each line
        separator = "-" * 80
        print(separator)
        report_lines.append(separator)

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print("\n" + "=" * 80)
    print(f"[OK] Report saved to: {output_path}")
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify order JSON path")
        print("Usage: python generate_panel_report.py <order_json_path>")
        sys.exit(1)

    order_json_path = sys.argv[1]

    # Derive paths
    output_dir = os.path.dirname(order_json_path)
    base_name = os.path.splitext(os.path.basename(order_json_path))[0]
    # Remove suffixes from unified format
    base_name = base_name.replace('_two_pass', '').replace('_unified_door_order', '')

    output_path = os.path.join(output_dir, f"{base_name}_panel_report.txt")

    # Door style path (hardcoded for now - door style 101)
    door_style_path = "C:/Users/nando/Projects/anyDoor/raised panel door OS/Door Catalog/Door Style/101_shaker.json"

    if not os.path.exists(order_json_path):
        print(f"[ERROR] Order JSON not found: {order_json_path}")
        sys.exit(1)

    if not os.path.exists(door_style_path):
        print(f"[ERROR] Door style not found: {door_style_path}")
        sys.exit(1)

    generate_report(order_json_path, door_style_path, output_path)
