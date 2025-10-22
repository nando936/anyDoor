"""
Generate Stile and Rail Report as an Image
Creates a formatted image similar to the Door Cut List capture
"""
import os
import sys
import json
from fractions import Fraction
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

def fraction_to_decimal(measurement):
    """Convert measurement string like '23 15/16' to decimal"""
    if not measurement or measurement == '':
        return 0.0

    parts = measurement.strip().split()

    if len(parts) == 0:
        return 0.0

    if len(parts) == 1:
        if '/' in parts[0]:
            frac = Fraction(parts[0])
            return float(frac)
        else:
            return float(parts[0])

    whole = int(parts[0])
    frac = Fraction(parts[1])
    return whole + float(frac)

def decimal_to_fraction(decimal_value):
    """Convert decimal to fraction string like '23 15/16'"""
    whole = int(decimal_value)
    remainder = decimal_value - whole

    if remainder < 0.001:
        return str(whole)

    sixteenths = round(remainder * 16)

    if sixteenths == 0:
        return str(whole)
    elif sixteenths == 16:
        return str(whole + 1)

    frac = Fraction(sixteenths, 16)

    if whole == 0:
        return str(frac)
    else:
        return f"{whole} {frac}"

def calculate_parts(finish_width, finish_height, door_style):
    """Calculate stile and rail dimensions"""
    stile_width = door_style['dimensions']['stile_width_decimal']
    rail_width = door_style['dimensions']['rail_width_decimal']
    stile_oversize = door_style['constants']['stile_oversize']
    rail_oversize = door_style['constants']['rail_oversize']
    stile_length_add = door_style['constants']['stile_length_addition']
    tongue_length = door_style['constants']['tongue_length']

    stile_qty = 2
    stile_cut_width = stile_width + stile_oversize
    stile_cut_length = finish_height + stile_length_add

    rail_qty = 2
    rail_cut_width = rail_width + rail_oversize
    rail_cut_length = finish_width - (stile_width * 2) + tongue_length

    return {
        'stiles': {
            'qty': stile_qty,
            'width': stile_cut_width,
            'length': stile_cut_length,
            'width_frac': decimal_to_fraction(stile_cut_width),
            'length_frac': decimal_to_fraction(stile_cut_length)
        },
        'rails': {
            'qty': rail_qty,
            'width': rail_cut_width,
            'length': rail_cut_length,
            'width_frac': decimal_to_fraction(rail_cut_width),
            'length_frac': decimal_to_fraction(rail_cut_length)
        }
    }

def generate_image_report(order_json_path, door_style_path, output_path):
    """Generate stile and rail report as image"""

    # Load data
    with open(order_json_path, 'r') as f:
        order_data = json.load(f)

    with open(door_style_path, 'r') as f:
        door_style = json.load(f)

    # Get info
    jobsite = order_data.get('fields', {}).get('jobsite', 'Unknown')
    wood_type = order_data.get('fields', {}).get('wood_type', 'Unknown')

    # Check size type
    check1 = order_data.get('check1', {})
    finished_checked = check1.get('finished_door_size', {}).get('checked', False)
    opening_checked = check1.get('opening_size', {}).get('checked', False)
    opening_add_value = check1.get('opening_size', {}).get('add_value', 'EMPTY')

    size_adjustment = 0.0
    if not finished_checked and opening_checked and opening_add_value != 'EMPTY':
        try:
            add_per_side = fraction_to_decimal(opening_add_value)
            size_adjustment = add_per_side * 2
        except:
            size_adjustment = 0.0

    # Collect all parts
    all_parts = []
    doors = order_data['doors_table']['rows']
    for door in doors:
        marker = door['marker']
        qty = int(door['qty'])
        width = fraction_to_decimal(door['width']) + size_adjustment
        height = fraction_to_decimal(door['height']) + size_adjustment
        parts = calculate_parts(width, height, door_style)

        all_parts.append({
            'qty': parts['stiles']['qty'] * qty,
            'part': 'STILE',
            'width': parts['stiles']['width'],
            'length': parts['stiles']['length'],
            'width_frac': parts['stiles']['width_frac'],
            'length_frac': parts['stiles']['length_frac'],
            'row': marker
        })

        all_parts.append({
            'qty': parts['rails']['qty'] * qty,
            'part': 'RAIL',
            'width': parts['rails']['width'],
            'length': parts['rails']['length'],
            'width_frac': parts['rails']['width_frac'],
            'length_frac': parts['rails']['length_frac'],
            'row': marker
        })

    # Process drawer fronts
    drawers = order_data['drawer_fronts_table']['rows']
    for drawer in drawers:
        marker = drawer['marker']
        qty = int(drawer['qty'])
        width = fraction_to_decimal(drawer['width']) + size_adjustment
        height = fraction_to_decimal(drawer['height']) + size_adjustment
        parts = calculate_parts(width, height, door_style)

        all_parts.append({
            'qty': parts['stiles']['qty'] * qty,
            'part': 'STILE',
            'width': parts['stiles']['width'],
            'length': parts['stiles']['length'],
            'width_frac': parts['stiles']['width_frac'],
            'length_frac': parts['stiles']['length_frac'],
            'row': marker
        })

        all_parts.append({
            'qty': parts['rails']['qty'] * qty,
            'part': 'RAIL',
            'width': parts['rails']['width'],
            'length': parts['rails']['length'],
            'width_frac': parts['rails']['width_frac'],
            'length_frac': parts['rails']['length_frac'],
            'row': marker
        })

    # Group parts
    from collections import defaultdict
    grouped = defaultdict(list)
    for part in all_parts:
        key = (part['part'], part['width'], part['length'], part['width_frac'], part['length_frac'])
        grouped[key].append({'qty': part['qty'], 'row': part['row']})

    grouped_list = []
    for key, rows in grouped.items():
        part_type, width, length, width_frac, length_frac = key
        total_qty = sum(r['qty'] for r in rows)
        grouped_list.append({
            'part': part_type,
            'width_frac': width_frac,
            'length_frac': length_frac,
            'total_qty': total_qty,
            'rows': rows,
            'length': length
        })

    grouped_list.sort(key=lambda x: x['length'], reverse=True)

    # Create image for 8.5x11 paper at 150 DPI
    dpi = 150
    width = int(8.5 * dpi)  # 1275 pixels
    height = int(11 * dpi)   # 1650 pixels

    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Use consistent font size except for material
    # 1 inch margins at 150 DPI = 150 pixels
    margin_left = 150
    margin_right = 150
    margin_top = 150

    try:
        font_regular = ImageFont.truetype("arial.ttf", 32)
        font_material = ImageFont.truetype("arialbd.ttf", 48)  # Bold, larger for material
    except:
        font_regular = ImageFont.load_default()
        font_material = ImageFont.load_default()

    # Header - Jobsite (top left)
    draw.text((margin_left, margin_top), jobsite, fill='black', font=font_regular)

    # Header - Company, Date and Time (top right)
    company = "The Raised Panel Door Factory"
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p")

    # Right align company and date
    company_bbox = draw.textbbox((0, 0), company, font=font_regular)
    company_width = company_bbox[2] - company_bbox[0]
    date_bbox = draw.textbbox((0, 0), date_str, font=font_regular)
    date_width = date_bbox[2] - date_bbox[0]
    time_bbox = draw.textbbox((0, 0), time_str, font=font_regular)
    time_width = time_bbox[2] - time_bbox[0]

    draw.text((width - company_width - margin_right, margin_top), company, fill='black', font=font_regular)
    draw.text((width - date_width - margin_right, margin_top + 40), date_str, fill='black', font=font_regular)
    draw.text((width - time_width - margin_right, margin_top + 80), time_str, fill='black', font=font_regular)

    # Title
    title = "Door Cut List - Stile and Rails"
    draw.text((margin_left, margin_top + 120), title, fill='black', font=font_regular)

    # Material (prominent and centered)
    material_bbox = draw.textbbox((0, 0), wood_type, font=font_material)
    material_width = material_bbox[2] - material_bbox[0]
    material_x = (width - material_width) // 2
    draw.text((material_x, margin_top + 180), wood_type, fill='black', font=font_material)

    # Table setup
    row_height = 50
    table_start_y = margin_top + 270
    header_y = table_start_y

    # Table header (gray background)
    draw.rectangle([(margin_left, header_y), (width - margin_right, header_y + row_height)], fill='#C0C0C0')

    # Column headers - adjusted for larger font
    draw.text((margin_left + 10, header_y + 10), "QTY", fill='black', font=font_regular)
    draw.text((margin_left + 120, header_y + 10), "PART", fill='black', font=font_regular)
    draw.text((margin_left + 280, header_y + 10), "WIDTH", fill='black', font=font_regular)
    draw.text((margin_left + 440, header_y + 10), "LENGTH", fill='black', font=font_regular)
    draw.text((margin_left + 620, header_y + 10), "(QTY) CAB #", fill='black', font=font_regular)

    # Table rows
    y = header_y + row_height
    for item in grouped_list:
        rows_str = ", ".join([f"({r['qty']}) {r['row']}" for r in item['rows']])

        draw.text((margin_left + 10, y + 10), str(item['total_qty']), fill='black', font=font_regular)
        draw.text((margin_left + 120, y + 10), item['part'], fill='black', font=font_regular)
        draw.text((margin_left + 280, y + 10), item['width_frac'], fill='black', font=font_regular)
        draw.text((margin_left + 440, y + 10), item['length_frac'], fill='black', font=font_regular)
        draw.text((margin_left + 620, y + 10), rows_str, fill='black', font=font_regular)

        # Draw row separator line
        y += row_height
        draw.line([(margin_left, y), (width - margin_right, y)], fill='#E0E0E0', width=1)

    # Save image
    img.save(output_path)
    print(f"[OK] Image report saved to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify order JSON path")
        print("Usage: python generate_stile_rail_image_report.py <order_json_path>")
        sys.exit(1)

    order_json_path = sys.argv[1]

    output_dir = os.path.dirname(order_json_path)
    base_name = os.path.splitext(os.path.basename(order_json_path))[0]
    base_name = base_name.replace('_two_pass', '')

    output_path = os.path.join(output_dir, f"{base_name}_stile_rail_report.png")

    door_style_path = "C:/Users/nando/Projects/anyDoor/raised panel door OS/Door Catalog/Door Style/101_shaker.json"

    if not os.path.exists(order_json_path):
        print(f"[ERROR] Order JSON not found: {order_json_path}")
        sys.exit(1)

    if not os.path.exists(door_style_path):
        print(f"[ERROR] Door style not found: {door_style_path}")
        sys.exit(1)

    generate_image_report(order_json_path, door_style_path, output_path)
