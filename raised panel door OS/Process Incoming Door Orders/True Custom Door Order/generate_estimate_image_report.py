"""
Generate Estimate Report as an Image
Creates a formatted image showing cost estimate for doors and drawer fronts
Doors: $30.50 each
Drawer Fronts: $25.50 each
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

def generate_image_report(order_json_path, output_path):
    """Generate estimate report as image"""

    # Load data
    with open(order_json_path, 'r') as f:
        order_data = json.load(f)

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

    # Process doors
    doors = order_data['doors_table']['rows']
    door_items_regular = []  # Under 4 sq ft
    door_items_by_overage = {}  # Group by sq ft over: {1: [], 2: [], 3: [], etc}
    total_doors = 0

    base_price = 30.50
    price_per_sqft = base_price / 4  # $7.625 per sq ft

    import math

    for door in doors:
        marker = door['marker']
        qty = int(door['qty'])
        width_str = door['width']
        height_str = door['height']
        note = door['note']

        # Calculate square footage
        width = fraction_to_decimal(width_str)
        height = fraction_to_decimal(height_str)
        sqft = (width * height) / 144  # Convert sq inches to sq ft

        # Calculate price based on square footage
        if sqft <= 4:
            price_each = base_price
            door_items_regular.append({
                'marker': marker,
                'qty': qty,
                'width': width_str,
                'height': height_str,
                'note': note,
                'sqft': sqft,
                'price_each': price_each,
                'subtotal': qty * price_each
            })
        else:
            # Round up additional sq ft
            additional_sqft = math.ceil(sqft - 4)
            price_each = base_price + (additional_sqft * price_per_sqft)

            # Group by additional sqft
            if additional_sqft not in door_items_by_overage:
                door_items_by_overage[additional_sqft] = []

            door_items_by_overage[additional_sqft].append({
                'marker': marker,
                'qty': qty,
                'width': width_str,
                'height': height_str,
                'note': note,
                'sqft': sqft,
                'additional_sqft': additional_sqft,
                'price_each': price_each,
                'subtotal': qty * price_each
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

        drawer_items.append({
            'marker': marker,
            'qty': qty,
            'width': width_str,
            'height': height_str,
            'price_each': 25.50,
            'subtotal': qty * 25.50
        })
        total_drawers += qty

    # Create image for 8.5x11 paper at 150 DPI
    dpi = 150
    width = int(8.5 * dpi)  # 1275 pixels
    height = int(11 * dpi)   # 1650 pixels

    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Use consistent font size except for material
    margin_left = 150
    margin_right = 150
    margin_top = 150

    try:
        font_regular = ImageFont.truetype("arial.ttf", 24)  # Smaller font
        font_material = ImageFont.truetype("arialbd.ttf", 48)  # Bold, larger for material
    except:
        font_regular = ImageFont.load_default()
        font_material = ImageFont.load_default()

    # Header - Company (centered at top)
    company = "The Raised Panel Door Factory"
    company_bbox = draw.textbbox((0, 0), company, font=font_material)
    company_width = company_bbox[2] - company_bbox[0]
    company_x = (width - company_width) // 2
    draw.text((company_x, margin_top), company, fill='black', font=font_material)

    # Title - ESTIMATE (centered below company)
    title = "ESTIMATE"
    title_bbox = draw.textbbox((0, 0), title, font=font_material)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, margin_top + 70), title, fill='black', font=font_material)

    # Date and Time (right aligned)
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p")
    date_bbox = draw.textbbox((0, 0), date_str, font=font_regular)
    date_width = date_bbox[2] - date_bbox[0]
    time_bbox = draw.textbbox((0, 0), time_str, font=font_regular)
    time_width = time_bbox[2] - time_bbox[0]
    draw.text((width - date_width - margin_right, margin_top + 150), date_str, fill='black', font=font_regular)
    draw.text((width - time_width - margin_right, margin_top + 180), time_str, fill='black', font=font_regular)

    # Jobsite (left aligned)
    jobsite_label = f"Job: {jobsite}"
    draw.text((margin_left, margin_top + 150), jobsite_label, fill='black', font=font_regular)

    # Table setup
    row_height = 50
    table_start_y = margin_top + 270
    y = table_start_y

    # Column positions (spread out)
    col_qty = margin_left + 10
    col_description = margin_left + 100
    col_price = margin_left + 450
    col_total = margin_left + 650

    # Draw column headers
    draw.text((col_qty, y), "QTY", fill='black', font=font_regular)
    draw.text((col_description, y), "DESCRIPTION", fill='black', font=font_regular)
    draw.text((col_price, y), "PRICE", fill='black', font=font_regular)
    draw.text((col_total, y), "TOTAL", fill='black', font=font_regular)
    y += 40

    # Draw underline under headers
    draw.line([(margin_left, y), (width - margin_right, y)], fill='black', width=2)
    y += 20

    # DOORS - REGULAR (Under 4 sq ft)
    if door_items_regular:
        # Count total units and get markers
        total_regular_units = sum(item['qty'] for item in door_items_regular)
        regular_total = sum(item['subtotal'] for item in door_items_regular)
        price_each = door_items_regular[0]['price_each']  # All same price

        # Build markers string like "(2) #1, (2) #2, (1) #3"
        markers_list = [f"({item['qty']}) {item['marker']}" for item in door_items_regular]
        markers_str = ", ".join(markers_list)

        # Table-style layout - single line (no cabinet numbers for regular doors)
        description = f"Door 101 - {wood_type}"
        draw.text((col_qty, y + 10), str(total_regular_units), fill='black', font=font_regular)
        draw.text((col_description, y + 10), description, fill='black', font=font_regular)
        draw.text((col_price, y + 10), f"${price_each:.2f}", fill='black', font=font_regular)
        draw.text((col_total, y + 10), f"${regular_total:.2f}", fill='black', font=font_regular)

        y += 50

    # DOORS - OVERSIZE (Over 4 sq ft) - Grouped by sq ft overage
    if door_items_by_overage:
        # Sort by overage amount (1 sq ft, 2 sq ft, etc)
        for additional_sqft in sorted(door_items_by_overage.keys()):
            items = door_items_by_overage[additional_sqft]

            # Count total units and get markers
            total_units = sum(item['qty'] for item in items)
            group_total = sum(item['subtotal'] for item in items)
            price_each = items[0]['price_each']  # All same price in this group

            # Build markers string
            markers_list = [f"({item['qty']}) {item['marker']}" for item in items]
            markers_str = ", ".join(markers_list)

            # Table-style layout - first line
            description = f"Door 101 +{additional_sqft}sf - {wood_type}"
            draw.text((col_qty, y + 10), str(total_units), fill='black', font=font_regular)
            draw.text((col_description, y + 10), description, fill='black', font=font_regular)
            draw.text((col_price, y + 10), f"${price_each:.2f}", fill='black', font=font_regular)
            draw.text((col_total, y + 10), f"${group_total:.2f}", fill='black', font=font_regular)

            # Cabinet numbers on next line
            y += 35
            max_width = width - margin_right - col_description
            words = markers_str.split(', ')
            lines = []
            current_line = []

            for word in words:
                test_line = ', '.join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=font_regular)
                if bbox[2] - bbox[0] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(', '.join(current_line))
                    current_line = [word]

            if current_line:
                lines.append(', '.join(current_line))

            # Draw wrapped lines
            for i, line in enumerate(lines):
                draw.text((col_description, y + (i * 30)), line, fill='black', font=font_regular)

            y += (len(lines) * 30) + 15

    # DRAWER FRONTS Section
    if drawer_items:
        # Count total units and get markers
        total_drawer_units = sum(item['qty'] for item in drawer_items)
        drawer_total = sum(item['subtotal'] for item in drawer_items)
        price_each = drawer_items[0]['price_each']  # All same price

        # Build markers string
        markers_list = [f"({item['qty']}) {item['marker']}" for item in drawer_items]
        markers_str = ", ".join(markers_list)

        # Table-style layout - single line (no cabinet numbers for drawers)
        description = f"Drawer 101 - {wood_type}"
        draw.text((col_qty, y + 10), str(total_drawer_units), fill='black', font=font_regular)
        draw.text((col_description, y + 10), description, fill='black', font=font_regular)
        draw.text((col_price, y + 10), f"${price_each:.2f}", fill='black', font=font_regular)
        draw.text((col_total, y + 10), f"${drawer_total:.2f}", fill='black', font=font_regular)

        y += 50

    # Grand Total
    oversize_total = sum(item['subtotal'] for group in door_items_by_overage.values() for item in group)
    grand_total = sum(item['subtotal'] for item in door_items_regular) + oversize_total + sum(item['subtotal'] for item in drawer_items)
    total_units = total_doors + total_drawers

    y += 20
    draw.rectangle([(margin_left, y), (width - margin_right, y + row_height)], fill='#404040')
    draw.text((col_price, y + 10), "GRAND TOTAL:", fill='white', font=font_regular)
    draw.text((col_total, y + 10), f"${grand_total:.2f}", fill='white', font=font_regular)

    # Save image
    img.save(output_path)
    print(f"[OK] Image report saved to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify order JSON path")
        print("Usage: python generate_estimate_image_report.py <order_json_path>")
        sys.exit(1)

    order_json_path = sys.argv[1]

    output_dir = os.path.dirname(order_json_path)
    base_name = os.path.splitext(os.path.basename(order_json_path))[0]
    base_name = base_name.replace('_two_pass', '')

    output_path = os.path.join(output_dir, f"{base_name}_estimate_report.png")

    if not os.path.exists(order_json_path):
        print(f"[ERROR] Order JSON not found: {order_json_path}")
        sys.exit(1)

    generate_image_report(order_json_path, output_path)
