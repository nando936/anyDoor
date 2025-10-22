"""
Generate Estimate PDF in Raised Panel Door Factory Format
Matches the company's estimate template exactly
"""
import os
import sys
import json
from fractions import Fraction
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.pdfgen import canvas

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

def generate_estimate_pdf(order_json_path, output_path, estimate_number=None):
    """Generate estimate PDF matching Raised Panel Door Factory template"""

    # Load order data
    with open(order_json_path, 'r') as f:
        order_data = json.load(f)

    # Get info
    jobsite = order_data.get('fields', {}).get('jobsite', 'Unknown')
    wood_type = order_data.get('fields', {}).get('wood_type', 'Unknown')
    door_style = order_data.get('fields', {}).get('door_style', '101')

    # Pricing
    base_door_price = 30.50
    price_per_sqft = base_door_price / 4  # $7.625 per sq ft
    drawer_price = 25.50

    # Process doors - Group by size category
    doors = order_data['doors_table']['rows']

    import math

    door_items_regular = []  # Under 4 sq ft
    door_items_by_overage = {}  # Group by sq ft over: {1: [], 2: [], 3: [], etc}

    for door in doors:
        marker = door['marker']
        qty = int(door['qty'])
        width_str = door['width']
        height_str = door['height']
        note = door.get('note', '')

        # Calculate square footage
        width = fraction_to_decimal(width_str)
        height = fraction_to_decimal(height_str)
        sqft = (width * height) / 144

        # Calculate price
        if sqft <= 4:
            price_each = base_door_price
            door_items_regular.append({
                'marker': marker,
                'qty': qty,
                'price_each': price_each,
                'subtotal': qty * price_each
            })
        else:
            additional_sqft = math.ceil(sqft - 4)
            price_each = base_door_price + (additional_sqft * price_per_sqft)

            if additional_sqft not in door_items_by_overage:
                door_items_by_overage[additional_sqft] = []

            door_items_by_overage[additional_sqft].append({
                'marker': marker,
                'qty': qty,
                'additional_sqft': additional_sqft,
                'price_each': price_each,
                'subtotal': qty * price_each
            })

    # Process drawers
    drawers = order_data['drawer_fronts_table']['rows']
    drawer_items = []

    for drawer in drawers:
        marker = drawer['marker']
        qty = int(drawer['qty'])
        width_str = drawer['width']
        height_str = drawer['height']

        drawer_items.append({
            'marker': marker,
            'qty': qty,
            'price_each': drawer_price,
            'subtotal': qty * drawer_price
        })

    # Build line items with grouped descriptions
    all_line_items = []

    # DOORS - REGULAR (Under 4 sq ft)
    if door_items_regular:
        total_regular_units = sum(item['qty'] for item in door_items_regular)
        regular_total = sum(item['subtotal'] for item in door_items_regular)
        price_each = door_items_regular[0]['price_each']

        # No cabinet numbers for regular doors
        description = f"Door {door_style} - {wood_type}"

        all_line_items.append({
            'description': description,
            'qty': total_regular_units,
            'rate': price_each,
            'amount': regular_total
        })

    # DOORS - OVERSIZE (Over 4 sq ft) - Grouped by sq ft overage
    if door_items_by_overage:
        for additional_sqft in sorted(door_items_by_overage.keys()):
            items = door_items_by_overage[additional_sqft]

            total_units = sum(item['qty'] for item in items)
            group_total = sum(item['subtotal'] for item in items)
            price_each = items[0]['price_each']

            # Build markers string
            markers_list = [f"({item['qty']}) {item['marker']}" for item in items]
            markers_str = ", ".join(markers_list)

            description = f"Door {door_style} +{additional_sqft}sf - {wood_type}\n{markers_str}"

            all_line_items.append({
                'description': description,
                'qty': total_units,
                'rate': price_each,
                'amount': group_total
            })

    # DRAWER FRONTS
    if drawer_items:
        total_drawer_units = sum(item['qty'] for item in drawer_items)
        drawer_total = sum(item['subtotal'] for item in drawer_items)
        price_each = drawer_items[0]['price_each']

        # No cabinet numbers for drawers
        description = f"Drawer {door_style} - {wood_type}"

        all_line_items.append({
            'description': description,
            'qty': total_drawer_units,
            'rate': price_each,
            'amount': drawer_total
        })

    # Calculate totals
    subtotal = sum(item['amount'] for item in all_line_items)
    tax = subtotal * 0.0825  # 8.25% tax
    total = subtotal + tax

    # Create PDF
    pdf = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch
    )

    story = []
    styles = getSampleStyleSheet()

    # Load logo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, 'logo.jpeg')

    logo_img = None
    if os.path.exists(logo_path):
        logo_img = Image(logo_path, width=2.0*inch, height=2.0*inch)

    # Header table with company info and logo
    header_data = [
        ['The Raised Panel Door Factory Inc', logo_img if logo_img else ''],
        ['209 Riggs St', ''],
        ['Conroe, TX  77301 US', ''],
        ['(936) 672-4235', ''],
        ['fernando@theraisedpaneldoor.com', ''],
        ['www.theraisedpaneldoor.com', '']
    ]

    header_table = Table(header_data, colWidths=[4.5*inch, 3*inch], rowHeights=[None, None, None, None, None, None])
    header_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), 12),
        ('FONTSIZE', (0, 1), (0, 5), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('SPAN', (1, 0), (1, 2)),  # Logo spans 3 rows
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.3*inch))

    # Estimate title
    estimate_style = ParagraphStyle(
        'EstimateTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#4A90E2'),
        spaceAfter=0.2*inch
    )
    story.append(Paragraph('Estimate', estimate_style))

    # Customer and estimate info
    if estimate_number is None:
        estimate_number = datetime.now().strftime('%Y%m%d%H%M')

    date_str = datetime.now().strftime('%m/%d/%Y')

    info_data = [
        ['ADDRESS', '', 'ESTIMATE', estimate_number],
        [jobsite, '', 'DATE', date_str],
        ['', '', '', ''],
        ['P.O. NUMBER', '', 'SALES REP', ''],
        ['', '', 'FS', '']
    ]

    info_table = Table(info_data, colWidths=[2*inch, 2.5*inch, 1.5*inch, 1.5*inch])
    info_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, 1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, 0), colors.grey),
        ('TEXTCOLOR', (2, 0), (2, 1), colors.grey),
        ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))

    # Line items table header
    line_items_data = [
        ['DESCRIPTION', 'QTY', 'RATE', 'AMOUNT']
    ]

    # Add line items
    for item in all_line_items:
        line_items_data.append([
            item['description'],
            str(item['qty']),
            f"{item['rate']:.2f}",
            f"{item['amount']:.2f}T"
        ])

    # Add empty rows if needed
    while len(line_items_data) < 8:
        line_items_data.append(['', '', '', ''])

    # Add subtotal, tax, total
    line_items_data.append(['', '', 'SUBTOTAL', f"{subtotal:,.2f}"])
    line_items_data.append(['', '', 'TAX', f"{tax:.2f}"])
    line_items_data.append(['', '', 'TOTAL', f"${total:,.2f}"])

    # Create table
    line_items_table = Table(line_items_data, colWidths=[4*inch, 0.75*inch, 1*inch, 1.75*inch])
    line_items_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D3E5F7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),

        # Data rows
        ('FONTSIZE', (0, 1), (-1, -4), 9),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

        # Subtotal/Tax/Total rows
        ('FONTNAME', (2, -3), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (2, -3), (-1, -1), 10),
        ('LINEABOVE', (2, -3), (-1, -3), 1, colors.grey),
        ('LINEABOVE', (2, -1), (-1, -1), 2, colors.black),
        ('FONTNAME', (2, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (3, -1), (3, -1), 12),
    ]))
    story.append(line_items_table)

    # Signature section
    story.append(Spacer(1, 0.5*inch))
    signature_data = [
        ['Accepted By', ''],
        ['', ''],
        ['Accepted Date', '']
    ]
    signature_table = Table(signature_data, colWidths=[1.5*inch, 5*inch])
    signature_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
    ]))
    story.append(signature_table)

    # Build PDF
    pdf.build(story)
    print(f"[OK] Estimate PDF generated: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify order JSON path")
        print("Usage: python generate_estimate_pdf.py <order_json_path> [estimate_number]")
        sys.exit(1)

    order_json_path = sys.argv[1]
    estimate_number = sys.argv[2] if len(sys.argv) > 2 else None

    output_dir = os.path.dirname(order_json_path)
    base_name = os.path.splitext(os.path.basename(order_json_path))[0]
    base_name = base_name.replace('_two_pass', '')

    output_path = os.path.join(output_dir, f"{base_name}_estimate.pdf")

    if not os.path.exists(order_json_path):
        print(f"[ERROR] Order JSON not found: {order_json_path}")
        sys.exit(1)

    generate_estimate_pdf(order_json_path, output_path, estimate_number)
