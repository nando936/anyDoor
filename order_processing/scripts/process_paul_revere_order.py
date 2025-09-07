"""
Process the Paul Revere (302) Door Order
Extract data from the PDF and generate shop report and cut list
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Order data extracted from the PDF
customer_data = {
    'name': 'SUAREZ CARPENTRY, INC.',
    'job': 'paul revere (302)',
    'date': '17 August 2017',
    'address': '',  # Not in the PDF
    'phone': '',    # Not in the PDF
    'email': ''     # Not in the PDF
}

# All items from the door list
items = [
    # Raised Panel Drawer Front (MDF) - N11, N13, N15 (2 each)
    {'qty': 2, 'width': '16 5/8', 'height': '10 5/16', 'type': 'drawer', 'material': 'MDF', 'style': 'Raised Panel', 'cabinet': 'N11'},
    {'qty': 2, 'width': '16 5/8', 'height': '10 5/16', 'type': 'drawer', 'material': 'MDF', 'style': 'Raised Panel', 'cabinet': 'N13'},
    {'qty': 2, 'width': '16 5/8', 'height': '10 5/16', 'type': 'drawer', 'material': 'MDF', 'style': 'Raised Panel', 'cabinet': 'N15'},
    {'qty': 1, 'width': '17 3/4', 'height': '12 3/8', 'type': 'drawer', 'material': 'MDF', 'style': 'Raised Panel'},
    {'qty': 1, 'width': '17 3/4', 'height': '12 1/4', 'type': 'drawer', 'material': 'MDF', 'style': 'Raised Panel'},
    {'qty': 1, 'width': '15 3/4', 'height': '12 3/8', 'type': 'drawer', 'material': 'MDF', 'style': 'Raised Panel'},
    {'qty': 1, 'width': '15 3/4', 'height': '12 1/4', 'type': 'drawer', 'material': 'MDF', 'style': 'Raised Panel'},
    {'qty': 2, 'width': '13 3/4', 'height': '12 3/8', 'type': 'drawer', 'material': 'MDF', 'style': 'Raised Panel'},
    {'qty': 2, 'width': '13 3/4', 'height': '12 1/4', 'type': 'drawer', 'material': 'MDF', 'style': 'Raised Panel'},
    
    # Solid Wood Drawer Front (Paint Grade)
    {'qty': 1, 'width': '29', 'height': '5 3/8', 'type': 'drawer', 'material': 'Paint Grade', 'style': 'Solid Wood'},
    {'qty': 1, 'width': '27 3/4', 'height': '4 1/4', 'type': 'drawer', 'material': 'Paint Grade', 'style': 'Solid Wood'},
    {'qty': 2, 'width': '27 5/8', 'height': '4 1/4', 'type': 'drawer', 'material': 'Paint Grade', 'style': 'Solid Wood'},
    {'qty': 1, 'width': '17 3/4', 'height': '5 3/8', 'type': 'drawer', 'material': 'Paint Grade', 'style': 'Solid Wood', 'cabinet': 'N8'},
    {'qty': 1, 'width': '16 5/8', 'height': '5 3/8', 'type': 'drawer', 'material': 'Paint Grade', 'style': 'Solid Wood', 'cabinet': 'N11'},
    {'qty': 1, 'width': '16 5/8', 'height': '5 3/8', 'type': 'drawer', 'material': 'Paint Grade', 'style': 'Solid Wood', 'cabinet': 'N13'},
    {'qty': 1, 'width': '16 5/8', 'height': '5 3/8', 'type': 'drawer', 'material': 'Paint Grade', 'style': 'Solid Wood', 'cabinet': 'N15'},
    {'qty': 1, 'width': '15 3/4', 'height': '5 3/8', 'type': 'drawer', 'material': 'Paint Grade', 'style': 'Solid Wood', 'cabinet': 'N3'},
    {'qty': 2, 'width': '13 3/4', 'height': '5 3/8', 'type': 'drawer', 'material': 'Paint Grade', 'style': 'Solid Wood'},
    
    # False Fronts (treated as drawers with no rails/stiles needed)
    {'qty': 2, 'width': '29 5/8', 'height': '5 3/8', 'type': 'false_front', 'material': 'Paint Grade', 'style': 'False Front'},
    {'qty': 1, 'width': '28 7/8', 'height': '5 3/8', 'type': 'false_front', 'material': 'Paint Grade', 'style': 'False Front'},
    {'qty': 1, 'width': '27 3/4', 'height': '5 3/8', 'type': 'false_front', 'material': 'Paint Grade', 'style': 'False Front'},
    {'qty': 1, 'width': '27 5/8', 'height': '5 3/8', 'type': 'false_front', 'material': 'Paint Grade', 'style': 'False Front'},
    
    # Square Raised Panel (MDF) - Doors
    {'qty': 2, 'width': '13 13/16', 'height': '24 3/4', 'type': 'door', 'material': 'MDF', 'style': 'Square Raised Panel'},
    {'qty': 4, 'width': '14 3/4', 'height': '24 3/4', 'type': 'door', 'material': 'MDF', 'style': 'Square Raised Panel'},
    {'qty': 2, 'width': '14 7/16', 'height': '24 3/4', 'type': 'door', 'material': 'MDF', 'style': 'Square Raised Panel'},
    {'qty': 2, 'width': '14 3/8', 'height': '24 3/4', 'type': 'door', 'material': 'MDF', 'style': 'Square Raised Panel'},
    {'qty': 2, 'width': '13 3/4', 'height': '24 3/4', 'type': 'door', 'material': 'MDF', 'style': 'Square Raised Panel'},
    
    # Square Raised Panel (Paint Grade) - Doors
    {'qty': 4, 'width': '16 1/2', 'height': '41 7/8', 'type': 'door', 'material': 'Paint Grade', 'style': 'Square Raised Panel'},
]

def fraction_to_decimal(fraction_str):
    """Convert fraction string to decimal"""
    if not fraction_str:
        return 0
    
    parts = fraction_str.strip().split()
    total = 0
    
    for part in parts:
        if '/' in part:
            num, den = part.split('/')
            total += float(num) / float(den)
        else:
            total += float(part)
    
    return total

def decimal_to_fraction(decimal):
    """Convert decimal to fraction string"""
    whole = int(decimal)
    remainder = decimal - whole
    
    if remainder < 0.125:
        return str(whole) if whole > 0 else "0"
    elif remainder < 0.375:
        return f"{whole} 1/4" if whole > 0 else "1/4"
    elif remainder < 0.625:
        return f"{whole} 1/2" if whole > 0 else "1/2"
    elif remainder < 0.875:
        return f"{whole} 3/4" if whole > 0 else "3/4"
    else:
        return str(whole + 1)

def calculate_cuts(items):
    """Calculate stiles and rails for all items"""
    cuts = []
    item_num = 1
    cabinet_groups = {}  # Track items by cabinet number
    
    for item in items:
        if item['type'] == 'false_front':
            continue  # False fronts don't need stiles/rails
            
        qty = item['qty']
        width = fraction_to_decimal(item['width'])
        height = fraction_to_decimal(item['height'])
        
        # Use cabinet number if available, otherwise use sequential numbering
        if 'cabinet' in item and item['cabinet']:
            label = item['cabinet']
            # Group items with same cabinet number and size
            group_key = f"{label}_{item['width']}x{item['height']}"
            if group_key not in cabinet_groups:
                cabinet_groups[group_key] = label
        else:
            label = f'#{item_num}'
        
        # Calculate according to formulas
        # Stiles: Height + 1/4" (2 per door/drawer)
        stile_length = height + 0.25
        
        # Rails: Width - 3 3/4" (2 per door/drawer)
        rail_length = width - 3.75
        
        # Add stiles (2 per piece)
        cuts.append({
            'qty': qty * 2,
            'length': stile_length,
            'width': '2 3/8',
            'part': 'STILES',
            'label': label,
            'description': f"{item['qty']} {item['type']} @ {item['width']} x {item['height']} ({item['material']})"
        })
        
        # Add rails (2 per piece)
        cuts.append({
            'qty': qty * 2,
            'length': rail_length,
            'width': '2 3/8',
            'part': 'RAILS',
            'label': label,
            'description': f"{item['qty']} {item['type']} @ {item['width']} x {item['height']} ({item['material']})"
        })
        
        item_num += 1
    
    return cuts

def calculate_totals(items):
    """Calculate total doors, drawers, hinges, etc."""
    total_doors = sum(item['qty'] for item in items if item['type'] == 'door')
    total_drawers = sum(item['qty'] for item in items if item['type'] == 'drawer')
    total_false_fronts = sum(item['qty'] for item in items if item['type'] == 'false_front')
    total_hinges = total_doors * 2  # 2 hinges per door
    
    return {
        'doors': total_doors,
        'drawers': total_drawers,
        'false_fronts': total_false_fronts,
        'total_pieces': total_doors + total_drawers + total_false_fronts,
        'hinges': total_hinges
    }

def generate_shop_report_html(customer_data, items, totals, cuts):
    """Generate shop report HTML"""
    
    # Calculate total linear feet needed for stiles
    total_stile_inches = 0
    for cut in cuts:
        if cut['part'] == 'STILES':
            total_stile_inches += cut['qty'] * cut['length']
    
    total_stile_feet = total_stile_inches / 12
    eight_foot_pieces = int((total_stile_inches / 96) + 0.999)  # Round up
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Shop Report - {customer_data['job']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ border: 2px solid black; padding: 10px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid black; padding: 5px; text-align: left; }}
        th {{ background-color: #f0f0f0; }}
        .checkbox {{ width: 20px; height: 20px; border: 1px solid black; display: inline-block; }}
        @media print {{ body {{ margin: 10px; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SHOP REPORT</h1>
        <p><strong>Date:</strong> {customer_data['date']}</p>
        <p><strong>Order #:</strong> _____________</p>
        <p><strong>Customer Name:</strong> {customer_data['name']}</p>
        <p><strong>Job Name:</strong> {customer_data['job']}</p>
        <p><strong>Start Date:</strong> _____________</p>
        <p><strong>Finish Date:</strong> _____________</p>
    </div>
    
    <div class="section">
        <h2>Door Specifications</h2>
        <p><strong>Wood Species:</strong> MDF & Paint Grade</p>
        <p><strong>Door Sizes:</strong> ☐ Opening Sizes ☐ Finish Sizes</p>
        <p><strong>Bore/Door Prep:</strong> ☐ Yes ☐ No</p>
        <p><strong>Hinge Type:</strong> Standard</p>
        <p><strong>Thickness:</strong> _____________</p>
    </div>
    
    <div class="section">
        <h2>Totals</h2>
        <p><strong>Total Doors:</strong> {totals['doors']}</p>
        <p><strong>Total Drawers:</strong> {totals['drawers']}</p>
        <p><strong>Total False Fronts:</strong> {totals['false_fronts']}</p>
        <p><strong>Total Pieces:</strong> {totals['total_pieces']}</p>
        <p><strong>Total Hinges Needed:</strong> {totals['hinges']}</p>
    </div>
    
    <div class="section">
        <h2>Material Requirements</h2>
        <table>
            <tr>
                <th>Material</th>
                <th>Size</th>
                <th>Quantity</th>
                <th>Notes</th>
            </tr>
            <tr>
                <td>STILES</td>
                <td>2 3/8" × 8'</td>
                <td>{eight_foot_pieces} pieces</td>
                <td>{total_stile_feet:.1f} linear feet total</td>
            </tr>
        </table>
    </div>
    
    <div style="page-break-before: always;"></div>
    
    <div class="section">
        <h2>PICKUP INFORMATION</h2>
        <p><strong>Name:</strong> {customer_data['name']}</p>
        <p><strong>Job #:</strong> {customer_data['job']}</p>
        <p><strong>Contact:</strong> _____________</p>
        <p><strong>Expected Pick-Up Date:</strong> _____________</p>
        <p><strong># of Doors:</strong> {totals['doors']}</p>
        <p><strong># of Drawers:</strong> {totals['drawers']}</p>
        <p><strong># of False Fronts:</strong> {totals['false_fronts']}</p>
        <p><strong>Hinges:</strong> {totals['hinges']}</p>
    </div>
</body>
</html>"""
    
    return html

def generate_cut_list_html(customer_data, cuts):
    """Generate cut list HTML with grouped sizes"""
    
    # Group cuts by length
    grouped_cuts = {}
    for cut in cuts:
        length = cut['length']
        if length not in grouped_cuts:
            grouped_cuts[length] = []
        grouped_cuts[length].append(cut)
    
    # Sort groups by length (descending)
    sorted_lengths = sorted(grouped_cuts.keys(), reverse=True)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cut List - {customer_data['job']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; font-size: 12px; }}
        h1 {{ font-size: 18px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid black; padding: 3px; text-align: left; }}
        th {{ background-color: #f0f0f0; font-weight: bold; }}
        .checkbox {{ width: 15px; height: 15px; border: 1px solid black; display: inline-block; }}
        .group-header {{ background-color: #e8e8e8; font-weight: bold; }}
        @media print {{ body {{ margin: 10px; }} }}
    </style>
</head>
<body>
    <h1>CUT LIST - {customer_data['job']}</h1>
    <p><strong>Date:</strong> {customer_data['date']} | <strong>Customer:</strong> {customer_data['name']}</p>
    
    <table>
        <tr>
            <th width="10%">QTY</th>
            <th width="15%">LENGTH</th>
            <th width="10%">WIDTH</th>
            <th width="10%">PART</th>
            <th width="20%">LABEL</th>
            <th width="45%">DESCRIPTION</th>
        </tr>"""
    
    for length in sorted_lengths:
        length_str = decimal_to_fraction(length)
        cuts_at_length = grouped_cuts[length]
        
        # Combine ALL items at this length (regardless of part type)
        total_qty = sum(cut['qty'] for cut in cuts_at_length)
        all_labels = []
        all_descriptions = []
        parts_set = set()
        widths_set = set()
        
        for cut in cuts_at_length:
            # Format label as "(qty) = #label ☐"
            label_without_hash = cut['label'].replace('#', '')
            all_labels.append(f"({cut['qty']}) = #{label_without_hash} ☐")
            all_descriptions.append(f"{cut['label']}: {cut['description']}")
            parts_set.add(cut['part'])
            widths_set.add(cut['width'])
        
        # Create label string with proper wrapping to keep units together
        # Calculate approximate character width for wrapping
        label_parts = []
        current_line = []
        current_length = 0
        
        for label in all_labels:
            label_length = len(label)
            # If adding this label would exceed ~15 chars and we have items, wrap
            if current_length > 0 and current_length + label_length + 2 > 15:
                label_parts.append(', '.join(current_line))
                current_line = [label]
                current_length = label_length
            else:
                current_line.append(label)
                current_length += label_length + (2 if current_length > 0 else 0)
        
        if current_line:
            label_parts.append(', '.join(current_line))
        
        labels_str = '<br>'.join(label_parts)
        
        # Stack descriptions on separate lines
        descriptions_str = '<br>'.join(all_descriptions)
        
        # Combine parts and widths
        parts_str = '/'.join(sorted(parts_set))
        widths_str = ', '.join(sorted(widths_set))
        
        html += f"""
        <tr>
            <td>{total_qty}</td>
            <td>{length_str}"</td>
            <td>{widths_str}"</td>
            <td>{parts_str}</td>
            <td>{labels_str}</td>
            <td style="font-size: 10px; line-height: 1.2;">{descriptions_str}</td>
        </tr>"""
    
    # Calculate totals
    total_stiles = sum(cut['qty'] for cut in cuts if cut['part'] == 'STILES')
    total_rails = sum(cut['qty'] for cut in cuts if cut['part'] == 'RAILS')
    total_pieces = total_stiles + total_rails
    
    html += f"""
    </table>
    
    <div style="margin-top: 20px;">
        <p><strong>Total Stiles:</strong> {total_stiles} pieces</p>
        <p><strong>Total Rails:</strong> {total_rails} pieces</p>
        <p><strong>Total Pieces:</strong> {total_pieces} pieces</p>
    </div>
</body>
</html>"""
    
    return html

# Main processing
print("Processing Paul Revere (302) Door Order")
print("-" * 50)

# Calculate cuts
cuts = calculate_cuts(items)

# Calculate totals
totals = calculate_totals(items)

print(f"Customer: {customer_data['name']}")
print(f"Job: {customer_data['job']}")
print(f"Total Doors: {totals['doors']}")
print(f"Total Drawers: {totals['drawers']}")
print(f"Total False Fronts: {totals['false_fronts']}")
print(f"Total Pieces: {totals['total_pieces']}")
print(f"Total Hinges Needed: {totals['hinges']}")

# Generate HTML files
shop_report_html = generate_shop_report_html(customer_data, items, totals, cuts)
cut_list_html = generate_cut_list_html(customer_data, cuts)

# Save HTML files
with open('paul_revere_shop_report.html', 'w', encoding='utf-8') as f:
    f.write(shop_report_html)
print("[OK] Generated paul_revere_shop_report.html")

with open('paul_revere_cut_list.html', 'w', encoding='utf-8') as f:
    f.write(cut_list_html)
print("[OK] Generated paul_revere_cut_list.html")

# Convert to PDF using Selenium
try:
    import base64
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import os
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    # Convert shop report to PDF
    file_path = os.path.abspath('paul_revere_shop_report.html')
    driver.get(f"file:///{file_path}")
    
    pdf_data = driver.execute_cdp_cmd('Page.printToPDF', {
        'landscape': False,
        'displayHeaderFooter': False,
        'printBackground': True,
        'scale': 1.0,
        'paperWidth': 8.5,
        'paperHeight': 11,
        'marginTop': 0.5,
        'marginBottom': 0.5,
        'marginLeft': 0.5,
        'marginRight': 0.5
    })
    
    with open('paul_revere_shop_report.pdf', 'wb') as f:
        f.write(base64.b64decode(pdf_data['data']))
    print("[OK] Generated paul_revere_shop_report.pdf")
    
    # Convert cut list to PDF
    file_path = os.path.abspath('paul_revere_cut_list.html')
    driver.get(f"file:///{file_path}")
    
    pdf_data = driver.execute_cdp_cmd('Page.printToPDF', {
        'landscape': False,
        'displayHeaderFooter': False,
        'printBackground': True,
        'scale': 1.0,
        'paperWidth': 8.5,
        'paperHeight': 11,
        'marginTop': 0.5,
        'marginBottom': 0.5,
        'marginLeft': 0.5,
        'marginRight': 0.5
    })
    
    with open('paul_revere_cut_list.pdf', 'wb') as f:
        f.write(base64.b64decode(pdf_data['data']))
    print("[OK] Generated paul_revere_cut_list.pdf")
    
    driver.quit()
    
except Exception as e:
    print(f"[!] PDF conversion failed: {e}")
    print("[!] HTML files have been generated successfully")

print("-" * 50)
print("Processing complete!")