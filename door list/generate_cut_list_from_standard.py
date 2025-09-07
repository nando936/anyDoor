import json
import sys
from collections import defaultdict
import os
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

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

def load_standard_door_list(json_file):
    """Load the standardized door list from JSON"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load door list: {e}")
        return []

def calculate_cuts_from_standard(door_list):
    """Calculate cuts from standardized door list"""
    cuts = []
    
    for cabinet in door_list:
        cabinet_num = cabinet['cabinet']
        items = cabinet['items']
        
        for item in items:
            # Skip false fronts - they don't need rails/stiles
            if item['type'] == 'false_front':
                continue
            
            qty = item['qty']
            width = fraction_to_decimal(item['width'])
            height = fraction_to_decimal(item['height'])
            
            # Calculate according to formulas
            # Stiles: Height + 1/4" (2 per door/drawer)
            stile_length = height + 0.25
            
            # Rails: Width - 3 3/4" (2 per door/drawer)
            rail_length = width - 3.75
            
            # Create description
            type_str = item['type'] if item['type'] != 'drawer' else 'drawer front'
            desc = f"{qty} {type_str} @ {item['width']} x {item['height']}"
            if item['material'] != 'Unknown':
                desc += f" ({item['material']})"
            
            # Add stiles (2 per piece)
            cuts.append({
                'qty': qty * 2,
                'length': stile_length,
                'width': '2 3/8',
                'part': 'STILES',
                'label': cabinet_num,
                'description': desc
            })
            
            # Add rails (2 per piece)
            cuts.append({
                'qty': qty * 2,
                'length': rail_length,
                'width': '2 3/8',
                'part': 'RAILS',
                'label': cabinet_num,
                'description': desc
            })
    
    return cuts

def generate_cut_list_html(cuts, customer_data):
    """Generate HTML for cut list"""
    
    # Group cuts by length
    grouped_cuts = defaultdict(list)
    for cut in cuts:
        grouped_cuts[cut['length']].append(cut)
    
    # Sort groups by length (descending)
    sorted_lengths = sorted(grouped_cuts.keys(), reverse=True)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cut List - {customer_data.get('job', 'Unknown Job')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; font-size: 12px; }}
        h1 {{ font-size: 18px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid black; padding: 3px; text-align: left; }}
        th {{ background-color: #f0f0f0; font-weight: bold; }}
        .checkbox {{ width: 15px; height: 15px; border: 1px solid black; display: inline-block; }}
        @media print {{ body {{ margin: 10px; }} }}
    </style>
</head>
<body>
    <h1>CUT LIST - {customer_data.get('job', 'Unknown Job')}</h1>
    <p><strong>Date:</strong> {customer_data.get('date', '')} | <strong>Customer:</strong> {customer_data.get('name', '')}</p>
    
    <table>
        <tr>
            <th width="10%">QTY</th>
            <th width="15%">LENGTH</th>
            <th width="10%">WIDTH</th>
            <th width="10%">PART</th>
            <th width="25%">LABEL</th>
            <th width="40%">DESCRIPTION</th>
        </tr>"""
    
    for length in sorted_lengths:
        length_str = decimal_to_fraction(length)
        cuts_at_length = grouped_cuts[length]
        
        # Group by cabinet and part type for this length
        cabinet_groups = defaultdict(list)
        for cut in cuts_at_length:
            key = (cut['label'], cut['part'])
            cabinet_groups[key].append(cut)
        
        # Combine all items at this length
        total_qty = sum(cut['qty'] for cut in cuts_at_length)
        all_labels = []
        all_descriptions = []
        parts_set = set()
        widths_set = set()
        
        # Process each cabinet group
        for (cabinet, part), group_cuts in sorted(cabinet_groups.items()):
            group_qty = sum(cut['qty'] for cut in group_cuts)
            # Format label as "(qty) = #cabinet ☐"
            label_str = f"({group_qty}) = {cabinet} ☐"
            all_labels.append(label_str)
            
            # Add unique descriptions
            for cut in group_cuts:
                if cut['description'] not in all_descriptions:
                    all_descriptions.append(f"{cabinet}: {cut['description']}")
            
            for cut in group_cuts:
                parts_set.add(cut['part'])
                widths_set.add(cut['width'])
        
        # Create label string with proper wrapping to keep units together
        label_parts = []
        current_line = []
        current_length = 0
        
        for label in all_labels:
            label_length = len(label)
            # If adding this label would exceed ~20 chars and we have items, wrap
            if current_length > 0 and current_length + label_length + 2 > 20:
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
    
    # Calculate linear feet needed
    total_linear_inches = sum(cut['qty'] * cut['length'] for cut in cuts)
    total_linear_feet = total_linear_inches / 12
    eight_foot_pieces = int((total_linear_inches / 96) + 0.999)
    
    html += f"""
    </table>
    
    <div style="margin-top: 20px;">
        <p><strong>Total Stiles:</strong> {total_stiles} pieces</p>
        <p><strong>Total Rails:</strong> {total_rails} pieces</p>
        <p><strong>Total Pieces:</strong> {total_pieces} pieces</p>
        <p><strong>Total Linear Feet:</strong> {total_linear_feet:.1f} ft</p>
        <p><strong>8-foot Boards Needed:</strong> {eight_foot_pieces} pieces</p>
    </div>
</body>
</html>"""
    
    return html

def html_to_pdf(html_path, pdf_path):
    """Convert HTML file to PDF using Selenium"""
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Load the HTML file
        file_url = f"file:///{os.path.abspath(html_path).replace(os.sep, '/')}"
        driver.get(file_url)
        
        # Generate PDF
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
        
        # Save PDF
        with open(pdf_path, 'wb') as f:
            f.write(base64.b64decode(pdf_data['data']))
        
        driver.quit()
        return True
    except Exception as e:
        print(f"[ERROR] PDF conversion failed: {e}")
        return False

# Main execution
if __name__ == "__main__":
    print("Generating Cut List from Standard Door List")
    print("=" * 60)
    
    # Load standard door list
    door_list = load_standard_door_list("standard_door_list.json")
    
    if not door_list:
        print("[ERROR] No door list found")
        sys.exit(1)
    
    # Customer data (you might want to extract this from the PDF or another source)
    customer_data = {
        'name': 'SUAREZ CARPENTRY, INC.',
        'job': 'paul revere (302)',
        'date': '17 August 2017'
    }
    
    # Calculate cuts
    cuts = calculate_cuts_from_standard(door_list)
    
    # Count totals for summary
    total_doors = 0
    total_drawers = 0
    total_false_fronts = 0
    
    for cabinet in door_list:
        for item in cabinet['items']:
            if item['type'] == 'door':
                total_doors += item['qty']
            elif item['type'] == 'drawer':
                total_drawers += item['qty']
            elif item['type'] == 'false_front':
                total_false_fronts += item['qty']
    
    print(f"Total Doors: {total_doors}")
    print(f"Total Drawer Fronts: {total_drawers}")
    print(f"Total False Fronts: {total_false_fronts}")
    print(f"Total Pieces: {total_doors + total_drawers + total_false_fronts}")
    print("-" * 60)
    
    # Generate HTML
    html_content = generate_cut_list_html(cuts, customer_data)
    
    # Save HTML
    html_file = "302_cut_list.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[OK] Generated {html_file}")
    
    # Convert to PDF
    pdf_file = "302_cut_list.pdf"
    if html_to_pdf(html_file, pdf_file):
        print(f"[OK] Generated {pdf_file}")
    
    print("\nProcessing complete!")