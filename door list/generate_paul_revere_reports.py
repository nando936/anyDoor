import json
import sys
import os
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def load_door_list(json_file):
    """Load the door list from JSON"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load door list: {e}")
        return []

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
    
    fractions = [
        (1/16, "1/16"), (1/8, "1/8"), (3/16, "3/16"), (1/4, "1/4"),
        (5/16, "5/16"), (3/8, "3/8"), (7/16, "7/16"), (1/2, "1/2"),
        (9/16, "9/16"), (5/8, "5/8"), (11/16, "11/16"), (3/4, "3/4"),
        (13/16, "13/16"), (7/8, "7/8"), (15/16, "15/16")
    ]
    
    if remainder < 0.03125:
        return f"{whole}"
    
    closest = min(fractions, key=lambda x: abs(x[0] - remainder))
    
    if whole > 0:
        return f"{whole} {closest[1]}"
    else:
        return closest[1]

def generate_shop_report_html(door_list):
    """Generate HTML for shop report in the proper format"""
    
    # Count totals
    total_doors = 0
    
    for entry in door_list:
        for item in entry['items']:
            if item['type'] == 'door':
                total_doors += item['qty']
    
    total_hinges = total_doors * 2  # 2 hinges per door
    
    # Build cabinet breakdown table
    cabinet_rows = ""
    for entry in door_list:
        for item in entry['items']:
            qty = item['qty']
            item_type = "D" if item['type'] == 'door' else ""
            cabinet_rows += f"""
            <tr>
                <td>Line #{entry['line_item']}</td>
                <td>{qty} {item_type}</td>
                <td>{item['width']} × {item['height']}</td>
            </tr>"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Shop Report - Paul Revere (302)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ border: 2px solid black; padding: 10px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid black; padding: 5px; text-align: left; }}
        th {{ background-color: #f0f0f0; }}
        .checkbox {{ width: 20px; height: 20px; border: 1px solid black; display: inline-block; }}
        .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        @media print {{ 
            body {{ margin: 10px; }}
            .page-break {{ page-break-after: always; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SHOP REPORT</h1>
        <div class="info-grid">
            <div><strong>Date:</strong> 1/7/2025</div>
            <div><strong>Order #:</strong> 302</div>
            <div><strong>Customer Name:</strong> Paul Revere</div>
            <div><strong>Job Name:</strong> Paul Revere Kitchen Remodel (302)</div>
            <div><strong>Start Date:</strong> _____________</div>
            <div><strong>Finish Date:</strong> _____________</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Door Specifications</h2>
        <table>
            <tr>
                <td width="50%"><strong>Wood Species:</strong> White Oak</td>
                <td width="50%"><strong>Door Sizes:</strong> ☐ Opening Sizes ☑ Finish Sizes</td>
            </tr>
            <tr>
                <td><strong>Outside Edge:</strong> Standard for #302</td>
                <td><strong>Inside/Sticky:</strong> Standard for #302</td>
            </tr>
            <tr>
                <td><strong>Panel Cut:</strong> 3/8" Plywood (Flat Panel ONLY)</td>
                <td><strong>Bore/Door Prep:</strong> ☑ Yes ☐ No</td>
            </tr>
            <tr>
                <td><strong>Hinge Type:</strong> Blum Soft Close Frameless 1/2"OL</td>
                <td><strong>Door Style:</strong> #302</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Production Summary</h2>
        <table>
            <tr>
                <th>Item Type</th>
                <th>Quantity</th>
                <th>Notes</th>
            </tr>
            <tr>
                <td>Doors</td>
                <td>{total_doors}</td>
                <td>Requires {total_hinges} hinges (2 per door)</td>
            </tr>
            <tr style="font-weight: bold;">
                <td>TOTAL PIECES</td>
                <td>{total_doors}</td>
                <td>&nbsp;</td>
            </tr>
        </table>
    </div>
    
    <div class="page-break"></div>
    
    <div class="section">
        <h2>Line Item Breakdown (Using Line Numbers as Cabinet References)</h2>
        <table>
            <tr>
                <th>Line Item #</th>
                <th>Items</th>
                <th>Dimensions</th>
            </tr>
            {cabinet_rows}
        </table>
    </div>
    
    <div class="section">
        <h2>Pickup Information</h2>
        <table>
            <tr>
                <td width="50%"><strong>Name:</strong> Paul Revere</td>
                <td width="50%"><strong>Job #:</strong> 302</td>
            </tr>
            <tr>
                <td><strong>Contact:</strong> (281) 555-1775</td>
                <td><strong>Expected Pickup Date:</strong> _____________</td>
            </tr>
            <tr>
                <td><strong># of Doors:</strong> {total_doors}</td>
                <td><strong>Hinges:</strong> {total_hinges}</td>
            </tr>
        </table>
    </div>
</body>
</html>"""
    
    return html

def generate_cut_list_html(door_list):
    """Generate HTML for cut list in the proper format"""
    
    # Group cuts by length for optimization
    cuts = []
    
    for entry in door_list:
        for item in entry['items']:
            if item['type'] == 'door':
                width = fraction_to_decimal(item['width'])
                height = fraction_to_decimal(item['height'])
                qty = item['qty']
                
                # Calculate stile lengths (height - 1/8")
                stile_length = height - 0.125
                
                # Calculate rail lengths (width - 4 1/8")
                rail_length = width - 4.125
                
                # Add stiles (2 per door)
                cuts.append({
                    'qty': qty * 2,
                    'length': stile_length,
                    'width': '2 3/8"',
                    'part': 'STILES',
                    'label': f"({qty * 2}) = Line #{entry['line_item']}",
                    'desc': f"Line #{entry['line_item']}: {qty} door @ {item['width']} x {item['height']}",
                    'notes': item.get('notes', '')
                })
                
                # Add rails (2 per door)
                cuts.append({
                    'qty': qty * 2,
                    'length': rail_length,
                    'width': '2 3/8"',
                    'part': 'RAILS',
                    'label': f"({qty * 2}) = Line #{entry['line_item']}",
                    'desc': f"Line #{entry['line_item']}: {qty} door @ {item['width']} x {item['height']}",
                    'notes': item.get('notes', '')
                })
    
    # Sort cuts by length (longest first)
    cuts.sort(key=lambda x: x['length'], reverse=True)
    
    # Combine similar lengths
    combined_cuts = []
    for cut in cuts:
        # Check if we can combine with existing cut
        found = False
        for combined in combined_cuts:
            if abs(combined['length'] - cut['length']) < 0.0625 and combined['part'] == cut['part']:
                combined['qty'] += cut['qty']
                combined['label'] += f"<br>{cut['label']}"
                combined['desc'] += f"<br>{cut['desc']}"
                if cut['notes']:
                    combined['notes'] += f"<br>{cut['notes']}"
                found = True
                break
        
        if not found:
            combined_cuts.append(cut.copy())
    
    # Build table rows
    table_rows = ""
    total_stiles = 0
    total_rails = 0
    
    for cut in combined_cuts:
        length_str = decimal_to_fraction(cut['length']) + '"'
        table_rows += f"""
        <tr>
            <td>{cut['qty']}</td>
            <td>{length_str}</td>
            <td>{cut['width']}</td>
            <td>{cut['part']}</td>
            <td>{cut['label']} ☐</td>
            <td style="font-size: 10px; line-height: 1.2;">{cut['desc']}<br>{cut['notes']}</td>
        </tr>"""
        
        if cut['part'] == 'STILES':
            total_stiles += cut['qty']
        else:
            total_rails += cut['qty']
    
    total_pieces = total_stiles + total_rails
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cut List - Paul Revere (302)</title>
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
    <h1>CUT LIST - Paul Revere Kitchen Remodel (302)</h1>
    <p><strong>Date:</strong> 1/7/2025 | <strong>Customer:</strong> Paul Revere | <strong>Wood:</strong> White Oak</p>
    
    <table>
        <tr>
            <th width="10%">QTY</th>
            <th width="15%">LENGTH</th>
            <th width="10%">WIDTH</th>
            <th width="10%">PART</th>
            <th width="25%">LABEL</th>
            <th width="40%">DESCRIPTION</th>
        </tr>
        {table_rows}
    </table>
    
    <div style="margin-top: 20px;">
        <p><strong>Total Stiles:</strong> {total_stiles} pieces</p>
        <p><strong>Total Rails:</strong> {total_rails} pieces</p>
        <p><strong>Total Pieces:</strong> {total_pieces} pieces</p>
    </div>
</body>
</html>"""
    
    return html

def html_to_pdf(html_content, pdf_path):
    """Convert HTML to PDF using Selenium"""
    try:
        # Save HTML to temp file
        temp_html = "temp_report.html"
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Load the HTML file
        file_url = f"file:///{os.path.abspath(temp_html).replace(os.sep, '/')}"
        driver.get(file_url)
        time.sleep(1)
        
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
        
        # Clean up temp file
        if os.path.exists(temp_html):
            os.remove(temp_html)
        
        return True
    except Exception as e:
        print(f"[ERROR] PDF conversion failed: {e}")
        return False

# Main execution
if __name__ == "__main__":
    print("Generating Paul Revere Reports (Job 302)")
    print("=" * 60)
    
    # Load Paul Revere door list
    door_list = load_door_list("paul_revere_302_door_list.json")
    
    if not door_list:
        print("[ERROR] No door list found")
        sys.exit(1)
    
    # Generate Shop Report
    print("\nGenerating Shop Report...")
    shop_html = generate_shop_report_html(door_list)
    
    # Save HTML
    shop_html_file = "paul_revere_302_shop_report.html"
    with open(shop_html_file, 'w', encoding='utf-8') as f:
        f.write(shop_html)
    print(f"[OK] Generated {shop_html_file}")
    
    # Convert to PDF
    shop_pdf_file = "paul_revere_302_shop_report.pdf"
    if html_to_pdf(shop_html, shop_pdf_file):
        print(f"[OK] Generated {shop_pdf_file}")
    
    # Generate Cut List
    print("\nGenerating Cut List...")
    cut_html = generate_cut_list_html(door_list)
    
    # Save HTML
    cut_html_file = "paul_revere_302_cut_list.html"
    with open(cut_html_file, 'w', encoding='utf-8') as f:
        f.write(cut_html)
    print(f"[OK] Generated {cut_html_file}")
    
    # Convert to PDF
    cut_pdf_file = "paul_revere_302_cut_list.pdf"
    if html_to_pdf(cut_html, cut_pdf_file):
        print(f"[OK] Generated {cut_pdf_file}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("Generated files:")
    print(f"  - {shop_html_file}")
    print(f"  - {shop_pdf_file}")
    print(f"  - {cut_html_file}")
    print(f"  - {cut_pdf_file}")