import sys
import json
import os
from datetime import datetime
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Set UTF-8 encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def load_door_list(json_file):
    """Load the door list from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_shop_report(door_list, customer_info):
    """Generate HTML shop report from door list"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Shop Report - Paul Revere Job 302</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        h1 {{ color: #333; border-bottom: 3px solid #333; padding-bottom: 10px; }}
        .info-section {{ background: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
        .info-item {{ padding: 5px; }}
        .info-label {{ font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #333; color: white; padding: 10px; text-align: left; }}
        td {{ border: 1px solid #ddd; padding: 8px; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .summary {{ background: #e8f4f8; padding: 15px; margin-top: 20px; border-radius: 5px; }}
        .total-row {{ font-weight: bold; background: #ddd !important; }}
        @media print {{ body {{ padding: 0; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SHOP REPORT</h1>
        <h2>The Raised Panel Door Factory, Inc.</h2>
    </div>
    
    <div class="info-section">
        <div class="info-grid">
            <div class="info-item">
                <span class="info-label">Customer:</span> {customer_info['name']}
            </div>
            <div class="info-item">
                <span class="info-label">Job Name:</span> {customer_info['job_name']}
            </div>
            <div class="info-item">
                <span class="info-label">Date:</span> {customer_info['date']}
            </div>
            <div class="info-item">
                <span class="info-label">Door Style:</span> #302
            </div>
            <div class="info-item">
                <span class="info-label">Wood Species:</span> White Oak
            </div>
            <div class="info-item">
                <span class="info-label">Phone:</span> {customer_info['phone']}
            </div>
        </div>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Line #</th>
                <th>Cabinet #</th>
                <th>Qty</th>
                <th>Width</th>
                <th>Height</th>
                <th>Type</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>"""
    
    total_doors = 0
    for entry in door_list:
        for item in entry['items']:
            html += f"""
            <tr>
                <td>{entry['line_item']}</td>
                <td>{entry['cabinet']}</td>
                <td>{item['qty']}</td>
                <td>{item['width']}</td>
                <td>{item['height']}</td>
                <td>{item['type'].title()}</td>
                <td>{item['notes']}</td>
            </tr>"""
            total_doors += item['qty']
    
    html += f"""
            <tr class="total-row">
                <td colspan="2">TOTAL</td>
                <td>{total_doors}</td>
                <td colspan="4">Total Doors</td>
            </tr>
        </tbody>
    </table>
    
    <div class="summary">
        <h3>Order Summary</h3>
        <ul>
            <li>Total Line Items: {len(door_list)}</li>
            <li>Total Doors: {total_doors}</li>
            <li>Material: White Oak</li>
            <li>Door Style: #302</li>
            <li>Overlay: 1/2" Overlay</li>
            <li>Hinge: Blum Soft Close Frameless</li>
        </ul>
    </div>
    
    <div style="margin-top: 40px; text-align: center; color: #666;">
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
    
    return html

def generate_cut_list(door_list):
    """Generate HTML cut list from door list"""
    
    # Calculate stiles and rails for each door
    cut_items = []
    for entry in door_list:
        for item in entry['items']:
            if item['type'] == 'door':
                # Parse dimensions
                width_str = item['width'].replace('"', '').strip()
                height_str = item['height'].replace('"', '').strip()
                
                # Convert to decimal
                width = parse_fraction(width_str)
                height = parse_fraction(height_str)
                
                # Calculate stile and rail dimensions
                # Stiles = height - 1/8" for clearance
                stile_length = height - 0.125
                # Rails = width - 4" (for 2" stiles on each side) - 1/8" clearance
                rail_length = width - 4.125
                
                cut_items.append({
                    'line': entry['line_item'],
                    'cabinet': entry['cabinet'],
                    'qty': item['qty'] * 2,  # 2 stiles per door
                    'part': 'Stile',
                    'width': '2"',
                    'length': format_dimension(stile_length),
                    'notes': item['notes']
                })
                
                cut_items.append({
                    'line': entry['line_item'],
                    'cabinet': entry['cabinet'],
                    'qty': item['qty'] * 2,  # 2 rails per door
                    'part': 'Rail',
                    'width': '2"',
                    'length': format_dimension(rail_length),
                    'notes': item['notes']
                })
                
                # Panel dimensions (width - 2.5", height - 2.5")
                panel_width = width - 2.5
                panel_height = height - 2.5
                
                cut_items.append({
                    'line': entry['line_item'],
                    'cabinet': entry['cabinet'],
                    'qty': item['qty'],
                    'part': 'Panel',
                    'width': format_dimension(panel_width),
                    'length': format_dimension(panel_height),
                    'notes': '3/8" Plywood'
                })
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Cut List - Paul Revere Job 302</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        h1 {{ color: #333; border-bottom: 3px solid #333; padding-bottom: 10px; }}
        .info-section {{ background: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #333; color: white; padding: 10px; text-align: left; }}
        td {{ border: 1px solid #ddd; padding: 8px; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .part-stile {{ background: #e8f4e8; }}
        .part-rail {{ background: #f4e8e8; }}
        .part-panel {{ background: #e8e8f4; }}
        @media print {{ body {{ padding: 0; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CUT LIST</h1>
        <h2>Paul Revere Kitchen Remodel - Job 302</h2>
    </div>
    
    <div class="info-section">
        <p><strong>Material:</strong> White Oak</p>
        <p><strong>Door Style:</strong> #302</p>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Line #</th>
                <th>Cabinet</th>
                <th>Part</th>
                <th>Qty</th>
                <th>Width</th>
                <th>Length</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>"""
    
    for item in cut_items:
        part_class = f"part-{item['part'].lower()}"
        html += f"""
            <tr class="{part_class}">
                <td>{item['line']}</td>
                <td>{item['cabinet']}</td>
                <td>{item['part']}</td>
                <td>{item['qty']}</td>
                <td>{item['width']}</td>
                <td>{item['length']}</td>
                <td>{item['notes']}</td>
            </tr>"""
    
    html += """
        </tbody>
    </table>
    
    <div style="margin-top: 40px; text-align: center; color: #666;">
        <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
</body>
</html>"""
    
    return html

def parse_fraction(dim_str):
    """Parse a dimension string with fractions to decimal"""
    # Remove any quotes
    dim_str = dim_str.replace('"', '').strip()
    
    # Split into whole and fraction parts
    parts = dim_str.split(' ')
    
    if len(parts) == 1:
        # Check if it's a fraction
        if '/' in parts[0]:
            nums = parts[0].split('/')
            return float(nums[0]) / float(nums[1])
        else:
            return float(parts[0])
    else:
        # Whole number plus fraction
        whole = float(parts[0])
        if '/' in parts[1]:
            nums = parts[1].split('/')
            fraction = float(nums[0]) / float(nums[1])
            return whole + fraction
        else:
            return whole

def format_dimension(decimal):
    """Format a decimal dimension back to fraction string"""
    whole = int(decimal)
    remainder = decimal - whole
    
    # Find closest fraction
    fractions = [
        (1/16, "1/16"),
        (1/8, "1/8"),
        (3/16, "3/16"),
        (1/4, "1/4"),
        (5/16, "5/16"),
        (3/8, "3/8"),
        (7/16, "7/16"),
        (1/2, "1/2"),
        (9/16, "9/16"),
        (5/8, "5/8"),
        (11/16, "11/16"),
        (3/4, "3/4"),
        (13/16, "13/16"),
        (7/8, "7/8"),
        (15/16, "15/16")
    ]
    
    if remainder < 0.03125:  # Less than 1/32
        return f"{whole}\""
    
    # Find closest fraction
    closest = min(fractions, key=lambda x: abs(x[0] - remainder))
    
    if whole > 0:
        return f"{whole} {closest[1]}\""
    else:
        return f"{closest[1]}\""

def html_to_pdf(html_content, output_file):
    """Convert HTML to PDF using Selenium"""
    # Save HTML to temp file
    temp_html = "temp_report.html"
    with open(temp_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Load HTML
        file_url = f"file:///{os.path.abspath(temp_html).replace(os.sep, '/')}"
        driver.get(file_url)
        time.sleep(1)
        
        # Print to PDF
        print_options = {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'preferCSSPageSize': True,
            'paperWidth': 8.5,
            'paperHeight': 11,
            'marginTop': 0.5,
            'marginBottom': 0.5,
            'marginLeft': 0.5,
            'marginRight': 0.5
        }
        
        result = driver.execute_cdp_cmd('Page.printToPDF', print_options)
        pdf_data = base64.b64decode(result['data'])
        
        with open(output_file, 'wb') as f:
            f.write(pdf_data)
        
        print(f"[OK] Generated {output_file}")
        
    finally:
        driver.quit()
        if os.path.exists(temp_html):
            os.remove(temp_html)

def main():
    # Load door list
    print("Processing Paul Revere Order 302...")
    print("-" * 40)
    
    door_list = load_door_list("paul_revere_302_door_list.json")
    
    # Customer info
    customer_info = {
        'name': 'Paul Revere',
        'job_name': 'Paul Revere Kitchen Remodel',
        'date': '1/7/2025',
        'phone': '(281) 555-1775',
        'email': 'prevere@colonialcabinets.com'
    }
    
    # Generate shop report
    print("Generating shop report...")
    shop_html = generate_shop_report(door_list, customer_info)
    with open("paul_revere_shop_report.html", 'w', encoding='utf-8') as f:
        f.write(shop_html)
    print("[OK] Created paul_revere_shop_report.html")
    
    html_to_pdf(shop_html, "paul_revere_shop_report.pdf")
    
    # Generate cut list
    print("\nGenerating cut list...")
    cut_html = generate_cut_list(door_list)
    with open("paul_revere_cut_list.html", 'w', encoding='utf-8') as f:
        f.write(cut_html)
    print("[OK] Created paul_revere_cut_list.html")
    
    html_to_pdf(cut_html, "paul_revere_cut_list.pdf")
    
    print("\n" + "=" * 40)
    print("Processing complete!")
    print("Generated files:")
    print("  - paul_revere_shop_report.html")
    print("  - paul_revere_shop_report.pdf")
    print("  - paul_revere_cut_list.html")
    print("  - paul_revere_cut_list.pdf")

if __name__ == "__main__":
    main()