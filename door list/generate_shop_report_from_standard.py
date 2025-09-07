import json
import sys
import os
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def load_standard_door_list(json_file):
    """Load the standardized door list from JSON"""
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

def calculate_materials_needed(door_list):
    """Calculate total materials needed"""
    total_linear_inches = 0
    
    for cabinet in door_list:
        for item in cabinet['items']:
            if item['type'] != 'false_front':
                qty = item['qty']
                width = fraction_to_decimal(item['width'])
                height = fraction_to_decimal(item['height'])
                
                # Each piece needs 2 stiles and 2 rails
                stile_length = height + 0.25
                rail_length = width - 3.75
                
                # Add to total (2 stiles + 2 rails per piece)
                total_linear_inches += qty * 2 * stile_length
                total_linear_inches += qty * 2 * rail_length
    
    total_linear_feet = total_linear_inches / 12
    eight_foot_pieces = int((total_linear_inches / 96) + 0.999)
    
    return total_linear_feet, eight_foot_pieces

def generate_shop_report_html(door_list, customer_data):
    """Generate HTML for shop report"""
    
    # Count totals
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
    
    total_hinges = total_doors * 2  # 2 hinges per door
    total_linear_feet, eight_foot_pieces = calculate_materials_needed(door_list)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Shop Report - {customer_data.get('job', 'Unknown Job')}</title>
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
            <div><strong>Date:</strong> {customer_data.get('date', '')}</div>
            <div><strong>Order #:</strong> _____________</div>
            <div><strong>Customer Name:</strong> {customer_data.get('name', '')}</div>
            <div><strong>Job Name:</strong> {customer_data.get('job', '')}</div>
            <div><strong>Start Date:</strong> _____________</div>
            <div><strong>Finish Date:</strong> _____________</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Door Specifications</h2>
        <table>
            <tr>
                <td width="50%"><strong>Wood Species:</strong> {customer_data.get('wood_species', 'MDF & Paint Grade')}</td>
                <td width="50%"><strong>Door Sizes:</strong> ☐ Opening Sizes ☐ Finish Sizes</td>
            </tr>
            <tr>
                <td><strong>Outside Edge:</strong> {customer_data.get('outside_edge', 'Standard')}</td>
                <td><strong>Inside/Sticky:</strong> {customer_data.get('inside_edge', 'Standard')}</td>
            </tr>
            <tr>
                <td><strong>Panel Cut:</strong> {customer_data.get('panel_cut', '3/8" Plywood')}</td>
                <td><strong>Bore/Door Prep:</strong> ☐ Yes ☐ No</td>
            </tr>
            <tr>
                <td><strong>Hinge Type:</strong> {customer_data.get('hinge_type', 'Standard')}</td>
                <td><strong>Thickness:</strong> _____________</td>
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
            <tr>
                <td>Drawer Fronts</td>
                <td>{total_drawers}</td>
                <td>No hinges required</td>
            </tr>
            <tr>
                <td>False Fronts</td>
                <td>{total_false_fronts}</td>
                <td>No rails/stiles required</td>
            </tr>
            <tr style="font-weight: bold;">
                <td>TOTAL PIECES</td>
                <td>{total_doors + total_drawers + total_false_fronts}</td>
                <td>&nbsp;</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Materials Required</h2>
        <table>
            <tr>
                <th>Material</th>
                <th>Quantity</th>
                <th>Specification</th>
            </tr>
            <tr>
                <td>Stile/Rail Stock</td>
                <td>{eight_foot_pieces} pieces</td>
                <td>2 3/8" × 8' ({total_linear_feet:.1f} linear feet total)</td>
            </tr>
            <tr>
                <td>Hinges</td>
                <td>{total_hinges} pieces</td>
                <td>{customer_data.get('hinge_type', 'Standard')}</td>
            </tr>
        </table>
    </div>
    
    <div class="page-break"></div>
    
    <div class="section">
        <h2>Cabinet Breakdown</h2>
        <table>
            <tr>
                <th>Cabinet #</th>
                <th>Items</th>
                <th>Dimensions</th>
            </tr>"""
    
    # Add cabinet details
    for cabinet in door_list:
        cabinet_num = cabinet['cabinet']
        items = cabinet['items']
        
        items_str = ""
        dims_str = ""
        
        for item in items:
            qty = item['qty']
            type_str = 'D' if item['type'] == 'door' else 'DF' if item['type'] == 'drawer' else 'FF'
            if items_str:
                items_str += "<br>"
                dims_str += "<br>"
            items_str += f"{qty} {type_str}"
            dims_str += f"{item['width']} × {item['height']}"
        
        html += f"""
            <tr>
                <td>{cabinet_num}</td>
                <td>{items_str}</td>
                <td>{dims_str}</td>
            </tr>"""
    
    html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Pickup Information</h2>
        <table>
            <tr>
                <td width="50%"><strong>Name:</strong> {customer_name}</td>
                <td width="50%"><strong>Job #:</strong> {job_name}</td>
            </tr>
            <tr>
                <td><strong>Contact:</strong> {contact}</td>
                <td><strong>Expected Pickup Date:</strong> _____________</td>
            </tr>
            <tr>
                <td><strong># of Doors:</strong> {total_pieces}</td>
                <td><strong>Hinges:</strong> {total_hinges}</td>
            </tr>
        </table>
    </div>
</body>
</html>""".format(
        customer_name=customer_data.get('name', ''),
        job_name=customer_data.get('job', ''),
        contact=customer_data.get('phone', '') or customer_data.get('email', ''),
        total_pieces=total_doors + total_drawers + total_false_fronts,
        total_hinges=total_hinges
    )
    
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
    print("Generating Shop Report from Standard Door List")
    print("=" * 60)
    
    # Load standard door list
    door_list = load_standard_door_list("standard_door_list.json")
    
    if not door_list:
        print("[ERROR] No door list found")
        sys.exit(1)
    
    # Customer data
    customer_data = {
        'name': 'SUAREZ CARPENTRY, INC.',
        'job': 'paul revere (302)',
        'date': '17 August 2017',
        'wood_species': 'MDF & Paint Grade',
        'outside_edge': 'Special',
        'inside_edge': 'Bevel',
        'panel_cut': '3/8" Plywood (Flat Panel ONLY)',
        'hinge_type': 'Standard'
    }
    
    # Generate HTML
    html_content = generate_shop_report_html(door_list, customer_data)
    
    # Save HTML
    html_file = "302_shop_report.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[OK] Generated {html_file}")
    
    # Convert to PDF
    pdf_file = "302_shop_report.pdf"
    if html_to_pdf(html_file, pdf_file):
        print(f"[OK] Generated {pdf_file}")
    
    print("\nShop Report complete!")