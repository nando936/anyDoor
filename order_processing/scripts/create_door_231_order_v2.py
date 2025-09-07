import json
import sys
from datetime import datetime
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

def extract_doors_only(door_list):
    """Extract only door items from the list"""
    doors = []
    for cabinet in door_list:
        for item in cabinet['items']:
            if item['type'] == 'door':
                doors.append({
                    'cabinet': cabinet['cabinet'],
                    'qty': item['qty'],
                    'width': item['width'],
                    'height': item['height']
                })
    return doors

def generate_order_form_html(doors):
    """Generate HTML order form with door #231 matching original layout"""
    
    # Customer info for Paul Revere
    customer_info = {
        'name': 'Paul Revere',
        'address': '1776 Liberty Lane, Boston TX 77385',
        'phone': '(281) 555-1775',
        'email': 'prevere@colonialcabinets.com',
        'job_name': 'Revere Kitchen Remodel',
        'date': datetime.now().strftime('%m/%d/%Y')
    }
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Door Order Form - Style 231</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: Arial, sans-serif; 
            font-size: 10pt;
            padding: 20px;
            max-width: 850px;
            margin: 0 auto;
        }}
        .header-section {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }}
        .left-header {{
            flex: 1;
        }}
        .logo-section {{
            width: 200px;
            text-align: center;
            border: 2px solid #000;
            padding: 10px;
            margin-left: 20px;
        }}
        h1 {{ 
            font-size: 16pt;
            margin-bottom: 5px;
        }}
        .date-job {{
            margin-bottom: 15px;
        }}
        .customer-section {{
            margin-bottom: 15px;
        }}
        .customer-section h3 {{
            font-size: 11pt;
            margin-bottom: 5px;
            text-decoration: underline;
        }}
        .field-row {{
            margin-bottom: 3px;
        }}
        .field-label {{
            display: inline-block;
            width: 70px;
            font-weight: bold;
        }}
        .door-info-section {{
            border: 2px solid #000;
            padding: 10px;
            margin-bottom: 15px;
        }}
        .door-info-title {{
            text-align: center;
            font-weight: bold;
            font-size: 11pt;
            margin-bottom: 10px;
            text-decoration: underline;
        }}
        .door-specs-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .door-specs-table td {{
            padding: 4px;
            border: 1px solid #000;
            vertical-align: middle;
        }}
        .spec-label {{
            font-weight: bold;
            background-color: #f0f0f0;
            width: 120px;
        }}
        .checkbox-container {{
            display: inline-block;
            margin-right: 10px;
        }}
        .checkbox {{
            width: 12px;
            height: 12px;
            border: 1px solid #000;
            display: inline-block;
            margin-right: 3px;
            position: relative;
            vertical-align: middle;
        }}
        .checked::after {{
            content: "X";
            position: absolute;
            top: -2px;
            left: 1px;
            font-weight: bold;
            font-size: 10px;
        }}
        .order-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }}
        .order-table th {{
            background-color: #f0f0f0;
            border: 1px solid #000;
            padding: 5px;
            font-weight: bold;
            text-align: center;
        }}
        .order-table td {{
            border: 1px solid #000;
            padding: 5px;
            text-align: center;
            height: 25px;
        }}
        .special-notes {{
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #000;
        }}
        .company-footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 9pt;
        }}
        @media print {{
            body {{ 
                padding: 10px;
                font-size: 9pt;
            }}
        }}
    </style>
</head>
<body>
    <div class="header-section">
        <div class="left-header">
            <h1>ORDER FORM</h1>
            <div class="date-job">
                <div><strong>Date:</strong> {customer_info['date']}</div>
                <div><strong>Job Name:</strong> {customer_info['job_name']}</div>
            </div>
            
            <div class="customer-section">
                <h3>Customer/Job Info:</h3>
                <div class="field-row">
                    <span class="field-label">Name:</span> {customer_info['name']}
                </div>
                <div class="field-row">
                    <span class="field-label">Address:</span> {customer_info['address']}
                </div>
                <div class="field-row">
                    <span class="field-label">Phone #:</span> {customer_info['phone']}
                </div>
                <div class="field-row">
                    <span class="field-label">Email:</span> {customer_info['email']}
                </div>
            </div>
        </div>
        
        <div class="logo-section">
            <img src="https://theraisedpaneldoor.com/zirw/932/i/u/10227018/i/menu/title.png" 
                 alt="The Raised Panel Door Factory" 
                 style="width: 120px; height: auto; margin-bottom: 5px;">
            <div style="font-weight: bold; font-size: 10pt;">THE RAISED PANEL</div>
            <div style="font-weight: bold; font-size: 10pt;">DOOR FACTORY, INC.</div>
            <div style="margin-top: 10px; font-size: 8pt;">
                209 RIGGS ST.<br>
                CONROE TX 77301<br><br>
                MAIN: (936) 672-4235<br>
                <span style="font-size: 7pt;">WWW.THERAISEDPANELDOOR.COM<br>
                fernando@theraisedpaneldoor.com</span>
            </div>
        </div>
    </div>
    
    <div class="door-info-section">
        <div class="door-info-title">Door Order Info:</div>
        <table class="door-specs-table">
            <tr>
                <td class="spec-label">Door Sizes:<br>(Check one)</td>
                <td>
                    <span class="checkbox-container">
                        <span class="checkbox"></span> Finish Sizes
                    </span>
                    <span style="margin: 0 10px;">or</span>
                    <span class="checkbox-container">
                        <span class="checkbox checked"></span> Opening Sizes
                    </span>
                </td>
                <td class="spec-label">Overlay Type:</td>
                <td>1/2" Overlay</td>
            </tr>
            <tr>
                <td class="spec-label">Bore/Door Prep:</td>
                <td>
                    <span class="checkbox-container">
                        <span class="checkbox"></span> Yes
                    </span>
                    <span class="checkbox-container">
                        <span class="checkbox checked"></span> No
                    </span>
                </td>
                <td class="spec-label">Hinge Type:</td>
                <td>Blum Soft Close Frameless 1/2"OL</td>
            </tr>
            <tr>
                <td class="spec-label">Wood Species:</td>
                <td>White Oak</td>
                <td class="spec-label">Door #:</td>
                <td style="font-weight: bold; color: red; font-size: 12pt;">231</td>
            </tr>
            <tr>
                <td class="spec-label">Outside Edge:</td>
                <td>Standard for #231</td>
                <td class="spec-label">Sticking (Inside):</td>
                <td>Standard for #231</td>
            </tr>
            <tr>
                <td class="spec-label">Panel Cut:</td>
                <td colspan="3">3/8" Plywood (Flat Panel ONLY)</td>
            </tr>
            <tr>
                <td class="spec-label">Drawer/Front type:</td>
                <td colspan="3">5 piece</td>
            </tr>
        </table>
    </div>
    
    <table class="order-table">
        <thead>
            <tr>
                <th width="5%">#</th>
                <th width="10%">QTY</th>
                <th width="15%">Width</th>
                <th width="15%">Height</th>
                <th width="25%">Doors/Drawer Details</th>
                <th width="30%">Additional Notes</th>
            </tr>
        </thead>
        <tbody>"""
    
    # Add door items
    item_num = 1
    total_doors = 0
    
    # Sort doors by cabinet number
    sorted_doors = sorted(doors, key=lambda x: (x['cabinet'].replace('#', '').zfill(3)))
    
    for door in sorted_doors:
        # Special case for cabinet #4 - make it qty 1 for trash drawer
        qty = 1 if door['cabinet'] == '#4' else door['qty']
        
        # Add note for cabinet #4
        notes = "No hinge boring - trash drawer" if door['cabinet'] == '#4' else f"Cabinet {door['cabinet']}"
        
        html += f"""
            <tr>
                <td>{item_num}</td>
                <td>{qty}</td>
                <td>{door['width']}</td>
                <td>{door['height']}</td>
                <td>doors</td>
                <td style="font-size: 9pt;">{notes}</td>
            </tr>"""
        item_num += 1
        total_doors += qty
    
    # Add empty rows to match original form (up to line 20 on first page)
    for i in range(item_num, 21):
        html += f"""
            <tr>
                <td>{i}</td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
            </tr>"""
    
    html += f"""
        </tbody>
    </table>
    
    <div style="margin-top: 10px; font-weight: bold;">
        Total Doors Ordered: {total_doors}
    </div>
    
    <div class="special-notes">
        <strong>Special Instructions:</strong><br>
        Door Style #231 - All doors only (no drawer fronts or false fronts)<br>
        <span style="color: red; font-weight: bold;">IMPORTANT: No hinge boring on Cabinet #4 - Trash drawer</span>
    </div>
    
    <div class="company-footer">
        <p style="margin-top: 30px; border-top: 1px solid #000; padding-top: 10px;">
            For Office Use Only: Order #_________ Date Received_________ Expected Completion_________
        </p>
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
            'marginTop': 0.4,
            'marginBottom': 0.4,
            'marginLeft': 0.4,
            'marginRight': 0.4
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
    print("Creating Door Order Form - Style #231 (Matching Original Layout)")
    print("=" * 60)
    
    # Load standard door list
    door_list = load_standard_door_list("standard_door_list.json")
    
    if not door_list:
        print("[ERROR] No door list found")
        sys.exit(1)
    
    # Extract only doors
    doors = extract_doors_only(door_list)
    
    print(f"Found {len(doors)} door items")
    print("-" * 60)
    
    # Generate HTML
    html_content = generate_order_form_html(doors)
    
    # Save HTML
    html_file = "door_231_order_form.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[OK] Generated {html_file}")
    
    # Convert to PDF
    pdf_file = "door_231_order_form.pdf"
    if html_to_pdf(html_file, pdf_file):
        print(f"[OK] Generated {pdf_file}")
    
    print("\nOrder form complete!")