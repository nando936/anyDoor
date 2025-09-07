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
    """Generate HTML order form with door #231"""
    
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
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px;
            font-size: 11pt;
        }}
        h1 {{ 
            text-align: center;
            font-size: 18pt;
            margin-bottom: 20px;
        }}
        .company-header {{
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }}
        .form-section {{
            margin-bottom: 20px;
            border: 1px solid #000;
            padding: 10px;
        }}
        .customer-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .field {{
            margin-bottom: 5px;
        }}
        .field label {{
            font-weight: bold;
            display: inline-block;
            width: 100px;
        }}
        .field input {{
            border: none;
            border-bottom: 1px solid #000;
            width: 250px;
        }}
        .specifications {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        .spec-item {{
            margin-bottom: 8px;
        }}
        .checkbox {{
            width: 15px;
            height: 15px;
            border: 1px solid #000;
            display: inline-block;
            margin-right: 5px;
            position: relative;
            vertical-align: middle;
        }}
        .checked::after {{
            content: "X";
            position: absolute;
            top: -3px;
            left: 2px;
            font-weight: bold;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            border: 1px solid #000;
            padding: 5px;
            text-align: center;
        }}
        th {{
            background-color: #f0f0f0;
            font-weight: bold;
        }}
        .notes-section {{
            margin-top: 20px;
            border: 1px solid #000;
            padding: 10px;
            min-height: 50px;
        }}
        @media print {{
            body {{ margin: 10px; }}
        }}
    </style>
</head>
<body>
    <div class="company-header" style="position: relative;">
        <div style="text-align: center; margin-bottom: 15px;">
            <svg width="150" height="150" viewBox="0 0 150 150" style="display: inline-block;">
                <!-- Door frame -->
                <rect x="25" y="20" width="100" height="110" fill="none" stroke="#8B4513" stroke-width="3"/>
                <!-- Door panels -->
                <rect x="35" y="30" width="35" height="40" fill="#D2691E" stroke="#8B4513" stroke-width="2"/>
                <rect x="80" y="30" width="35" height="40" fill="#D2691E" stroke="#8B4513" stroke-width="2"/>
                <rect x="35" y="80" width="35" height="40" fill="#D2691E" stroke="#8B4513" stroke-width="2"/>
                <rect x="80" y="80" width="35" height="40" fill="#D2691E" stroke="#8B4513" stroke-width="2"/>
                <!-- Door handle -->
                <circle cx="110" cy="75" r="3" fill="#696969"/>
                <!-- Company name arc -->
                <path id="textPath" d="M 30 140 Q 75 125 120 140" fill="none"/>
                <text font-family="Arial" font-size="10" font-weight="bold" fill="#8B4513">
                    <textPath href="#textPath" startOffset="50%" text-anchor="middle">
                        RAISED PANEL DOORS
                    </textPath>
                </text>
            </svg>
        </div>
        <div style="font-size: 16pt; font-weight: bold; margin-bottom: 5px;">
            THE RAISED PANEL DOOR FACTORY
        </div>
        <div>
            209 Riggs St, Conroe TX 77301<br>
            Phone: (936) 890-4338
        </div>
    </div>
    
    <h1>CABINET DOOR ORDER FORM</h1>
    
    <div class="form-section">
        <h2>Customer Information</h2>
        <div class="customer-info">
            <div>
                <div class="field">
                    <label>Name:</label>
                    <span style="border-bottom: 1px solid #000; display: inline-block; width: 250px;">{customer_info['name']}</span>
                </div>
                <div class="field">
                    <label>Phone:</label>
                    <span style="border-bottom: 1px solid #000; display: inline-block; width: 250px;">{customer_info['phone']}</span>
                </div>
                <div class="field">
                    <label>Job Name:</label>
                    <span style="border-bottom: 1px solid #000; display: inline-block; width: 250px;">{customer_info['job_name']}</span>
                </div>
            </div>
            <div>
                <div class="field">
                    <label>Address:</label>
                    <span style="border-bottom: 1px solid #000; display: inline-block; width: 250px;">{customer_info['address']}</span>
                </div>
                <div class="field">
                    <label>Email:</label>
                    <span style="border-bottom: 1px solid #000; display: inline-block; width: 250px;">{customer_info['email']}</span>
                </div>
                <div class="field">
                    <label>Date:</label>
                    <span style="border-bottom: 1px solid #000; display: inline-block; width: 250px;">{customer_info['date']}</span>
                </div>
            </div>
        </div>
    </div>
    
    <div class="form-section">
        <h2>Door Specifications</h2>
        <div class="specifications">
            <div>
                <div class="spec-item">
                    <strong>Wood Species:</strong> White Oak
                </div>
                <div class="spec-item">
                    <strong>Door Style Number:</strong> <span style="font-size: 14pt; font-weight: bold; color: red;">231</span>
                </div>
                <div class="spec-item">
                    <strong>Outside Edge:</strong> Standard for #231
                </div>
                <div class="spec-item">
                    <strong>Inside/Sticking:</strong> Standard for #231
                </div>
                <div class="spec-item">
                    <strong>Panel Cut:</strong> 3/8" Plywood (Flat Panel ONLY)
                </div>
            </div>
            <div>
                <div class="spec-item">
                    <strong>Overlay Type:</strong> 1/2" Overlay
                </div>
                <div class="spec-item">
                    <strong>Bore/Door Prep:</strong> 
                    <span class="checkbox"></span> Yes
                    <span class="checkbox checked"></span> No
                </div>
                <div class="spec-item">
                    <strong>Hinge Type:</strong> Blum Soft Close Frameless 1/2"OL
                </div>
                <div class="spec-item">
                    <strong>Door Sizes:</strong>
                    <span class="checkbox checked"></span> Opening Sizes
                    <span class="checkbox"></span> Finish Sizes
                </div>
                <div class="spec-item">
                    <strong>Drawer/Front Type:</strong> 5 piece
                </div>
            </div>
        </div>
    </div>
    
    <div class="form-section">
        <h2>Door Order Details</h2>
        <p><strong>ALL DOORS - Style #231</strong></p>
        <table>
            <thead>
                <tr>
                    <th width="10%">Item #</th>
                    <th width="15%">QTY</th>
                    <th width="20%">Width</th>
                    <th width="20%">Height</th>
                    <th width="15%">Type</th>
                    <th width="20%">Cabinet #</th>
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
        
        html += f"""
                <tr>
                    <td>{item_num}</td>
                    <td>{qty}</td>
                    <td>{door['width']}"</td>
                    <td>{door['height']}"</td>
                    <td>doors</td>
                    <td>{door['cabinet']}</td>
                </tr>"""
        item_num += 1
        total_doors += qty
    
    # Add some empty rows for potential additions
    for i in range(5):
        html += f"""
                <tr>
                    <td>{item_num + i}</td>
                    <td>&nbsp;</td>
                    <td>&nbsp;</td>
                    <td>&nbsp;</td>
                    <td>&nbsp;</td>
                    <td>&nbsp;</td>
                </tr>"""
    
    html += f"""
            </tbody>
        </table>
        <p style="margin-top: 10px;"><strong>Total Doors: {total_doors}</strong></p>
    </div>
    
    <div class="notes-section">
        <h3>Special Instructions / Notes:</h3>
        <p>Door Style #231 - All doors only (no drawer fronts or false fronts)</p>
        <p><strong>IMPORTANT: No hinge boring on Cabinet #4 - Trash drawer</strong></p>
    </div>
    
    <div style="margin-top: 30px; border-top: 1px solid #000; padding-top: 10px;">
        <p><strong>For Office Use Only:</strong></p>
        <p>Order #: _____________ &nbsp;&nbsp;&nbsp; Date Received: _____________ &nbsp;&nbsp;&nbsp; Expected Completion: _____________</p>
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
    print("Creating Door Order Form - Style #231")
    print("=" * 60)
    
    # Load standard door list
    door_list = load_standard_door_list("standard_door_list.json")
    
    if not door_list:
        print("[ERROR] No door list found")
        sys.exit(1)
    
    # Extract only doors
    doors = extract_doors_only(door_list)
    
    print(f"Found {len(doors)} door items (totaling {sum(d['qty'] for d in doors)} doors)")
    print("-" * 60)
    
    # Show door summary
    for door in sorted(doors, key=lambda x: x['cabinet'].replace('#', '').zfill(3)):
        print(f"  {door['cabinet']}: {door['qty']} doors @ {door['width']} x {door['height']}")
    
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