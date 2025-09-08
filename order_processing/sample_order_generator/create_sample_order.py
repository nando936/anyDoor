"""
Create Sample Customer Order Forms
This script generates sample order forms using the standard template format
"""

import sys
import os
from datetime import datetime, timedelta
import random

sys.stdout.reconfigure(encoding='utf-8')

def generate_order_html(customer_info, door_specs, door_items, special_notes=""):
    """Generate HTML order form using the standard template"""
    
    # Calculate total doors
    total_doors = sum(item.get('qty', 0) for item in door_items if item.get('qty'))
    
    # Build door items rows
    door_rows = ""
    for i in range(20):  # Always 20 rows in the table
        if i < len(door_items):
            item = door_items[i]
            door_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td>{item.get('qty', '')}</td>
                <td>{item.get('width', '')}</td>
                <td>{item.get('height', '')}</td>
                <td>{item.get('type', '')}</td>
                <td style="font-size: 9pt;">{item.get('notes', '')}</td>
            </tr>"""
        else:
            door_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
            </tr>"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Door Order Form - Style {door_specs['door_number']}</title>
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
                        <span class="checkbox{'checked' if door_specs['size_type'] == 'finish' else ''}"></span> Finish Sizes
                    </span>
                    <span style="margin: 0 10px;">or</span>
                    <span class="checkbox-container">
                        <span class="checkbox{' checked' if door_specs['size_type'] == 'opening' else ''}"></span> Opening Sizes
                    </span>
                </td>
                <td class="spec-label">Overlay Type:</td>
                <td>{door_specs['overlay']}</td>
            </tr>
            <tr>
                <td class="spec-label">Bore/Door Prep:</td>
                <td>
                    <span class="checkbox-container">
                        <span class="checkbox{' checked' if door_specs['bore_prep'] else ''}"></span> Yes
                    </span>
                    <span class="checkbox-container">
                        <span class="checkbox{' checked' if not door_specs['bore_prep'] else ''}"></span> No
                    </span>
                </td>
                <td class="spec-label">Hinge Type:</td>
                <td>{door_specs['hinge_type']}</td>
            </tr>
            <tr>
                <td class="spec-label">Wood Species:</td>
                <td>{door_specs['wood_species']}</td>
                <td class="spec-label">Door #:</td>
                <td style="font-weight: bold; color: red; font-size: 12pt;">{door_specs['door_number']}</td>
            </tr>
            <tr>
                <td class="spec-label">Outside Edge:</td>
                <td>{door_specs['outside_edge']}</td>
                <td class="spec-label">Sticking (Inside):</td>
                <td>{door_specs['inside_edge']}</td>
            </tr>
            <tr>
                <td class="spec-label">Panel Cut:</td>
                <td colspan="3">{door_specs['panel_cut']}</td>
            </tr>
            <tr>
                <td class="spec-label">Drawer/Front type:</td>
                <td colspan="3">{door_specs['drawer_type']}</td>
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
        <tbody>{door_rows}
        </tbody>
    </table>
    
    <div style="margin-top: 10px; font-weight: bold;">
        Total Doors Ordered: {total_doors}
    </div>
    
    <div class="special-notes">
        <strong>Special Instructions:</strong><br>
        {special_notes if special_notes else f"Door Style #{door_specs['door_number']} - All doors only (no drawer fronts or false fronts)"}
    </div>
    
    <div class="company-footer">
        <p style="margin-top: 30px; border-top: 1px solid #000; padding-top: 10px;">
            For Office Use Only: Order #_________ Date Received_________ Expected Completion_________
        </p>
    </div>
</body>
</html>"""
    
    return html


def create_sample_order(door_style, customer_name, output_file=None):
    """Create a sample order with given door style and customer"""
    
    # Sample customer data
    customers = {
        "Benjamin Franklin": {
            "name": "Benjamin Franklin",
            "address": "1706 Market Street, Philadelphia TX 77019",
            "phone": "(713) 555-1706",
            "email": "bfranklin@customkitchens.com",
            "job_name": "Franklin Master Kitchen"
        },
        "George Washington": {
            "name": "George Washington",
            "address": "1600 Mount Vernon Way, Houston TX 77001",
            "phone": "(832) 555-1776",
            "email": "gwashington@colonialcabinets.com",
            "job_name": "Washington Estate Kitchen"
        },
        "Thomas Jefferson": {
            "name": "Thomas Jefferson",
            "address": "931 Monticello Drive, Austin TX 78701",
            "phone": "(512) 555-1743",
            "email": "tjefferson@finecabinets.com",
            "job_name": "Jefferson Library Cabinets"
        }
    }
    
    # Get customer info
    if customer_name in customers:
        customer_info = customers[customer_name].copy()
    else:
        # Create generic customer
        customer_info = {
            "name": customer_name,
            "address": "123 Main Street, Houston TX 77001",
            "phone": "(713) 555-0000",
            "email": "customer@example.com",
            "job_name": f"{customer_name} Kitchen Project"
        }
    
    customer_info['date'] = datetime.now().strftime('%m/%d/%Y')
    
    # Door specifications based on style
    door_specs = {
        'door_number': door_style,
        'size_type': 'finish' if random.choice([True, False]) else 'opening',
        'overlay': '1/2" Overlay',
        'bore_prep': True,
        'hinge_type': 'Blum Soft Close Frameless 1/2"OL',
        'wood_species': 'Paint Grade & Stain Grade Maple (see notes)',
        'outside_edge': f'Standard for #{door_style}',
        'inside_edge': f'Standard for #{door_style}',
        'panel_cut': '1/4" MDF Raised Panel' if door_style == '103' else '3/8" Plywood (Flat Panel ONLY)',
        'drawer_type': '5 piece'
    }
    
    # Generate sample door items
    door_items = []
    cabinet_num = 1
    materials = ['Paint Grade', 'Stain Grade Maple']
    
    # Generate 8-12 random door items
    num_items = random.randint(8, 12)
    for i in range(num_items):
        width = random.choice(['11 3/4', '14 1/2', '15 3/4', '17 1/2', '20 3/8', '23 7/8'])
        height = random.choice(['18 3/4', '24 3/4', '30 1/4', '36 1/2', '42 1/2'])
        qty = random.choice([1, 2, 2])  # More likely to be 2
        material = random.choice(materials)
        
        door_items.append({
            'qty': qty,
            'width': width,
            'height': height,
            'type': 'doors',
            'notes': f'Cabinet #{cabinet_num} - {material}'
        })
        cabinet_num += 1
    
    # Generate HTML
    html = generate_order_html(customer_info, door_specs, door_items)
    
    # Save to file
    if not output_file:
        output_file = f"door_{door_style}_order_form.html"
    
    output_path = os.path.join('..', 'need to process', output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"[OK] Created sample order: {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage
    print("Sample Order Generator")
    print("=" * 50)
    
    if len(sys.argv) > 2:
        door_style = sys.argv[1]
        customer_name = " ".join(sys.argv[2:])
        create_sample_order(door_style, customer_name)
    else:
        print("Usage: python create_sample_order.py [door_style] [customer_name]")
        print("\nExample:")
        print("  python create_sample_order.py 103 Benjamin Franklin")
        print("  python create_sample_order.py 231 George Washington")
        print("\nCreating default sample...")
        create_sample_order("103", "Benjamin Franklin")