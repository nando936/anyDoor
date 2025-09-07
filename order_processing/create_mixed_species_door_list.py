"""
Create Door List for Paul Revere Mixed Species Order
Paint Grade (Lines 1-5) and White Oak (Lines 6-8)
"""

import os
from datetime import datetime

# Customer information from the mixed species PDF
customer_info = {
    'name': 'Paul Revere',
    'address': '1776 Liberty Lane, Boston TX 77385',
    'phone': '(281) 555-1775',
    'email': 'prevere@colonialcabinets.com',
    'job_name': 'Revere Kitchen Remodel',
    'date': '09/07/2025',
    'door_style': '231'
}

# Door items with mixed wood species
door_items = [
    {'cabinet': 1, 'qty': 2, 'width': '14 3/8', 'height': '24 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #1'},
    {'cabinet': 2, 'qty': 2, 'width': '14 7/16', 'height': '24 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #2'},
    {'cabinet': 3, 'qty': 1, 'width': '14 3/4', 'height': '24 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'No hinge boring - trash drawer'},
    {'cabinet': 4, 'qty': 2, 'width': '13 3/4', 'height': '24 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #6'},
    {'cabinet': 5, 'qty': 2, 'width': '14 3/4', 'height': '24 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #7'},
    {'cabinet': 6, 'qty': 2, 'width': '13 13/16', 'height': '24 3/4', 'type': 'door', 'material': 'White Oak', 'notes': 'Cabinet #10'},
    {'cabinet': 7, 'qty': 2, 'width': '16 1/2', 'height': '41 7/8', 'type': 'door', 'material': 'White Oak', 'notes': 'Cabinet #23'},
    {'cabinet': 8, 'qty': 2, 'width': '16 1/2', 'height': '41 7/8', 'type': 'door', 'material': 'White Oak', 'notes': 'Cabinet #24'}
]

door_style = '231'

# Create output folder for mixed species order
output_folder = 'output/paul_revere_231_mixed'
os.makedirs(output_folder, exist_ok=True)

print("Creating Door List for Mixed Species Order")
print("=" * 60)
print(f"Customer: {customer_info['name']}")
print(f"Job: {customer_info['job_name']}")
print(f"Door Style: #{door_style}")
print(f"Wood Species: Paint Grade & White Oak")
print("=" * 60)

# Create door list HTML
door_pic_path = f"../../door pictures/{door_style} door pic.JPG"
door_profile_path = f"../../door pictures/{door_style} door profile.JPG"

html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Standard Door List - {customer_info['name']} Mixed Species Order</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ border: 2px solid #333; padding: 15px; margin-bottom: 20px; }}
        .door-pictures {{ text-align: center; margin-bottom: 20px; }}
        .door-pictures img {{ max-height: 200px; margin: 10px; border: 1px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #333; color: white; padding: 8px; text-align: left; }}
        td {{ border: 1px solid #ccc; padding: 6px; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .special-note {{ color: #cc0000; font-weight: bold; }}
        .paint-grade {{ background-color: #e8f4ff; }}
        .white-oak {{ background-color: #fff4e8; }}
        .species-summary {{ 
            margin-top: 20px; 
            padding: 10px; 
            background: #f0f0f0; 
            border: 1px solid #999;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>STANDARD DOOR LIST - MIXED SPECIES</h1>
        <p><strong>Customer:</strong> {customer_info['name']}</p>
        <p><strong>Job:</strong> {customer_info['job_name']}</p>
        <p><strong>Date:</strong> {customer_info['date']}</p>
        <p><strong>Door Style:</strong> #{door_style}</p>
        <p><strong>Wood Species:</strong> Paint Grade & White Oak (see individual line items)</p>
        <p><strong>Hinge Type:</strong> Blum Soft Close Frameless 1/2"OL</p>
        <p><strong>Overlay:</strong> 1/2" Overlay</p>
    </div>
    
    <div class="door-pictures">
        <p><em>Door Style #{door_style} Pictures</em></p>
        <img src="{door_pic_path}" alt="Door {door_style} Front">
        <img src="{door_profile_path}" alt="Door {door_style} Profile">
    </div>
    
    <table>
        <thead>
            <tr>
                <th width="10%">Cabinet #</th>
                <th width="8%">Qty</th>
                <th width="12%">Width</th>
                <th width="12%">Height</th>
                <th width="10%">Type</th>
                <th width="18%">Material</th>
                <th width="8%">Style</th>
                <th width="22%">Notes</th>
            </tr>
        </thead>
        <tbody>"""

# Count totals by species
paint_grade_count = 0
white_oak_count = 0

for item in door_items:
    special_class = ' class="special-note"' if 'no hinge' in item['notes'].lower() else ''
    row_class = 'paint-grade' if item['material'] == 'Paint Grade' else 'white-oak'
    
    if item['material'] == 'Paint Grade':
        paint_grade_count += item['qty']
    else:
        white_oak_count += item['qty']
    
    html += f"""
            <tr class="{row_class}">
                <td>#{item['cabinet']}</td>
                <td>{item['qty']}</td>
                <td>{item['width']}</td>
                <td>{item['height']}</td>
                <td>{item['type']}</td>
                <td><strong>{item['material']}</strong></td>
                <td>{door_style}</td>
                <td{special_class}>{item['notes']}</td>
            </tr>"""

html += f"""
        </tbody>
    </table>
    
    <div class="species-summary">
        <h3>Species Summary</h3>
        <table style="width: auto;">
            <tr>
                <td style="padding: 5px; background: #e8f4ff; border: 1px solid #333;"><strong>Paint Grade:</strong></td>
                <td style="padding: 5px; border: 1px solid #333;">{paint_grade_count} doors (Cabinets #1-5)</td>
            </tr>
            <tr>
                <td style="padding: 5px; background: #fff4e8; border: 1px solid #333;"><strong>White Oak:</strong></td>
                <td style="padding: 5px; border: 1px solid #333;">{white_oak_count} doors (Cabinets #6-8)</td>
            </tr>
            <tr style="font-weight: bold;">
                <td style="padding: 5px; border: 1px solid #333;">TOTAL:</td>
                <td style="padding: 5px; border: 1px solid #333;">{paint_grade_count + white_oak_count} doors</td>
            </tr>
        </table>
        <p style="margin-top: 10px;"><span class="special-note">IMPORTANT: No hinge boring on Cabinet #3 - Trash drawer</span></p>
    </div>
</body>
</html>"""

# Save door list HTML
door_list_file = f"{output_folder}/paul_revere_231_mixed_door_list.html"
with open(door_list_file, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n[OK] Created door list: {door_list_file}")

# Save JSON data for reference
import json
json_data = {
    'customer_info': customer_info,
    'door_style': door_style,
    'door_items': door_items,
    'species_summary': {
        'paint_grade': paint_grade_count,
        'white_oak': white_oak_count,
        'total': paint_grade_count + white_oak_count
    },
    'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

json_file = f"{output_folder}/paul_revere_231_mixed_data.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2)

print(f"[OK] Saved JSON data: {json_file}")

print("\n" + "=" * 60)
print("Door List Summary:")
print(f"  Paint Grade: {paint_grade_count} doors")
print(f"  White Oak: {white_oak_count} doors")
print(f"  Total: {paint_grade_count + white_oak_count} doors")
print("\nStopping here as requested.")