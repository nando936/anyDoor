"""
Process Paul Revere Door Order #231
Creates door list only (stopping after door list as requested)
"""

import json
import os
from datetime import datetime

# Customer information extracted from PDF
customer_info = {
    'name': 'Paul Revere',
    'address': '1776 Liberty Lane, Boston TX 77385',
    'phone': '(281) 555-1775',
    'email': 'prevere@colonialcabinets.com',
    'job_name': 'Revere Kitchen Remodel',
    'job_number': '231',
    'date': '09/07/2025',
    'wood_species': 'White Oak',
    'hinge_type': 'Blum Soft Close Frameless 1/2"OL',
    'overlay': '1/2" Overlay',
    'bore_prep': True,
    'panel_cut': '3/8" Plywood (Flat Panel ONLY)',
    'outside_edge': 'Standard for #231',
    'inside_edge': 'Standard for #231',
    'door_sizes': 'Opening Sizes'
}

# Door items extracted from PDF
door_items = [
    {'line': 1, 'cabinet': 1, 'qty': 2, 'width': '14 3/8', 'height': '24 3/4', 'type': 'door', 'notes': 'Cabinet #1'},
    {'line': 2, 'cabinet': 2, 'qty': 2, 'width': '14 7/16', 'height': '24 3/4', 'type': 'door', 'notes': 'Cabinet #2'},
    {'line': 3, 'cabinet': 3, 'qty': 1, 'width': '14 3/4', 'height': '24 3/4', 'type': 'door', 'notes': 'No hinge boring - trash drawer'},
    {'line': 4, 'cabinet': 4, 'qty': 2, 'width': '13 3/4', 'height': '24 3/4', 'type': 'door', 'notes': 'Cabinet #6'},
    {'line': 5, 'cabinet': 5, 'qty': 2, 'width': '14 3/4', 'height': '24 3/4', 'type': 'door', 'notes': 'Cabinet #7'},
    {'line': 6, 'cabinet': 6, 'qty': 2, 'width': '13 13/16', 'height': '24 3/4', 'type': 'door', 'notes': 'Cabinet #10'},
    {'line': 7, 'cabinet': 7, 'qty': 2, 'width': '16 1/2', 'height': '41 7/8', 'type': 'door', 'notes': 'Cabinet #23'},
    {'line': 8, 'cabinet': 8, 'qty': 2, 'width': '16 1/2', 'height': '41 7/8', 'type': 'door', 'notes': 'Cabinet #24'}
]

door_style = '231'

# Create output folder
output_folder = 'output/paul_revere_231'
os.makedirs(output_folder, exist_ok=True)

print("Processing Paul Revere Door Order #231")
print("=" * 60)
print(f"Customer: {customer_info['name']}")
print(f"Job: {customer_info['job_name']}")
print(f"Door Style: #{door_style}")
print(f"Total Doors: 15")
print(f"Output folder: {output_folder}")
print("=" * 60)

# Create door list HTML
door_pic_path = f"../../door pictures/{door_style} door pic.JPG"
door_profile_path = f"../../door pictures/{door_style} door profile.JPG"

html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Standard Door List - {customer_info['name']} Job {door_style}</title>
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
    </style>
</head>
<body>
    <div class="header">
        <h1>STANDARD DOOR LIST</h1>
        <p>Customer: {customer_info['name']}</p>
        <p>Job: {customer_info['job_name']}</p>
        <p>Date: {customer_info['date']}</p>
        <p>Door Style: #{door_style} - {customer_info['wood_species']}</p>
        <p>Hinge Type: {customer_info['hinge_type']}</p>
        <p>Overlay: {customer_info['overlay']}</p>
    </div>
    
    <div class="door-pictures">
        <p><em>Door Style #{door_style} Pictures</em></p>
        <img src="{door_pic_path}" alt="Door {door_style} Front">
        <img src="{door_profile_path}" alt="Door {door_style} Profile">
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Cabinet #</th>
                <th>Qty</th>
                <th>Width</th>
                <th>Height</th>
                <th>Type</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>"""

for item in door_items:
    special_class = ' class="special-note"' if 'no hinge' in item['notes'].lower() else ''
    html += f"""
            <tr>
                <td>#{item['cabinet']}</td>
                <td>{item['qty']}</td>
                <td>{item['width']}</td>
                <td>{item['height']}</td>
                <td>{item['type']}</td>
                <td{special_class}>{item['notes']}</td>
            </tr>"""

html += """
        </tbody>
    </table>
    
    <div style="margin-top: 20px; padding: 10px; background: #f0f0f0;">
        <strong>Total Doors Ordered: 15</strong><br>
        <strong>Special Instructions:</strong> Door Style #231 - All doors only (no drawer fronts or false fronts)<br>
        <span class="special-note">IMPORTANT: No hinge boring on Line #3 (Cabinet #3) - Trash drawer</span>
    </div>
</body>
</html>"""

# Save door list HTML
door_list_file = f"{output_folder}/paul_revere_231_door_list.html"
with open(door_list_file, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n[OK] Created door list: {door_list_file}")

# Save JSON data for reference
json_data = {
    'customer_info': customer_info,
    'door_style': door_style,
    'door_items': door_items,
    'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
json_file = f"{output_folder}/paul_revere_231_data.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2)

print(f"[OK] Saved JSON data: {json_file}")

print("\n" + "=" * 60)
print("Door list created successfully!")
print("Stopping here as requested.")
print(f"Door list available at: {door_list_file}")