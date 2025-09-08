"""
Process Benjamin Franklin Order (Door Style 103)
"""

import sys
import os
sys.path.append('.')
sys.stdout.reconfigure(encoding='utf-8')

# Import processing functions
import importlib.util
spec = importlib.util.spec_from_file_location("process_new_order", "1_process_new_order.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

process_order = module.process_order
generate_shop_report_html = module.generate_shop_report_html
generate_cut_list_html = module.generate_cut_list_html
load_door_specs = module.load_door_specs

# Customer information
customer_info = {
    'name': 'Benjamin Franklin',
    'address': '1706 Market Street, Philadelphia TX 77019',
    'phone': '(713) 555-1706',
    'email': 'bfranklin@customkitchens.com',
    'job_name': 'Franklin Master Kitchen',
    'job_number': '103',
    'date': '01/08/2025',
    'wood_species': 'Mixed (Paint Grade & Stain Grade Maple)',
    'hinge_type': 'Blum Soft Close Frameless 1/2"OL',
    'overlay': '1/2"',
    'door_sizes': 'Finish Sizes',
    'bore_prep': True,
    'outside_edge': 'Standard for #103',
    'inside_edge': 'Standard for #103',
    'panel_cut': '1/4" MDF Raised Panel',
    'drawer_type': '5 piece'
}

# Door items (already in finish sizes)
door_items = [
    {'cabinet': 1, 'qty': 2, 'width': '14 1/2', 'height': '30 1/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #1'},
    {'cabinet': 2, 'qty': 2, 'width': '15 3/4', 'height': '30 1/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #2'},
    {'cabinet': 3, 'qty': 1, 'width': '23 7/8', 'height': '30 1/4', 'type': 'door', 'material': 'Stain Grade Maple', 'notes': 'Cabinet #3'},
    {'cabinet': 4, 'qty': 2, 'width': '17 1/2', 'height': '30 1/4', 'type': 'door', 'material': 'Stain Grade Maple', 'notes': 'Cabinet #4'},
    {'cabinet': 5, 'qty': 2, 'width': '11 3/4', 'height': '24 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #5'},
    {'cabinet': 6, 'qty': 1, 'width': '17 7/8', 'height': '24 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #6'},
    {'cabinet': 7, 'qty': 2, 'width': '14 7/8', 'height': '24 3/4', 'type': 'door', 'material': 'Stain Grade Maple', 'notes': 'Cabinet #7'},
    {'cabinet': 8, 'qty': 2, 'width': '17 1/2', 'height': '24 3/4', 'type': 'door', 'material': 'Stain Grade Maple', 'notes': 'Cabinet #8'},
    {'cabinet': 9, 'qty': 2, 'width': '20 3/8', 'height': '36 1/2', 'type': 'door', 'material': 'Stain Grade Maple', 'notes': 'Cabinet #9'},
    {'cabinet': 10, 'qty': 2, 'width': '14 1/4', 'height': '42 1/2', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #10'},
    {'cabinet': 11, 'qty': 1, 'width': '35 1/2', 'height': '18 3/4', 'type': 'door', 'material': 'Stain Grade Maple', 'notes': 'Cabinet #11 - Horizontal'},
    {'cabinet': 12, 'qty': 2, 'width': '15 5/8', 'height': '19 1/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #12'}
]

door_style = '103'

print("=" * 60)
print("Processing Benjamin Franklin Order - Door Style 103")
print("=" * 60)

# Process the order to create finish door list
door_list_html = process_order(customer_info, door_items, door_style, 'benjamin_franklin_103')

# Load door specs
door_specs = load_door_specs(door_style)

# Generate shop report
shop_report_html = generate_shop_report_html(customer_info, door_items, door_style, door_specs)

# Generate cut list
cut_list_html = generate_cut_list_html(customer_info, door_items, door_style, door_specs)

# Save all reports
output_folder = 'output/benjamin_franklin_103'
os.makedirs(output_folder, exist_ok=True)

# Save door list
door_list_file = f"{output_folder}/finish_door_list.html"
with open(door_list_file, 'w', encoding='utf-8') as f:
    f.write(door_list_html)
print(f"\n[OK] Finish door list created: {door_list_file}")

# Save shop report
shop_report_file = f"{output_folder}/shop_report.html"
with open(shop_report_file, 'w', encoding='utf-8') as f:
    f.write(shop_report_html)
print(f"[OK] Shop report created: {shop_report_file}")

# Save cut list
cut_list_file = f"{output_folder}/cut_list.html"
with open(cut_list_file, 'w', encoding='utf-8') as f:
    f.write(cut_list_html)
print(f"[OK] Cut list created: {cut_list_file}")

print("\n" + "=" * 60)
print("Order Processing Complete")
print("All reports generated in: output/benjamin_franklin_103/")
print("=" * 60)
print("\nNext step: Run critical_verification.py to validate the order")