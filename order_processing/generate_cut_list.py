"""
Generate Cut List for Paul Revere Order
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

generate_cut_list_html = module.generate_cut_list_html
load_door_specs = module.load_door_specs

# Customer information
customer_info = {
    'name': 'Paul Revere',
    'address': '1776 Liberty Lane, Boston TX 77385',
    'phone': '(281) 555-1775',
    'email': 'prevere@colonialcabinets.com',
    'job_name': 'Revere Kitchen Remodel',
    'job_number': '231',
    'date': '09/07/2025',
    'wood_species': 'Mixed (Paint Grade & White Oak)',
    'hinge_type': 'Blum Soft Close Frameless 1/2"OL',
    'overlay': '1/2"',
    'door_sizes': 'Opening Sizes',
    'bore_prep': False,
    'outside_edge': 'Standard for #231',
    'inside_edge': 'Standard for #231', 
    'panel_cut': '3/8" Plywood (Flat Panel ONLY)',
    'drawer_type': '5 piece'
}

# Door items (using finish sizes since they've been converted)
door_items = [
    {'cabinet': 1, 'qty': 2, 'width': '15 3/8', 'height': '25 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #1'},
    {'cabinet': 2, 'qty': 2, 'width': '15 7/16', 'height': '25 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #2'},
    {'cabinet': 3, 'qty': 1, 'width': '15 3/4', 'height': '25 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'No hinge boring - trash drawer'},
    {'cabinet': 4, 'qty': 2, 'width': '14 3/4', 'height': '25 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #6'},
    {'cabinet': 5, 'qty': 2, 'width': '15 3/4', 'height': '25 3/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #7'},
    {'cabinet': 6, 'qty': 2, 'width': '14 13/16', 'height': '25 3/4', 'type': 'door', 'material': 'White Oak', 'notes': 'Cabinet #10'},
    {'cabinet': 7, 'qty': 2, 'width': '17 1/2', 'height': '42 7/8', 'type': 'door', 'material': 'White Oak', 'notes': 'Cabinet #23'},
    {'cabinet': 8, 'qty': 2, 'width': '17 1/2', 'height': '42 7/8', 'type': 'door', 'material': 'White Oak', 'notes': 'Cabinet #24'}
]

door_style = '231'

print("=" * 60)
print("Generating Cut List")
print("=" * 60)

# Load door specs
door_specs = load_door_specs(door_style)

# Generate cut list HTML
cut_list_html = generate_cut_list_html(customer_info, door_items, door_style, door_specs)

# Save to output folder
output_folder = 'output/paul_revere_231'
os.makedirs(output_folder, exist_ok=True)

cut_list_file = f"{output_folder}/cut_list.html"
with open(cut_list_file, 'w', encoding='utf-8') as f:
    f.write(cut_list_html)

print(f"\n[OK] Cut list created: {cut_list_file}")

print("\n" + "=" * 60)
print("Cut List Complete")
print("File created:")
print(f"  - HTML: {cut_list_file}")