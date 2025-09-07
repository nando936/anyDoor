"""
Process Paul Revere Order #231
Starting fresh from the original order
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

load_door_specs = module.load_door_specs
create_finish_door_list_html = module.create_finish_door_list_html

print("=" * 60)
print("Processing New Order: Paul Revere Kitchen Remodel")
print("=" * 60)
print("\n[STEP 1] Extracting data from original order...")

# Extract customer information from original order
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
    'door_sizes': 'Opening Sizes',  # From the checkbox in original order
    'bore_prep': False,  # "No" is checked in original
    'outside_edge': 'Standard for #231',
    'inside_edge': 'Standard for #231',
    'panel_cut': '3/8" Plywood (Flat Panel ONLY)',
    'drawer_type': '5 piece'
}

# Extract door items from original order (Lines 1-8)
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

print("   Customer: Paul Revere")
print("   Job: Revere Kitchen Remodel")
print("   Door Style: #231")
print("   Total Items: 8 lines")
print("   Size Type: Opening Sizes (will convert to finish)")
print("   Materials: Lines 1-5 Paint Grade, Lines 6-8 White Oak")

# Create output folder
output_folder = f'output/{customer_info["name"].lower().replace(" ", "_")}_{door_style}'
os.makedirs(output_folder, exist_ok=True)

print(f"\n[STEP 2] Creating finish door list...")

# Load door specifications
door_specs = load_door_specs(door_style)

# Create finish door list
door_list_html = create_finish_door_list_html(customer_info, door_items, door_style, door_specs)

# Save door list
door_list_file = f"{output_folder}/finish_door_list.html"
with open(door_list_file, 'w', encoding='utf-8') as f:
    f.write(door_list_html)

print(f"   [OK] Created: {door_list_file}")

print("\n[STEP 3] Running critical verification...")
print("   This will verify all data was correctly processed")
print("\nStopping here - will run verification next")