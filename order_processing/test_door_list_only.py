"""
Test door list generation with specs - Stop after door list
"""

import sys
import os
sys.path.append('.')
sys.stdout.reconfigure(encoding='utf-8')

# Import the needed functions
import importlib.util
spec = importlib.util.spec_from_file_location("process_new_order", "1_process_new_order.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

load_door_specs = module.load_door_specs
create_finish_door_list_html = module.create_finish_door_list_html

# Customer information - testing with Opening Sizes
customer_info = {
    'name': 'Paul Revere',
    'address': '1776 Liberty Lane, Boston TX 77385',
    'phone': '(281) 555-1775',
    'email': 'prevere@colonialcabinets.com',
    'job_name': 'Revere Kitchen Remodel',
    'job_number': '231-TEST',
    'date': '09/07/2025',
    'wood_species': 'Paint Grade & White Oak',
    'hinge_type': 'Blum Soft Close Frameless 1/2"OL',
    'overlay': '1/2"',
    'door_sizes': 'Opening Sizes',  # This will trigger conversion
    'bore_prep': False,
    'outside_edge': 'Standard for #231',
    'inside_edge': 'Standard for #231',
    'panel_cut': '3/8" Plywood (Flat Panel ONLY)'
}

# Door items with opening sizes (will be converted to finish)
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

print("Testing Finish Door List Generation")
print("=" * 60)
print("Features being tested:")
print("1. Door style #231 specifications loading")
print("2. Opening sizes to finish sizes conversion")
print("3. Specifications display in door list")
print("=" * 60)

# Load specifications
print("\nLoading Door Specifications...")
door_specs = load_door_specs(door_style)

# Create finish door list
print("\nCreating Finish Door List...")
door_list_html = create_finish_door_list_html(customer_info, door_items, door_style, door_specs)

# Save to file
output_folder = 'output/door_list_test'
os.makedirs(output_folder, exist_ok=True)
door_list_file = f"{output_folder}/finish_door_list_test.html"

with open(door_list_file, 'w', encoding='utf-8') as f:
    f.write(door_list_html)

print(f"   [OK] Created: {door_list_file}")

print("\n" + "=" * 60)
print("Door List Test Complete!")
print(f"Open {door_list_file} to review:")
print("- Check that opening sizes are converted to finish sizes")
print("- Verify specifications are displayed at bottom")
print("- Example: 13 7/8\" opening + (2 x 1/2\" overlay) = 14 7/8\" finish")
print("\nStopping after door list as requested.")