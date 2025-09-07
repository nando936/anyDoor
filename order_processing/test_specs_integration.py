"""
Test the specs integration with Paul Revere mixed species order
"""

import sys
import os
sys.path.append('.')

# Import the main processing function
import importlib.util
spec = importlib.util.spec_from_file_location("process_new_order", "1_process_new_order.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
process_order = module.process_order

# Customer information from the mixed species order
customer_info = {
    'name': 'Paul Revere',
    'address': '1776 Liberty Lane, Boston TX 77385',
    'phone': '(281) 555-1775',
    'email': 'prevere@colonialcabinets.com',
    'job_name': 'Revere Kitchen Remodel',
    'job_number': '231-MIXED',
    'date': '09/07/2025',
    'wood_species': 'Paint Grade & White Oak',
    'hinge_type': 'Blum Soft Close Frameless 1/2"OL',
    'overlay': '1/2"',
    'door_sizes': 'Opening Sizes',  # Testing opening to finish conversion
    'bore_prep': False,
    'outside_edge': 'Standard for #231',
    'inside_edge': 'Standard for #231',
    'panel_cut': '3/8" Plywood (Flat Panel ONLY)'
}

# Door items with mixed wood species - using opening sizes
door_items = [
    {'cabinet': 1, 'qty': 2, 'width': '13 7/8', 'height': '24 1/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #1'},
    {'cabinet': 2, 'qty': 2, 'width': '13 15/16', 'height': '24 1/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #2'},
    {'cabinet': 3, 'qty': 1, 'width': '14 1/4', 'height': '24 1/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'No hinge boring - trash drawer'},
    {'cabinet': 4, 'qty': 2, 'width': '13 1/4', 'height': '24 1/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #6'},
    {'cabinet': 5, 'qty': 2, 'width': '14 1/4', 'height': '24 1/4', 'type': 'door', 'material': 'Paint Grade', 'notes': 'Cabinet #7'},
    {'cabinet': 6, 'qty': 2, 'width': '13 5/16', 'height': '24 1/4', 'type': 'door', 'material': 'White Oak', 'notes': 'Cabinet #10'},
    {'cabinet': 7, 'qty': 2, 'width': '16', 'height': '41 3/8', 'type': 'door', 'material': 'White Oak', 'notes': 'Cabinet #23'},
    {'cabinet': 8, 'qty': 2, 'width': '16', 'height': '41 3/8', 'type': 'door', 'material': 'White Oak', 'notes': 'Cabinet #24'}
]

door_style = '231'

# Create output folder for test
output_folder = 'output/paul_revere_231_specs_test'
os.makedirs(output_folder, exist_ok=True)

output_prefix = f"{output_folder}/paul_revere_231_specs"

print("Testing Specs Integration")
print("=" * 60)
print("This will test:")
print("1. Loading door style #231 specifications")
print("2. Converting opening sizes to finish sizes (adding 2x overlay)")
print("3. Including specs in door list and shop report")
print("=" * 60)

# Process the order with specs
process_order(customer_info, door_items, door_style, output_prefix)

print("\n" + "=" * 60)
print("Test Complete!")
print("Check output/paul_revere_231_specs_test/ for results")
print("\nNote: Opening sizes have been converted to finish sizes")
print("Example: 13 7/8\" opening + (2 x 1/2\" overlay) = 14 7/8\" finish")