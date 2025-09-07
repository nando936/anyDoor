"""
Process Order from Extraction Template
This is the standard Step 2 after extraction
"""

import sys
import os
import json
sys.path.append('.')
sys.stdout.reconfigure(encoding='utf-8')

# Import processing functions
import importlib.util
spec = importlib.util.spec_from_file_location("process_new_order", "1_process_new_order.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

process_order = module.process_order

# Load extracted data
with open('extraction_template.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

customer_info = data['customer_info']
door_items = data['door_items']

# Generate output prefix from customer name and job number
customer_name = customer_info['name'].lower().replace(' ', '_')
job_number = customer_info['job_number']
output_prefix = f"{customer_name}_{job_number}"

print("=" * 60)
print(f"Processing Order: {customer_info['name']} - Job #{job_number}")
print("=" * 60)

# Process the order - files will automatically go to output/[customer_job]/
process_order(customer_info, door_items, customer_info['door_style'], output_prefix)

print("\n" + "=" * 60)
print("Order Processing Complete")
print(f"Output folder: output/{output_prefix}/")
print("=" * 60)
print("\nNext step: Run critical_verification.py to verify the order")