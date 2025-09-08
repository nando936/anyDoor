"""
Generate Sample Door Order for Testing
This script creates sample door orders in both JSON and PDF format
for testing the order processing system.

Usage:
    python generate_sample_order.py [door_style] [customer_name]
    
Examples:
    python generate_sample_order.py 103 "John Smith"
    python generate_sample_order.py 231 "Jane Doe"
"""

import json
import sys
import os
from datetime import datetime
import random

# Add parent directory to path for imports
sys.path.append('..')
sys.stdout.reconfigure(encoding='utf-8')

def generate_sample_order(door_style="103", customer_name="Sample Customer"):
    """Generate a sample order with realistic data"""
    
    # Sample customer data
    customer_info = {
        "name": customer_name,
        "address": f"{random.randint(100, 9999)} Main Street, Houston TX {random.randint(77001, 77099)}",
        "phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
        "email": f"{customer_name.lower().replace(' ', '.')}@example.com",
        "job_name": f"{customer_name.split()[0]} Kitchen Remodel",
        "job_number": str(random.randint(100, 999)),
        "date": datetime.now().strftime("%m/%d/%Y"),
        "wood_species": random.choice(["White Oak", "Paint Grade", "Stain Grade Maple", "Cherry"]),
        "door_style": door_style,
        "hinge_type": "Blum Soft Close Frameless 1/2\"OL",
        "overlay": "1/2\"",
        "bore_prep": random.choice([True, False]),
        "door_sizes": random.choice(["finish", "opening"]),
        "panel_cut": "1/4\" MDF Raised Panel" if door_style == "103" else "3/8\" Plywood (Flat Panel ONLY)",
        "outside_edge": f"Standard for #{door_style}",
        "inside_edge": f"Standard for #{door_style}",
        "drawer_type": "5 piece"
    }
    
    # Generate random door items
    door_items = []
    cabinet_num = 1
    
    # Mix of different door sizes and types
    sizes = [
        ("14 3/8", "24 3/4"),
        ("15 1/2", "30 1/4"),
        ("17 7/8", "24 3/4"),
        ("23 3/4", "30 1/4"),
        ("11 3/4", "24 3/4"),
        ("35 1/2", "18 3/4"),  # Horizontal
        ("20 3/8", "36 1/2"),
        ("14 1/4", "42 1/2")
    ]
    
    # Generate 8-12 cabinets
    num_cabinets = random.randint(8, 12)
    
    for i in range(num_cabinets):
        width, height = random.choice(sizes)
        qty = random.choice([1, 2, 2])  # Most commonly 2 doors per cabinet
        
        # Occasionally add special notes
        notes = ""
        if random.random() < 0.1:
            notes = random.choice([
                "No hinge boring - trash drawer",
                f"Cabinet #{cabinet_num}",
                "Horizontal grain",
                ""
            ])
        
        door_items.append({
            "cabinet": cabinet_num,
            "qty": qty,
            "width": width,
            "height": height,
            "type": "door",
            "material": customer_info["wood_species"],
            "notes": notes
        })
        
        cabinet_num += 1
    
    # Add a drawer front occasionally
    if random.random() < 0.3:
        door_items.append({
            "cabinet": cabinet_num,
            "qty": 1,
            "width": random.choice(["15 3/4", "17 1/2", "23 7/8"]),
            "height": random.choice(["5 3/4", "7 1/2", "9 1/4"]),
            "type": "drawer",
            "material": customer_info["wood_species"],
            "notes": ""
        })
    
    return customer_info, door_items

def save_sample_order(customer_info, door_items, filename_prefix):
    """Save sample order to JSON file"""
    
    order_data = {
        "customer_info": customer_info,
        "door_items": door_items
    }
    
    # Save to JSON
    json_filename = f"{filename_prefix}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(order_data, f, indent=2)
    
    print(f"[OK] Created sample order: {json_filename}")
    print(f"     Customer: {customer_info['name']}")
    print(f"     Job: {customer_info['job_name']} (#{customer_info['job_number']})")
    print(f"     Door Style: #{customer_info['door_style']}")
    print(f"     Items: {len(door_items)} cabinets")
    
    return json_filename

def main():
    """Main function to generate sample orders"""
    
    # Parse command line arguments
    door_style = sys.argv[1] if len(sys.argv) > 1 else "103"
    customer_name = sys.argv[2] if len(sys.argv) > 2 else "Sample Customer"
    
    print("=" * 60)
    print("SAMPLE ORDER GENERATOR")
    print("=" * 60)
    
    # Generate the sample order
    customer_info, door_items = generate_sample_order(door_style, customer_name)
    
    # Create filename from customer name and job number
    filename_prefix = f"sample_{customer_info['name'].lower().replace(' ', '_')}_{customer_info['job_number']}"
    
    # Save the order
    json_file = save_sample_order(customer_info, door_items, filename_prefix)
    
    print("\nTo process this order:")
    print(f"1. Copy {json_file} to parent directory as extraction_template.json")
    print(f"2. Run: python ../process_extracted_order.py {json_file}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()