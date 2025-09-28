"""
Create pairing data for page 2 manually
"""
import json

# Load measurement data
with open('//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_2_measurements_data.json', 'r') as f:
    meas_data = json.load(f)

# Find the actual measurements
height_meas = None
width_meas = None

for m in meas_data['measurements']:
    if m['text'] == '35 1/8':
        height_meas = m
    elif m['text'] == '23 9/16':
        width_meas = m

# Create pairing
pairing_data = {
    "openings": [
        {
            "number": 1,
            "width": width_meas['text'],
            "height": height_meas['text'],
            "specification": f"{width_meas['text']} W × {height_meas['text']} H",
            "width_pos": width_meas['position'],
            "height_pos": height_meas['position'],
            "width_finished": False,
            "height_finished": False,
            "finished_size": False,
            "notations": []
        }
    ]
}

# Save pairing data
output_path = '//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_2_openings_data.json'
with open(output_path, 'w') as f:
    json.dump(pairing_data, f, indent=2)

print(f"Created pairing data: {output_path}")
print(f"Opening 1: {pairing_data['openings'][0]['specification']}")
print(f"  Width position: {width_meas['position']}")
print(f"  Height position: {height_meas['position']}")