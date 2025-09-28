"""
Debug why markers are overlapping with text
"""
import json
import numpy as np

# Load the data
with open('//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3_measurements_data.json', 'r') as f:
    measurement_data = json.load(f)

with open('//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3_openings_data.json', 'r') as f:
    pairing_data = json.load(f)

measurements = measurement_data['measurements']
openings = pairing_data['openings']

# Simulate the marker placement logic for opening 2 (which overlaps)
opening = openings[1]  # Opening 2: 12 7/8 W x 5 1/2 H
print(f"Debugging Opening {opening['number']}: {opening['specification']}")
print("=" * 60)

# Get intersection point
width_pos = opening['width_pos']
height_pos = opening['height_pos']
intersection_x = int(width_pos[0])
intersection_y = int(height_pos[1])
print(f"Intersection point: ({intersection_x}, {intersection_y})")

# Build avoid_regions like the code does
avoid_regions = []

def add_measurement_bounds(meas_text, pos, measurements):
    for m in measurements:
        if m['text'] == meas_text:
            if 'full_bounds' in m and m['full_bounds']:
                region = {
                    'left': m['full_bounds']['left'],
                    'right': m['full_bounds']['right'],
                    'top': m['full_bounds']['top'],
                    'bottom': m['full_bounds']['bottom'],
                    'center': pos
                }
                avoid_regions.append(region)
                print(f"  Added bounds for '{meas_text}': left={region['left']}, right={region['right']}, top={region['top']}, bottom={region['bottom']}")
                return
            elif 'bounds' in m:
                left_bound, right_bound = m['bounds']
                region = {
                    'left': left_bound,
                    'right': right_bound,
                    'top': pos[1] - 20,
                    'bottom': pos[1] + 20,
                    'center': pos
                }
                avoid_regions.append(region)
                print(f"  Added bounds for '{meas_text}': left={region['left']}, right={region['right']}, top={region['top']}, bottom={region['bottom']}")
                return

# Add bounds for width and height of current opening
print("\nAdding bounds for current opening:")
add_measurement_bounds(opening['width'], width_pos, measurements)
add_measurement_bounds(opening['height'], height_pos, measurements)

# Add all other measurements
print("\nAdding bounds for other measurements:")
for m in measurements:
    if 'text' in m and 'position' in m:
        if m['text'] != opening['width'] and m['text'] != opening['height']:
            add_measurement_bounds(m['text'], m['position'], measurements)

# Now test marker positions
radius = 45
offset_distance = 120
test_positions = [
    (intersection_x - offset_distance, intersection_y),  # Left
    (intersection_x + offset_distance, intersection_y),  # Right
    (intersection_x - offset_distance, intersection_y - offset_distance),  # Upper-left
    (intersection_x + offset_distance, intersection_y - offset_distance),  # Upper-right
    (intersection_x - offset_distance, intersection_y + offset_distance),  # Lower-left
    (intersection_x + offset_distance, intersection_y + offset_distance),  # Lower-right
    (intersection_x, intersection_y - offset_distance),  # Above
    (intersection_x, intersection_y + offset_distance),  # Below
]

print("\n" + "=" * 60)
print("Testing marker positions:")
print("=" * 60)

for i, (test_x, test_y) in enumerate(test_positions):
    position_names = ["Left", "Right", "Upper-left", "Upper-right", "Lower-left", "Lower-right", "Above", "Below"]

    # Calculate marker bounds
    marker_left = test_x - radius
    marker_right = test_x + radius
    marker_top = test_y - radius
    marker_bottom = test_y + radius + 100  # Extra space for dimension labels

    print(f"\nPosition {i+1}: {position_names[i]} at ({test_x}, {test_y})")
    print(f"  Marker bounds: left={marker_left}, right={marker_right}, top={marker_top}, bottom={marker_bottom}")

    overlap_score = 0
    overlaps = []

    for region in avoid_regions:
        # Check for overlap
        if (marker_left < region['right'] and marker_right > region['left'] and
            marker_top < region['bottom'] and marker_bottom > region['top']):
            overlap_x = min(marker_right, region['right']) - max(marker_left, region['left'])
            overlap_y = min(marker_bottom, region['bottom']) - max(marker_top, region['top'])
            score = overlap_x * overlap_y
            overlap_score += score

            # Find which measurement this is
            for m in measurements:
                if 'full_bounds' in m and m['full_bounds']:
                    if (m['full_bounds']['left'] == region['left'] and
                        m['full_bounds']['top'] == region['top']):
                        overlaps.append(f"'{m['text']}' (score={score})")
                        break

    if overlap_score > 0:
        print(f"  OVERLAPS with: {', '.join(overlaps)}")
        print(f"  Total overlap score: {overlap_score}")
    else:
        print(f"  NO OVERLAP - Clear position!")

# Show what's actually being rendered
print("\n" + "=" * 60)
print("Actual rendered text with marker:")
print("Opening 2 marker adds text: '12 7/8 x 5 1/2'")
print("This text is rendered BELOW the marker circle")