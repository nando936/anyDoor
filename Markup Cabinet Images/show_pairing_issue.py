"""
Show the issue with current pairing logic and what it should be
"""
import json

# Load the measurements
with open("//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3_measurements_data.json", 'r') as f:
    data = json.load(f)

print("="*60)
print("CURRENT PAIRING ISSUE")
print("="*60)

# Get positions
widths = []
heights = []

for m in data['measurements']:
    if m['text'] in data['horizontal']:
        widths.append({'text': m['text'], 'x': m['position'][0], 'y': m['position'][1]})
    elif m['text'] in data['vertical']:
        heights.append({'text': m['text'], 'x': m['position'][0], 'y': m['position'][1]})

print("\nWidth positions (Y coordinate):")
for w in widths:
    print(f"  {w['text']:10} at Y={w['y']:.0f}")

print("\nHeight positions (Y coordinate):")
for h in sorted(heights, key=lambda x: x['y']):
    print(f"  {h['text']:10} at Y={h['y']:.0f}")

print("\n" + "="*60)
print("WHAT'S HAPPENING NOW (AFTER FIX):")
print("="*60)
# Load the fixed results
with open("//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3_openings_data.json", 'r') as f:
    fixed_data = json.load(f)

for opening in fixed_data['openings']:
    if opening['width'] == '23':
        print(f"Width '23' at Y={opening['width_pos'][1]:.0f} pairs with '{opening['height']}' at Y={opening['height_pos'][1]:.0f}")
        if opening['height_pos'][1] < opening['width_pos'][1]:
            print(f"  => This is ABOVE the width (Y={opening['height_pos'][1]:.0f} < Y={opening['width_pos'][1]:.0f}) - CORRECT!")
        else:
            print(f"  => This is BELOW the width (Y={opening['height_pos'][1]:.0f} > Y={opening['width_pos'][1]:.0f}) - WRONG!")

print("\n" + "="*60)
print("WHAT SHOULD HAPPEN (CORRECT):")
print("="*60)

# Find heights above each width
for w in widths:
    print(f"\nWidth '{w['text']}' at Y={w['y']:.0f} should look for heights ABOVE it:")
    above_heights = [h for h in heights if h['y'] < w['y']]  # Y < width means above

    if above_heights:
        # Sort by distance
        above_heights_with_dist = []
        for h in above_heights:
            dist = ((h['x'] - w['x'])**2 + (h['y'] - w['y'])**2)**0.5
            above_heights_with_dist.append((h, dist))

        above_heights_with_dist.sort(key=lambda x: x[1])

        for h, dist in above_heights_with_dist:
            print(f"  - Height '{h['text']}' at Y={h['y']:.0f} (distance: {dist:.0f})")

        best = above_heights_with_dist[0][0]
        print(f"  => SHOULD PAIR with '{best['text']}' at Y={best['y']:.0f}")
    else:
        print(f"  - No heights above this width!")

print("\n" + "="*60)
print("THE FIX:")
print("="*60)
print("In proximity_pairing_detector.py, line 212-214:")
print("Add condition to only consider heights above:")
print("")
print("for height in heights:")
print("    # Skip if already used")
print("    if id(height) in used_heights:")
print("        continue")
print("    ")
print("    # ONLY CONSIDER HEIGHTS ABOVE (smaller Y)")
print("    if height['y'] >= width['y']:  # Skip if below or same level")
print("        continue")
print("    ")
print("    # Calculate distance...")