"""
Draw row markers on a copy of the page image
"""
import os
import sys
import json
import cv2
import numpy as np

def draw_markers_on_page(image_path, json_path, output_path):
    """Draw row number markers on the page image"""

    # Convert paths
    if image_path.startswith('\\\\'):
        image_path = image_path.replace('\\', '/')
    if json_path.startswith('\\\\'):
        json_path = json_path.replace('\\', '/')
    if output_path.startswith('\\\\'):
        output_path = output_path.replace('\\', '/')

    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return False

    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Positions in JSON are now in full image coordinates
    # Just use them directly!

    # Draw markers for doors table
    doors_rows = data['doors_table']['rows']
    for row in doors_rows:
        marker = row['marker']
        # Position is already in full image coordinates
        pos_x = row['position']['x']
        pos_y = row['position']['y']

        # Draw marker to the left of QTY column
        marker_x = pos_x - 50  # 50 pixels to the left
        marker_y = pos_y

        # Draw white filled circle (background)
        cv2.circle(image, (marker_x, marker_y), 22, (255, 255, 255), -1)  # White filled circle

        # Draw red circle outline
        cv2.circle(image, (marker_x, marker_y), 22, (0, 0, 255), 2)  # Red outline

        # Draw text (red with #)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text = marker  # Keep the # symbol
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = marker_x - text_size[0] // 2
        text_y = marker_y + text_size[1] // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)  # Red text

    # Draw markers for drawer fronts table
    drawer_rows = data['drawer_fronts_table']['rows']
    for row in drawer_rows:
        marker = row['marker']
        # Position is already in full image coordinates
        pos_x = row['position']['x']
        pos_y = row['position']['y']

        # Draw marker to the left of QTY column
        marker_x = pos_x - 50  # 50 pixels to the left
        marker_y = pos_y

        # Draw white filled circle (background)
        cv2.circle(image, (marker_x, marker_y), 22, (255, 255, 255), -1)  # White filled circle

        # Draw red circle outline
        cv2.circle(image, (marker_x, marker_y), 22, (0, 0, 255), 2)  # Red outline

        # Draw text (red with #)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text = marker  # Keep the # symbol
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = marker_x - text_size[0] // 2
        text_y = marker_y + text_size[1] // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)  # Red text

    # Draw notes panel in bottom left
    check1 = data.get('check1', {})
    finished_checked = check1.get('finished_door_size', {}).get('checked', False)
    opening_checked = check1.get('opening_size', {}).get('checked', False)
    opening_add_value = check1.get('opening_size', {}).get('add_value', 'EMPTY')
    hinge_checked = check1.get('hinge_type', {}).get('checked', False)
    hinge_cut = check1.get('hinge_type', {}).get('cut', 'EMPTY')
    hinge_supply = check1.get('hinge_type', {}).get('supply', 'EMPTY')

    # Determine size note
    size_note = ""
    if finished_checked:
        size_note = "All Finished Sizes"
    elif opening_checked and opening_add_value != 'EMPTY':
        size_note = f"Opening Size + {opening_add_value} = Finished Size"
    else:
        size_note = "All Finished Sizes"  # Default

    # Determine hinge note
    hinge_note = ""
    if not hinge_checked and hinge_cut == 'EMPTY' and hinge_supply == 'EMPTY':
        hinge_note = "NO Hinge and NO Prep"
    else:
        hinge_note = f"Hinge: Cut={hinge_cut}, Supply={hinge_supply}"

    # Panel dimensions and position
    panel_height = 180
    panel_width = 700
    panel_x = 20
    panel_y = image.shape[0] - panel_height - 20  # Bottom left with 20px margin

    # Draw white background with red border
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), -1)
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 255), 4)

    # Draw title in red
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "ORDER NOTES", (panel_x + 15, panel_y + 40), font, 1.0, (0, 0, 255), 3)

    # Draw size note in red
    cv2.putText(image, f"Size: {size_note}", (panel_x + 15, panel_y + 90), font, 0.8, (0, 0, 255), 2)

    # Draw hinge note in red
    cv2.putText(image, f"{hinge_note}", (panel_x + 15, panel_y + 135), font, 0.8, (0, 0, 255), 2)

    # Save marked image
    cv2.imwrite(output_path, image)
    print(f"[OK] Saved marked image to: {output_path}")

    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify image path")
        print("Usage: python draw_markers.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Derive JSON and output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.dirname(image_path)

    json_path = os.path.join(output_dir, f"{base_name}_two_pass.json")
    output_path = os.path.join(output_dir, f"{base_name}_marked.png")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(json_path):
        print(f"[ERROR] JSON not found: {json_path}")
        print(f"[INFO] Run extract_two_pass.py first")
        sys.exit(1)

    draw_markers_on_page(image_path, json_path, output_path)
