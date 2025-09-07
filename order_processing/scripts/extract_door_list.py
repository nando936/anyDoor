import pdfplumber
import json
import re
import sys
from collections import defaultdict

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def extract_door_list_from_pdf(pdf_path):
    """Extract door list from PDF and create standardized format"""
    
    print(f"Extracting door list from: {pdf_path}")
    print("-" * 50)
    
    # Dictionary to store items by cabinet number
    cabinet_items = defaultdict(list)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from all pages
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
            
            # Split into lines for processing
            lines = all_text.split('\n')
            
            # Process each line to extract door/drawer information
            for line in lines:
                # Skip empty lines and headers
                if not line.strip() or 'DOOR LIST' in line or 'Cabinet Vision' in line:
                    continue
                
                # Pattern to match lines with quantity, dimensions, and cabinet numbers
                # Examples: "6 16 5/8 x 10 5/16 DF N 11 (2), 13 (2), 15 (2)"
                #          "2 13 13/16 x 24 3/4 D N 17, 18"
                
                # Check if line contains dimensions (x or X)
                if ' x ' in line.lower():
                    # Try to parse the line
                    parts = line.strip().split()
                    
                    # Find the quantity (first number)
                    qty_match = re.match(r'^(\d+)', line.strip())
                    if not qty_match:
                        continue
                    
                    total_qty = int(qty_match.group(1))
                    
                    # Find dimensions (numbers x numbers)
                    dim_pattern = r'(\d+(?:\s+\d+/\d+|\s*/\s*\d+)?)\s*[xX]\s*(\d+(?:\s+\d+/\d+|\s*/\s*\d+)?)'
                    dim_match = re.search(dim_pattern, line)
                    if not dim_match:
                        continue
                    
                    width = dim_match.group(1).strip()
                    height = dim_match.group(2).strip()
                    
                    # Determine type (P = Panel/door, DF = drawer front, FF = false front)
                    item_type = 'unknown'
                    if ' P ' in line or ' P\t' in line or line.endswith(' P'):
                        item_type = 'door'
                    elif ' DF ' in line or ' DF\t' in line or line.endswith(' DF'):
                        item_type = 'drawer'
                    elif ' FF ' in line or ' FF\t' in line or line.endswith(' FF'):
                        item_type = 'false_front'
                    
                    # Determine material/style
                    material = 'Unknown'
                    style = 'Unknown'
                    
                    line_upper = line.upper()
                    if 'MDF' in line_upper:
                        material = 'MDF'
                    elif 'PAINT GRADE' in line_upper:
                        material = 'Paint Grade'
                    elif 'WHITE OAK' in line_upper:
                        material = 'White Oak'
                    elif 'MAPLE' in line_upper:
                        material = 'Maple'
                    
                    if 'RAISED PANEL' in line_upper:
                        style = 'Raised Panel'
                    elif 'SOLID WOOD' in line_upper:
                        style = 'Solid Wood'
                    elif 'SQUARE' in line_upper:
                        style = 'Square Raised Panel'
                    elif 'SHAKER' in line_upper:
                        style = 'Shaker'
                    elif 'SLAB' in line_upper:
                        style = 'Slab'
                    
                    # Extract cabinet numbers 
                    # For drawers/false fronts: after N (e.g., "N 11")
                    # For doors: just the number (e.g., "10" not "N 10")
                    if item_type == 'door':
                        # For doors, look for number after L or R
                        cabinet_pattern = r'[LR]\s+(\d+(?:\s*,\s*\d+)*)'
                    else:
                        # For drawers/false fronts: N followed by numbers
                        cabinet_pattern = r'N\s+(\d+(?:\s*\(\d+\))?(?:\s*,\s*\d+(?:\s*\(\d+\))?)*)'
                    
                    cabinet_match = re.search(cabinet_pattern, line)
                    
                    if cabinet_match:
                        cabinet_str = cabinet_match.group(1)
                        
                        # Parse cabinet numbers and their quantities
                        # Examples: "11 (2), 13 (2), 15 (2)" or "17, 18" or "11"
                        cabinet_parts = re.findall(r'(\d+)(?:\s*\((\d+)\))?', cabinet_str)
                        
                        for cab_num, cab_qty in cabinet_parts:
                            # If quantity specified in parentheses, use it
                            # Otherwise, if multiple cabinets listed, assume qty is divided
                            if cab_qty:
                                qty_for_this_cabinet = int(cab_qty)
                            elif len(cabinet_parts) > 1:
                                # Divide total quantity by number of cabinets
                                qty_for_this_cabinet = total_qty // len(cabinet_parts)
                            else:
                                qty_for_this_cabinet = total_qty
                            
                            cabinet_key = f"#{cab_num}"
                            
                            # Check if this exact item already exists for this cabinet
                            existing = False
                            for item in cabinet_items[cabinet_key]:
                                if (item['width'] == width and 
                                    item['height'] == height and 
                                    item['type'] == item_type):
                                    item['qty'] += qty_for_this_cabinet
                                    existing = True
                                    break
                            
                            if not existing:
                                cabinet_items[cabinet_key].append({
                                    'qty': qty_for_this_cabinet,
                                    'width': width,
                                    'height': height,
                                    'type': item_type,
                                    'material': material,
                                    'style': style
                                })
                    else:
                        # No cabinet number, use sequential numbering
                        cabinet_key = f"#AUTO_{len(cabinet_items) + 1}"
                        cabinet_items[cabinet_key].append({
                            'qty': total_qty,
                            'width': width,
                            'height': height,
                            'type': item_type,
                            'material': material,
                            'style': style
                        })
            
            # Convert to standardized format
            standardized_list = []
            for cabinet_num in sorted(cabinet_items.keys(), key=lambda x: (not x.startswith('#AUTO'), x)):
                items = cabinet_items[cabinet_num]
                standardized_list.append({
                    'cabinet': cabinet_num,
                    'items': items
                })
            
            return standardized_list
            
    except Exception as e:
        print(f"[ERROR] Failed to extract door list: {e}")
        return []

def save_door_list(door_list, output_file):
    """Save standardized door list to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(door_list, f, indent=2)
        print(f"[OK] Saved standardized door list to {output_file}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save door list: {e}")
        return False

def print_door_list_summary(door_list):
    """Print a summary of the extracted door list"""
    print("\nExtracted Door List Summary:")
    print("=" * 60)
    
    total_doors = 0
    total_drawers = 0
    total_false_fronts = 0
    
    for cabinet in door_list:
        cabinet_num = cabinet['cabinet']
        items = cabinet['items']
        
        print(f"\n{cabinet_num}:")
        for item in items:
            qty = item['qty']
            width = item['width']
            height = item['height']
            item_type = item['type']
            material = item['material']
            
            if item_type == 'door':
                total_doors += qty
                type_str = "door" if qty == 1 else "doors"
            elif item_type == 'drawer':
                total_drawers += qty
                type_str = "drawer front" if qty == 1 else "drawer fronts"
            elif item_type == 'false_front':
                total_false_fronts += qty
                type_str = "false front" if qty == 1 else "false fronts"
            else:
                type_str = item_type
            
            print(f"  - {qty} {type_str} @ {width} x {height} ({material})")
    
    print("\n" + "=" * 60)
    print(f"Total Doors: {total_doors}")
    print(f"Total Drawer Fronts: {total_drawers}")
    print(f"Total False Fronts: {total_false_fronts}")
    print(f"Total Pieces: {total_doors + total_drawers + total_false_fronts}")

# Main execution
if __name__ == "__main__":
    # Process the Paul Revere order
    pdf_path = "1-302_door_order.pdf"
    output_file = "standard_door_list.json"
    
    # Extract door list from PDF
    door_list = extract_door_list_from_pdf(pdf_path)
    
    if door_list:
        # Print summary
        print_door_list_summary(door_list)
        
        # Save to file
        save_door_list(door_list, output_file)
    else:
        print("[ERROR] No door list extracted")