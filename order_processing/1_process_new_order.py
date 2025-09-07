"""
Main Order Processing Script
This is the primary script for processing new door orders

Workflow:
1. Extract data from user-submitted PDF
2. Create finish door list (HTML with pictures)
3. Generate shop report
4. Generate cut list
5. Convert all to PDFs

CRITICAL: After generating door list, you MUST run:
   python critical_verification.py
   
If verification fails:
   1. Fix the data extraction
   2. Fix this processing script if needed
   3. Regenerate the door list
   4. Run verification again
   5. REPEAT until verification passes with ZERO errors
   
NEVER proceed to production without passing verification!
"""

import json
import sys
import os
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from datetime import datetime

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def load_door_specs(door_style):
    """Load specifications for a door style if available"""
    specs_file = f"door pictures/{door_style} specs.txt"
    specs = {}
    
    if os.path.exists(specs_file):
        with open(specs_file, 'r', encoding='utf-8') as f:
            specs_text = f.read()
            specs['raw_text'] = specs_text
            
            # Parse specific values from specs
            lines = specs_text.lower().split('\n')
            for line in lines:
                if 'stiles and rails' in line and 'wide' in line:
                    # Extract width (e.g., "3" wide")
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)["\']?\s*wide', line)
                    if match:
                        specs['stile_rail_width'] = match.group(1)
                elif 'thick' in line:
                    match = re.search(r'(\d+/\d+|\d+(?:\.\d+)?)["\']?\s*thick', line)
                    if match:
                        specs['material_thickness'] = match.group(1)
                elif 'sticking' in line:
                    match = re.search(r'sticking.*?(\d+/\d+|\d+(?:\.\d+)?)', line)
                    if match:
                        specs['sticking'] = match.group(1)
                elif 'mitre cut' in line:
                    specs['is_mitre_cut'] = True
                elif '5 piece' in line:
                    specs['pieces'] = '5'
            
        print(f"   [OK] Loaded specs for door style {door_style}")
    else:
        print(f"   [!] No specs file found for door style {door_style}")
    
    return specs

def convert_opening_to_finish_size(width_str, height_str, overlay, is_opening_size):
    """Convert opening sizes to finish sizes if needed"""
    if not is_opening_size:
        return width_str, height_str
    
    # Parse the overlay (e.g., "1/2" becomes 0.5)
    def parse_fraction(frac_str):
        if '/' in frac_str:
            parts = frac_str.split('/')
            return float(parts[0]) / float(parts[1])
        return float(frac_str)
    
    # Parse dimensions
    def parse_dimension(dim_str):
        # Handle format like "14 3/8"
        parts = dim_str.strip().split()
        total = 0
        for part in parts:
            if '/' in part:
                num, den = part.split('/')
                total += float(num) / float(den)
            else:
                total += float(part)
        return total
    
    def format_dimension(value):
        # Convert back to fractional format
        whole = int(value)
        fraction = value - whole
        
        if fraction < 0.0625:  # Less than 1/16
            return str(whole) if whole > 0 else "0"
        
        # Find closest common fraction
        fractions = [
            (1/16, "1/16"), (1/8, "1/8"), (3/16, "3/16"), (1/4, "1/4"),
            (5/16, "5/16"), (3/8, "3/8"), (7/16, "7/16"), (1/2, "1/2"),
            (9/16, "9/16"), (5/8, "5/8"), (11/16, "11/16"), (3/4, "3/4"),
            (13/16, "13/16"), (7/8, "7/8"), (15/16, "15/16")
        ]
        
        closest = min(fractions, key=lambda x: abs(x[0] - fraction))
        
        if whole > 0:
            return f"{whole} {closest[1]}"
        else:
            return closest[1]
    
    # Parse overlay
    overlay_value = 0
    if '1/2' in overlay:
        overlay_value = 0.5
    elif '3/4' in overlay:
        overlay_value = 0.75
    elif '1/4' in overlay:
        overlay_value = 0.25
    else:
        overlay_value = parse_fraction(overlay.replace('"', '').strip())
    
    # Convert dimensions
    width_value = parse_dimension(width_str)
    height_value = parse_dimension(height_str)
    
    # Add twice the overlay for opening sizes
    finish_width = width_value + (2 * overlay_value)
    finish_height = height_value + (2 * overlay_value)
    
    return format_dimension(finish_width), format_dimension(finish_height)

def create_finish_door_list_html(customer_info, door_items, door_style, door_specs):
    """Create HTML finish door list with pictures and specifications"""
    
    # Check for door pictures - use absolute paths
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    door_pic_path = os.path.join(base_path, "door pictures", f"{door_style} door pic.JPG").replace('\\', '/')
    door_profile_path = os.path.join(base_path, "door pictures", f"{door_style} door profile.JPG").replace('\\', '/')
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Finish Door List - {customer_info['name']} Job {customer_info['job_number']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ border: 2px solid #333; padding: 15px; margin-bottom: 20px; }}
        h1 {{ font-size: 20px; margin: 0 0 15px 0; text-align: center; }}
        .header-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
        .header-section {{ }}
        .header-section h3 {{ font-size: 14px; margin: 0 0 8px 0; text-decoration: underline; }}
        .header-section p {{ margin: 5px 0; font-size: 12px; }}
        .door-pictures {{ text-align: center; margin-bottom: 20px; }}
        .door-pictures img {{ max-height: 150px; margin: 8px; border: 1px solid #ccc; }}
        .door-pictures p {{ margin: 5px 0; font-size: 12px; font-style: italic; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #333; color: white; padding: 8px; text-align: left; }}
        td {{ border: 1px solid #ccc; padding: 6px; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .editable {{ background: transparent; border: none; width: 100%; padding: 2px; }}
        .editable:focus {{ background: #ffffcc; outline: 1px dashed #999; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FINISH DOOR LIST - STYLE #{door_style}</h1>
        <div class="header-grid">
            <div class="header-section">
                <h3>Customer Information</h3>
                <p><strong>Customer:</strong> {customer_info['name']}</p>
                <p><strong>Address:</strong> {customer_info.get('address', '')}</p>
                <p><strong>Phone:</strong> {customer_info.get('phone', '')}</p>
                <p><strong>Email:</strong> {customer_info.get('email', '')}</p>
                <p><strong>Job:</strong> {customer_info['job_name']}</p>
                <p><strong>Job #:</strong> {customer_info['job_number']}</p>
                <p><strong>Date:</strong> {customer_info['date']}</p>
            </div>
            <div class="header-section">
                <h3>Hardware & Materials</h3>
                <p><strong>Bore/Door Prep:</strong> {'No' if customer_info.get('bore_prep') == False else 'Yes'}</p>
                <p><strong>Hinge Type:</strong> {customer_info.get('hinge_type', 'Standard')}</p>
                <p><strong>Overlay:</strong> {customer_info.get('overlay', '1/2"')}</p>
                <p><strong>Wood Species:</strong> {customer_info.get('wood_species', 'White Oak')}</p>
            </div>
            <div class="header-section">
                <h3>Door Specifications</h3>
                <p><strong>Outside Edge:</strong> {customer_info.get('outside_edge', 'Standard')}</p>
                <p><strong>Sticking (Inside):</strong> {customer_info.get('inside_edge', 'Standard')}</p>
                <p><strong>Panel Cut:</strong> {customer_info.get('panel_cut', 'Standard')}</p>
                <p><strong>Drawer/Front Type:</strong> {customer_info.get('drawer_type', '5 piece')}</p>
                <p><strong>Size Type:</strong> {'Opening Sizes (converted to finish)' if customer_info.get('door_sizes') == 'Opening Sizes' else 'Finish Sizes (as provided)'}</p>
            </div>
        </div>
    </div>
    
    <div class="door-pictures">
        <p><em>Door Style #{door_style} & Hardware</em></p>
        <img src="file:///{door_pic_path}" alt="Door {door_style} Front">
        <img src="file:///{door_profile_path}" alt="Door {door_style} Profile">
        <img src="file:///{os.path.join(base_path, 'door pictures', 'Blum half inche overlay hinge.JPG').replace(os.sep, '/')}" alt="Hinge">
    </div>
    
    <table>
        <thead>
            <tr>
                <th width="10%">Cabinet #</th>
                <th width="8%">Qty</th>
                <th width="12%">Width</th>
                <th width="12%">Height</th>
                <th width="10%">Type</th>
                <th width="15%">Material</th>
                <th width="8%">Style</th>
                <th width="25%">Notes</th>
            </tr>
        </thead>
        <tbody>"""
    
    # Determine if we need to convert opening sizes to finish sizes
    is_opening_size = customer_info.get('door_sizes') == 'Opening Sizes'
    overlay = customer_info.get('overlay', '1/2"')
    
    for item in door_items:
        # Only convert sizes if they are opening sizes
        if is_opening_size:
            finish_width, finish_height = convert_opening_to_finish_size(
                item['width'], item['height'], overlay, True
            )
        else:
            # Use finish sizes as provided
            finish_width, finish_height = item['width'], item['height']
        
        html += f"""
            <tr>
                <td>#{item['cabinet']}</td>
                <td><input class="editable" value="{item['qty']}" contenteditable="true"></td>
                <td><input class="editable" value="{finish_width}" contenteditable="true"></td>
                <td><input class="editable" value="{finish_height}" contenteditable="true"></td>
                <td><input class="editable" value="{item['type']}" contenteditable="true"></td>
                <td><input class="editable" value="{item.get('material', customer_info.get('wood_species', ''))}" contenteditable="true"></td>
                <td><input class="editable" value="{door_style}" contenteditable="true"></td>
                <td><input class="editable" value="{item.get('notes', '')}" contenteditable="true"></td>
            </tr>"""
    
    html += """
        </tbody>
    </table>
    
    <script>
        // Auto-save changes
        document.querySelectorAll('.editable').forEach(input => {
            input.addEventListener('blur', () => {
                console.log('Data changed:', new Date().toLocaleString());
            });
        });
    </script>
</body>
</html>"""
    
    return html

def fraction_to_decimal(fraction_str):
    """Convert fraction string to decimal"""
    if not fraction_str:
        return 0
    
    parts = fraction_str.strip().split()
    total = 0
    
    for part in parts:
        if '/' in part:
            num, den = part.split('/')
            total += float(num) / float(den)
        else:
            total += float(part)
    
    return total

def decimal_to_fraction(decimal):
    """Convert decimal to fraction string"""
    whole = int(decimal)
    remainder = decimal - whole
    
    fractions = [
        (1/16, "1/16"), (1/8, "1/8"), (3/16, "3/16"), (1/4, "1/4"),
        (5/16, "5/16"), (3/8, "3/8"), (7/16, "7/16"), (1/2, "1/2"),
        (9/16, "9/16"), (5/8, "5/8"), (11/16, "11/16"), (3/4, "3/4"),
        (13/16, "13/16"), (7/8, "7/8"), (15/16, "15/16")
    ]
    
    if remainder < 0.03125:
        return f"{whole}"
    
    closest = min(fractions, key=lambda x: abs(x[0] - remainder))
    
    if whole > 0:
        return f"{whole} {closest[1]}"
    else:
        return closest[1]

def generate_shop_report_html(customer_info, door_items, door_style, door_specs=None):
    """Generate shop report HTML with specifications"""
    
    # Get dimensions from specs if available
    stile_width = "3"  # Default from 231 specs
    material_thickness = "13/16"  # Default from 231 specs
    
    if door_specs and door_specs.get('raw_text'):
        # Parse from specs text
        specs_text = door_specs['raw_text'].lower()
        
        # Extract stile/rail width (look for "3" wide" or similar)
        import re
        width_match = re.search(r'stiles? and rails? are (\d+(?:\s*\d+/\d+)?)"?\s*wide', specs_text)
        if width_match:
            stile_width = width_match.group(1).strip('"').strip()
        
        # Extract material thickness
        thickness_match = re.search(r'material is (\d+/\d+)\s*thick', specs_text)
        if thickness_match:
            material_thickness = thickness_match.group(1)
    
    # Calculate materials by wood species
    materials_by_species = {}
    total_hinges = 0
    
    for item in door_items:
        species = item.get('material', customer_info.get('wood_species', 'White Oak'))
        if species not in materials_by_species:
            materials_by_species[species] = {
                'doors': 0,
                'drawers': 0,
                'linear_inches': 0
            }
        
        if item['type'] == 'door':
            materials_by_species[species]['doors'] += item['qty']
            # Count hinges - 2 per door (but check for "no bore" notes)
            if 'no bore' not in item.get('notes', '').lower() and 'no hinge' not in item.get('notes', '').lower():
                total_hinges += item['qty'] * 2
        elif item['type'] == 'drawer':
            materials_by_species[species]['drawers'] += item['qty']
        
        # Calculate linear inches for this item
        if item['type'] != 'false_front':
            width = fraction_to_decimal(item['width'])
            height = fraction_to_decimal(item['height'])
            qty = item['qty']
            # For mitre cut doors, stiles and rails are exact dimensions
            materials_by_species[species]['linear_inches'] += qty * 2 * (width + height)
    
    # Calculate total doors and drawers
    total_doors = sum(mat['doors'] for mat in materials_by_species.values())
    total_drawers = sum(mat['drawers'] for mat in materials_by_species.values())
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Shop Report - {customer_info['job_name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ border: 2px solid black; padding: 15px; margin-bottom: 20px; }}
        h1 {{ font-size: 24px; margin: 0 0 15px 0; }}
        h2 {{ font-size: 18px; margin: 15px 0 10px 0; text-decoration: underline; }}
        .section {{ margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid black; padding: 8px; text-align: left; font-size: 14px; }}
        th {{ background-color: #f0f0f0; font-weight: bold; }}
        .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 14px; }}
        p {{ font-size: 14px; margin: 8px 0; }}
        @media print {{ 
            body {{ margin: 10px; }}
            .page-break {{ page-break-after: always; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SHOP REPORT</h1>
        <div class="info-grid">
            <div><strong>Date:</strong> {customer_info['date']}</div>
            <div><strong>Order #:</strong> {customer_info['job_number']}</div>
            <div><strong>Customer Name:</strong> {customer_info['name']}</div>
            <div><strong>Job Name:</strong> {customer_info['job_name']}</div>
            <div><strong>Start Date:</strong> _____________</div>
            <div><strong>Finish Date:</strong> _____________</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Door Specifications</h2>
        <table>
            <tr>
                <td width="50%"><strong>Wood Species:</strong> {customer_info.get('wood_species', 'Mixed (Paint Grade & White Oak)')}</td>
                <td width="50%"><strong>Door Style:</strong> #{door_style}</td>
            </tr>
            <tr>
                <td><strong>Hinge Type:</strong> {customer_info.get('hinge_type', 'Standard')}</td>
                <td><strong>Overlay:</strong> {customer_info.get('overlay', '1/2" Overlay')}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Materials Required</h2>
        <table>
            <tr>
                <th>Item Type</th>
                <th>Quantity</th>
                <th>Size</th>
                <th>Material</th>
            </tr>"""
    
    # Add doors by species
    for species, counts in materials_by_species.items():
        if counts['doors'] > 0:
            html += f"""
            <tr>
                <td>Doors</td>
                <td>{counts['doors']}</td>
                <td>Various</td>
                <td>{species}</td>
            </tr>"""
    
    # Add drawer fronts by species
    for species, counts in materials_by_species.items():
        if counts['drawers'] > 0:
            html += f"""
            <tr>
                <td>Drawer Fronts</td>
                <td>{counts['drawers']}</td>
                <td>Various</td>
                <td>{species}</td>
            </tr>"""
    
    # Add stile sticks by species
    for species, counts in materials_by_species.items():
        if counts['linear_inches'] > 0:
            # Calculate 8-foot pieces needed
            eight_foot_pieces = int((counts['linear_inches'] / 96) + 0.999)  # Round up
            html += f"""
            <tr>
                <td>Stile Sticks</td>
                <td>{eight_foot_pieces} pcs</td>
                <td>{stile_width}" x {material_thickness}" x 8'</td>
                <td>{species}</td>
            </tr>"""
    
    # Add hinges as separate line item
    if total_hinges > 0:
        html += f"""
            <tr>
                <td>Hinges</td>
                <td>{total_hinges}</td>
                <td>-</td>
                <td>{customer_info.get('hinge_type', 'Blum Soft Close Frameless 1/2"OL')}</td>
            </tr>"""
    
    html += f"""
        </table>
    </div>
    
    <div class="section">
        <h2>Door Style #{door_style} & Hardware</h2>
        <div style="display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap; justify-content: center;">
            <div style="text-align: center;">
                <img src="file:///{os.path.abspath('door pictures/' + door_style + ' door pic.JPG').replace(os.sep, '/')}" 
                     alt="Door Style {door_style}" 
                     style="max-width: 200px; height: auto; border: 1px solid #ccc;">
                <p style="margin-top: 5px;"><strong>Door Picture</strong></p>
            </div>
            <div style="text-align: center;">
                <img src="file:///{os.path.abspath('door pictures/' + door_style + ' door profile.JPG').replace(os.sep, '/')}" 
                     alt="Door Profile {door_style}" 
                     style="max-width: 200px; height: auto; border: 1px solid #ccc;">
                <p style="margin-top: 5px;"><strong>Door Profile</strong></p>
            </div>
            <div style="text-align: center;">
                <img src="file:///{os.path.abspath('door pictures/Blum half inche overlay hinge.JPG').replace(os.sep, '/')}" 
                     alt="Hinge" 
                     style="max-width: 200px; height: auto; border: 1px solid #ccc;">
                <p style="margin-top: 5px;"><strong>Hinge - Blum 1/2" Overlay</strong></p>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    return html

def generate_cut_list_html(customer_info, door_items, door_style, door_specs=None):
    """Generate cut list HTML"""
    
    # Get dimensions from specs if available
    stile_width = "3"  # Default from 231 specs
    material_thickness = "13/16"  # Default from 231 specs
    is_mitre_cut = True  # Default for 231
    
    if door_specs and door_specs.get('raw_text'):
        # Parse from specs text
        specs_text = door_specs['raw_text'].lower()
        
        # Check if it's a mitre cut door
        if 'mitre cut' in specs_text:
            is_mitre_cut = True
        
        # Extract stile/rail width
        import re
        width_match = re.search(r'stiles? and rails? are (\d+(?:\s*\d+/\d+)?)"?\s*wide', specs_text)
        if width_match:
            stile_width = width_match.group(1).strip('"').strip()
    
    cuts = []
    
    for idx, item in enumerate(door_items, 1):
        if item['type'] == 'door' or item['type'] == 'drawer':
            width = fraction_to_decimal(item['width'])
            height = fraction_to_decimal(item['height'])
            qty = item['qty']
            material = item.get('material', customer_info.get('wood_species', 'White Oak'))
            
            if is_mitre_cut:
                # For mitre cut doors, stiles and rails are exact outside dimensions
                # Stiles = door height
                stile_length = height
                # Rails = door width
                rail_length = width
            else:
                # For cope and stick doors (traditional calculation)
                # Stiles (height + 1/4")
                stile_length = height + 0.25
                # Rails (width - 4 1/2" + 3/4")
                rail_length = width - 3.75
            
            cuts.append({
                'qty': qty * 2,
                'length': stile_length,
                'width': f'{stile_width}"',
                'part': 'STILES',
                'label': f"Cabinet #{item['cabinet']}",
                'desc': f"{qty} {item['type']} @ {item['width']} x {item['height']}",
                'material': material
            })
            
            cuts.append({
                'qty': qty * 2,
                'length': rail_length,
                'width': f'{stile_width}"',
                'part': 'RAILS',
                'label': f"Cabinet #{item['cabinet']}",
                'desc': f"{qty} {item['type']} @ {item['width']} x {item['height']}",
                'material': material
            })
    
    # Sort by length (longest first)
    cuts.sort(key=lambda x: x['length'], reverse=True)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cut List - {customer_info['job_name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; font-size: 11px; }}
        h1 {{ font-size: 16px; }}
        h2 {{ font-size: 14px; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid black; padding: 3px; text-align: left; }}
        th {{ background-color: #f0f0f0; font-weight: bold; }}
        @media print {{ body {{ margin: 10px; }} }}
    </style>
</head>
<body>
    <h1>CUT LIST - {customer_info['job_name']} ({customer_info['job_number']})</h1>
    <p><strong>Date:</strong> {customer_info['date']} | <strong>Customer:</strong> {customer_info['name']} | <strong>Door Style:</strong> #{door_style}</p>
    <p><strong>Wood Species:</strong> {customer_info.get('wood_species', 'White Oak')} | <strong>Cut Type:</strong> {'Mitre Cut' if is_mitre_cut else 'Cope & Stick'}</p>
    
    <table>
        <tr>
            <th width="8%">QTY</th>
            <th width="12%">LENGTH</th>
            <th width="10%">WIDTH</th>
            <th width="10%">PART</th>
            <th width="12%">CABINET</th>
            <th width="15%">MATERIAL</th>
            <th width="33%">DESCRIPTION</th>
        </tr>"""
    
    total_stiles = 0
    total_rails = 0
    materials_totals = {}
    
    for cut in cuts:
        length_str = decimal_to_fraction(cut['length']) + '"'
        material = cut.get('material', 'White Oak')
        
        # Track totals by material
        if material not in materials_totals:
            materials_totals[material] = {'stiles': 0, 'rails': 0}
        
        if cut['part'] == 'STILES':
            materials_totals[material]['stiles'] += cut['qty']
            total_stiles += cut['qty']
        else:
            materials_totals[material]['rails'] += cut['qty']
            total_rails += cut['qty']
        
        html += f"""
        <tr>
            <td>{cut['qty']}</td>
            <td>{length_str}</td>
            <td>{cut['width']}</td>
            <td>{cut['part']}</td>
            <td>{cut['label']}</td>
            <td>{material}</td>
            <td style="font-size: 10px;">{cut['desc']}</td>
        </tr>"""
    
    html += f"""
    </table>
    
    <div style="margin-top: 20px;">
        <h2>Summary</h2>
        <p><strong>Total Stiles:</strong> {total_stiles} pieces</p>
        <p><strong>Total Rails:</strong> {total_rails} pieces</p>
        <p><strong>Total Pieces:</strong> {total_stiles + total_rails} pieces</p>
        
        <h2>By Material</h2>"""
    
    for material, counts in materials_totals.items():
        html += f"""
        <p><strong>{material}:</strong> {counts['stiles']} stiles, {counts['rails']} rails = {counts['stiles'] + counts['rails']} pieces</p>"""
    
    html += """
    </div>
</body>
</html>"""
    
    return html

def html_to_pdf(html_content, pdf_path):
    """Convert HTML to PDF using Selenium"""
    try:
        # Save HTML to temp file
        temp_html = "temp_report.html"
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Load the HTML file
        file_url = f"file:///{os.path.abspath(temp_html).replace(os.sep, '/')}"
        driver.get(file_url)
        time.sleep(1)
        
        # Generate PDF
        pdf_data = driver.execute_cdp_cmd('Page.printToPDF', {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'scale': 1.0,
            'paperWidth': 8.5,
            'paperHeight': 11,
            'marginTop': 0.5,
            'marginBottom': 0.5,
            'marginLeft': 0.5,
            'marginRight': 0.5
        })
        
        # Save PDF
        with open(pdf_path, 'wb') as f:
            f.write(base64.b64decode(pdf_data['data']))
        
        driver.quit()
        
        # Clean up temp file
        if os.path.exists(temp_html):
            os.remove(temp_html)
        
        return True
    except Exception as e:
        print(f"[ERROR] PDF conversion failed: {e}")
        return False

def process_order(customer_info, door_items, door_style, output_prefix):
    """Main function to process an order"""
    
    print(f"Processing order for {customer_info['name']}")
    print("=" * 60)
    
    # Load door specifications if available
    print("\n0. Loading Door Specifications...")
    door_specs = load_door_specs(door_style)
    
    # 1. Create finish door list
    print("\n1. Creating Finish Door List...")
    door_list_html = create_finish_door_list_html(customer_info, door_items, door_style, door_specs)
    door_list_file = f"{output_prefix}_door_list.html"
    with open(door_list_file, 'w', encoding='utf-8') as f:
        f.write(door_list_html)
    print(f"   [OK] Created: {door_list_file}")
    
    # 2. Generate shop report
    print("\n2. Generating Shop Report...")
    shop_report_html = generate_shop_report_html(customer_info, door_items, door_style, door_specs)
    shop_report_file = f"{output_prefix}_shop_report.html"
    with open(shop_report_file, 'w', encoding='utf-8') as f:
        f.write(shop_report_html)
    print(f"   [OK] Created: {shop_report_file}")
    
    # 3. Generate cut list
    print("\n3. Generating Cut List...")
    cut_list_html = generate_cut_list_html(customer_info, door_items, door_style, door_specs)
    cut_list_file = f"{output_prefix}_cut_list.html"
    with open(cut_list_file, 'w', encoding='utf-8') as f:
        f.write(cut_list_html)
    print(f"   [OK] Created: {cut_list_file}")
    
    # 4. Convert all to PDFs
    print("\n4. Converting to PDFs...")
    html_to_pdf(door_list_html, f"{output_prefix}_door_list.pdf")
    print(f"   [OK] Created: {output_prefix}_door_list.pdf")
    
    html_to_pdf(shop_report_html, f"{output_prefix}_shop_report.pdf")
    print(f"   [OK] Created: {output_prefix}_shop_report.pdf")
    
    html_to_pdf(cut_list_html, f"{output_prefix}_cut_list.pdf")
    print(f"   [OK] Created: {output_prefix}_cut_list.pdf")
    
    # 5. Save JSON for future reference
    print("\n5. Saving JSON data...")
    json_data = {
        'customer_info': customer_info,
        'door_style': door_style,
        'door_items': door_items,
        'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    json_file = f"{output_prefix}_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"   [OK] Created: {json_file}")
    
    print("\n" + "=" * 60)
    print("Order processing complete!")
    print(f"All files created with prefix: {output_prefix}_*")

# Example usage
if __name__ == "__main__":
    # Example customer info (would be extracted from PDF)
    customer_info = {
        'name': 'Sample Customer',
        'job_name': 'Kitchen Remodel',
        'job_number': '999',
        'date': datetime.now().strftime('%m/%d/%Y'),
        'wood_species': 'White Oak',
        'hinge_type': 'Blum Soft Close',
        'overlay': '1/2" Overlay'
    }
    
    # Example door items (would be extracted from PDF)
    door_items = [
        {'line': 1, 'cabinet': 1, 'qty': 2, 'width': '14 3/8', 'height': '24 3/4', 'type': 'door', 'notes': ''},
        {'line': 2, 'cabinet': 2, 'qty': 1, 'width': '18', 'height': '30', 'type': 'door', 'notes': 'Glass insert'},
    ]
    
    door_style = '231'
    output_prefix = 'sample_order'
    
    print("This is the main order processing script.")
    print("To process a real order, update the customer_info and door_items.")
    print("\nExample processing with sample data:")
    
    # process_order(customer_info, door_items, door_style, output_prefix)