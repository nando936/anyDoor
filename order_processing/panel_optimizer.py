"""
Panel Optimizer Module
Optimizes panel cutting layouts for sheet goods (4x8 sheets)
Generates visual diagrams and cutting instructions
Portrait layout: One sheet pattern per page with instructions below
"""

import json
import math
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Panel:
    """Represents a panel to be cut"""
    cabinet: str
    width: float
    height: float
    material: str
    quantity: int = 1
    grain_direction: str = "vertical"  # vertical or horizontal
    label: str = ""
    
    def area(self) -> float:
        return self.width * self.height
    
    def can_rotate(self) -> bool:
        """Check if panel can be rotated (grain doesn't matter)"""
        return self.grain_direction == "none"

@dataclass
class PlacedPanel:
    """A panel that has been placed on a sheet"""
    panel: Panel
    x: float
    y: float
    rotated: bool = False
    
    @property
    def width(self) -> float:
        return self.panel.height if self.rotated else self.panel.width
    
    @property
    def height(self) -> float:
        return self.panel.width if self.rotated else self.panel.height
    
    @property
    def x2(self) -> float:
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        return self.y + self.height

@dataclass
class Sheet:
    """Represents a sheet with placed panels"""
    width: float = 48.0
    height: float = 96.0
    panels: List[PlacedPanel] = None
    material: str = ""
    sheet_number: int = 1
    
    def __post_init__(self):
        if self.panels is None:
            self.panels = []
    
    @property
    def used_area(self) -> float:
        return sum(p.panel.area() for p in self.panels)
    
    @property
    def efficiency(self) -> float:
        total_area = self.width * self.height
        if total_area == 0:
            return 0
        return (self.used_area / total_area) * 100
    
    @property
    def waste_area(self) -> float:
        return (self.width * self.height) - self.used_area

class PanelOptimizer:
    """
    Optimizes panel placement on sheets using First Fit Decreasing Height algorithm
    with shelf packing for better efficiency
    """
    
    def __init__(self, kerf: float = 0.125):
        self.kerf = kerf  # Blade width
        self.sheet_width = 48.0
        self.sheet_height = 96.0
        self.min_waste_size = 6.0  # Minimum size to save waste pieces
        
    def optimize(self, panels: List[Panel]) -> List[Sheet]:
        """
        Optimize panel placement on sheets
        Returns list of sheets with placed panels
        """
        # Expand panels by quantity
        expanded_panels = []
        for panel in panels:
            for i in range(panel.quantity):
                p = Panel(
                    cabinet=panel.cabinet,
                    width=panel.width,
                    height=panel.height,
                    material=panel.material,
                    quantity=1,
                    grain_direction=panel.grain_direction,
                    label=f"{panel.cabinet}-{i+1}" if panel.quantity > 1 else panel.cabinet
                )
                expanded_panels.append(p)
        
        # Sort panels by area (largest first)
        expanded_panels.sort(key=lambda p: p.area(), reverse=True)
        
        # Group by material
        by_material = {}
        for panel in expanded_panels:
            if panel.material not in by_material:
                by_material[panel.material] = []
            by_material[panel.material].append(panel)
        
        # Optimize each material group
        all_sheets = []
        sheet_number = 1
        
        for material, material_panels in by_material.items():
            sheets = self._pack_panels(material_panels, material, sheet_number)
            all_sheets.extend(sheets)
            sheet_number += len(sheets)
        
        return all_sheets
    
    def _pack_panels(self, panels: List[Panel], material: str, start_sheet_num: int) -> List[Sheet]:
        """Pack panels onto sheets using shelf algorithm"""
        sheets = []
        remaining_panels = panels.copy()
        sheet_num = start_sheet_num
        
        while remaining_panels:
            sheet = Sheet(
                width=self.sheet_width,
                height=self.sheet_height,
                material=material,
                sheet_number=sheet_num
            )
            
            # Try to fill this sheet
            placed = self._fill_sheet(sheet, remaining_panels)
            
            if placed:
                sheets.append(sheet)
                sheet_num += 1
            else:
                # Couldn't place any panel (too large)
                print(f"Warning: Panel too large for sheet: {remaining_panels[0]}")
                remaining_panels.pop(0)
        
        return sheets
    
    def _fill_sheet(self, sheet: Sheet, panels: List[Panel]) -> bool:
        """Fill a sheet with panels using shelf packing"""
        placed_any = False
        current_y = 0
        
        while current_y < sheet.height and panels:
            # Create a shelf at current_y
            shelf_height = 0
            current_x = 0
            panels_to_remove = []
            
            # Try to fill this shelf
            for i, panel in enumerate(panels):
                # Try normal orientation
                if self._can_place(sheet, panel.width, panel.height, current_x, current_y):
                    placed = PlacedPanel(
                        panel=panel,
                        x=current_x,
                        y=current_y,
                        rotated=False
                    )
                    sheet.panels.append(placed)
                    panels_to_remove.append(i)
                    current_x += panel.width + self.kerf
                    shelf_height = max(shelf_height, panel.height)
                    placed_any = True
                    
                # Try rotated if grain allows
                elif panel.can_rotate() and self._can_place(sheet, panel.height, panel.width, current_x, current_y):
                    placed = PlacedPanel(
                        panel=panel,
                        x=current_x,
                        y=current_y,
                        rotated=True
                    )
                    sheet.panels.append(placed)
                    panels_to_remove.append(i)
                    current_x += panel.height + self.kerf
                    shelf_height = max(shelf_height, panel.width)
                    placed_any = True
                
                if current_x >= sheet.width:
                    break
            
            # Remove placed panels
            for i in reversed(panels_to_remove):
                panels.pop(i)
            
            # Move to next shelf
            if shelf_height > 0:
                current_y += shelf_height + self.kerf
            else:
                break
        
        return placed_any
    
    def _can_place(self, sheet: Sheet, width: float, height: float, x: float, y: float) -> bool:
        """Check if a panel can be placed at position"""
        # Check sheet boundaries
        if x + width > sheet.width or y + height > sheet.height:
            return False
        
        # Check overlap with existing panels
        for placed in sheet.panels:
            if not (x + width <= placed.x or x >= placed.x2 or
                   y + height <= placed.y or y >= placed.y2):
                return False
        
        return True
    
    def generate_cutting_instructions(self, sheet: Sheet) -> List[str]:
        """Generate step-by-step cutting instructions for a sheet"""
        instructions = []
        
        if not sheet.panels:
            return ["No panels on this sheet"]
        
        # Sort panels by position for logical cutting order
        panels = sorted(sheet.panels, key=lambda p: (p.y, p.x))
        
        # Group panels by Y position (horizontal cuts first)
        y_groups = {}
        for panel in panels:
            y_key = round(panel.y, 2)
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append(panel)
        
        cut_num = 1
        
        # Generate cutting sequence
        for y_pos in sorted(y_groups.keys()):
            if y_pos > 0:
                instructions.append(f"{cut_num}. Rip at {self._format_dimension(y_pos)} from bottom edge")
                cut_num += 1
            
            # Sort panels in this row by X position
            row_panels = sorted(y_groups[y_pos], key=lambda p: p.x)
            
            for panel in row_panels:
                if panel.x > 0:
                    instructions.append(f"{cut_num}. Cross-cut at {self._format_dimension(panel.x)} from left edge")
                    cut_num += 1
                
                piece_name = f"Panel {panel.panel.label} - Cabinet #{panel.panel.cabinet}"
                dimensions = f"{self._format_dimension(panel.width)} x {self._format_dimension(panel.height)}"
                if panel.rotated:
                    instructions.append(f"    → {piece_name} ({dimensions}) [ROTATED]")
                else:
                    instructions.append(f"    → {piece_name} ({dimensions})")
        
        # Add waste calculation
        waste = sheet.waste_area
        if waste > self.min_waste_size * self.min_waste_size:
            waste_width = self._format_dimension(math.sqrt(waste))
            instructions.append(f"\nUsable waste: Approximately {waste_width} square")
            instructions.append("(Save for drawer bottoms or test pieces)")
        
        return instructions
    
    def _format_dimension(self, value: float) -> str:
        """Format dimension as fraction string"""
        # Convert to nearest 1/16"
        sixteenths = round(value * 16)
        
        if sixteenths % 16 == 0:
            return f"{sixteenths // 16}\""
        
        whole = sixteenths // 16
        fraction = sixteenths % 16
        
        # Simplify fraction
        if fraction % 8 == 0:
            fraction_str = "1/2"
        elif fraction % 4 == 0:
            fraction_str = f"{fraction // 4}/4"
        elif fraction % 2 == 0:
            fraction_str = f"{fraction // 2}/8"
        else:
            fraction_str = f"{fraction}/16"
        
        if whole > 0:
            return f"{whole} {fraction_str}\""
        else:
            return f"{fraction_str}\""
    
    def generate_svg_diagram(self, sheet: Sheet, scale: float = 0.125) -> str:
        """Generate SVG diagram of sheet layout - rotated 90 degrees for portrait printing"""
        # For 8.5x11 paper with 0.5" margins: usable width = 7.5"
        # Sheet is 48"x96", rotated becomes 96"x48"
        # Scale: 96" sheet width / 7.5" page width = 1:12.8 scale
        # Using 1:13 scale for clean numbers
        scale = 0.078125  # This gives us 7.5" width for 96" sheet
        
        # Original dimensions
        orig_width = sheet.width * scale
        orig_height = sheet.height * scale
        
        # Rotated dimensions (swap width and height for display)
        display_width = orig_height  # 96" becomes width
        display_height = orig_width   # 48" becomes height
        
        # SVG dimensions in pixels (assuming 72 dpi)
        svg_width = display_width * 72
        svg_height = display_height * 72
        
        svg = f'''<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .sheet {{ fill: #f5f5f5; stroke: #333; stroke-width: 2; }}
            .panel {{ fill: #e8f4f8; stroke: #2196F3; stroke-width: 1; }}
            .grain-v {{ fill: url(#grainV); }}
            .grain-h {{ fill: url(#grainH); }}
            .label {{ font-family: Arial; font-size: 12px; text-anchor: middle; }}
            .dimension {{ font-family: Arial; font-size: 10px; fill: #666; }}
        </style>
        
        <!-- Define wood grain patterns -->
        <defs>
            <pattern id="grainV" x="0" y="0" width="4" height="10" patternUnits="userSpaceOnUse">
                <line x1="2" y1="0" x2="2" y2="10" stroke="#d4a373" stroke-width="0.5" opacity="0.3"/>
            </pattern>
            <pattern id="grainH" x="0" y="0" width="10" height="4" patternUnits="userSpaceOnUse">
                <line x1="0" y1="2" x2="10" y2="2" stroke="#d4a373" stroke-width="0.5" opacity="0.3"/>
            </pattern>
        </defs>
        
        
        <!-- Apply rotation transform to entire sheet -->
        <g transform="rotate(90 {svg_width/2} {svg_height/2}) translate({(svg_width-svg_height)/2} {-(svg_width-svg_height)/2})">
            <!-- Sheet outline -->
            <rect x="10" y="10" width="{orig_width * 72 - 20}" height="{orig_height * 72 - 20}" class="sheet"/>
            '''
        
        # Draw panels with rotation (flip Y axis for bottom-left origin)
        # Calculate the actual drawing scale to fit within the margins
        margin = 10  # pixels
        sheet_width_pixels = orig_width * 72 - 2 * margin  # 250 pixels for 48" sheet
        sheet_height_pixels = orig_height * 72 - 2 * margin  # 520 pixels for 96" sheet
        
        # Actual scale factors for drawing within margins
        # This ensures panels are scaled to fit within the visible sheet rectangle
        x_scale = sheet_width_pixels / sheet.width
        y_scale = sheet_height_pixels / sheet.height
        
        for placed in sheet.panels:
            # Scale panel positions to pixels within margins
            px = margin + placed.x * x_scale
            # Flip Y coordinate: SVG uses top-left, we want bottom-left
            py = margin + (sheet.height - placed.y - placed.height) * y_scale
            pw = placed.width * x_scale
            ph = placed.height * y_scale
            
            # Determine grain class
            grain_class = "grain-h" if placed.rotated else "grain-v"
            
            svg += f'''
            <!-- Panel {placed.panel.label} -->
            <rect x="{px}" y="{py}" width="{pw}" height="{ph}" 
                  class="panel {grain_class}"/>
            <g transform="rotate(-90 {px + pw/2} {py + ph/2})">
                <text x="{px + pw/2}" y="{py + ph/2}" class="label">
                    {placed.panel.label}
                </text>
                <text x="{px + pw/2}" y="{py + ph/2 + 12}" class="dimension">
                    {self._format_dimension(placed.width)} x {self._format_dimension(placed.height)}
                </text>
            </g>'''
        
        # Close the rotation group
        svg += '''
        </g>
        
        <!-- Sheet info (not rotated) -->
        <text x="{}" y="{}" class="dimension" text-anchor="middle">
            {}\" x {}\" Sheet - Efficiency: {:.1f}%
        </text>
        </svg>'''.format(
            svg_width/2, svg_height + 50,
            sheet.width, sheet.height, sheet.efficiency
        )
        
        return svg


def generate_optimizer_report(customer_info: Dict, door_items: List[Dict], 
                             door_style: str, door_specs: Dict = None) -> str:
    """Generate complete optimizer report HTML with one sheet per page"""
    
    from datetime import datetime
    
    # Extract panels from door items
    panels = extract_panels_from_doors(door_items, door_specs)
    
    # Initialize optimizer
    optimizer = PanelOptimizer(kerf=0.125)
    
    # Optimize panel placement
    sheets = optimizer.optimize(panels)
    
    # Generate HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Panel Optimizer Report - {customer_info['job_name']}</title>
    <style>
        @page {{
            size: letter portrait;
            margin: 0.5in;
        }}
        
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        
        .page {{
            page-break-after: always;
            min-height: 10in;
            padding: 0.5in;
            position: relative;
        }}
        
        .page:last-child {{
            page-break-after: auto;
        }}
        
        .header {{
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        h1 {{
            font-size: 20px;
            margin: 0 0 10px 0;
        }}
        
        h2 {{
            font-size: 16px;
            margin: 15px 0 10px 0;
            background: #333;
            color: white;
            padding: 5px 10px;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 12px;
        }}
        
        .diagram {{
            border: 1px solid #ccc;
            padding: 10px;
            margin: 20px auto;
            background: white;
            display: flex;
            justify-content: center;
        }}
        
        .instructions {{
            background: #f9f9f9;
            border: 1px solid #ddd;
            padding: 15px;
            margin-top: 20px;
            font-size: 12px;
        }}
        
        .instructions ol {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        
        .instructions li {{
            margin: 5px 0;
            line-height: 1.4;
        }}
        
        .summary {{
            background: #e8f4f8;
            border: 1px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            font-size: 14px;
        }}
        
        .stat-box {{
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }}
        
        .stat-label {{
            font-size: 11px;
            color: #666;
            margin-top: 5px;
        }}
        
        .materials-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .materials-table th, .materials-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        
        .materials-table th {{
            background: #f0f0f0;
        }}
        
        @media print {{
            .page {{
                margin: 0;
                padding: 0;
            }}
        }}
    </style>
</head>
<body>
"""
    
    # Summary page
    total_sheets = len(sheets)
    total_panels = sum(len(s.panels) for s in sheets)
    avg_efficiency = sum(s.efficiency for s in sheets) / len(sheets) if sheets else 0
    total_waste = sum(s.waste_area() for s in sheets)
    
    # Group sheets by material
    by_material = {}
    for sheet in sheets:
        if sheet.material not in by_material:
            by_material[sheet.material] = []
        by_material[sheet.material].append(sheet)
    
    html += f"""
    <div class="page">
        <div class="header">
            <h1>PANEL OPTIMIZATION REPORT</h1>
            <div class="info-grid">
                <div><strong>Customer:</strong> {customer_info['name']}</div>
                <div><strong>Job:</strong> {customer_info['job_name']} (#{customer_info['job_number']})</div>
                <div><strong>Date:</strong> {customer_info['date']}</div>
                <div><strong>Door Style:</strong> #{door_style}</div>
            </div>
        </div>
        
        <div class="summary">
            <h2 style="margin-top: 0;">OPTIMIZATION SUMMARY</h2>
            <div class="summary-grid">
                <div class="stat-box">
                    <div class="stat-value">{total_sheets}</div>
                    <div class="stat-label">TOTAL SHEETS</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{total_panels}</div>
                    <div class="stat-label">PANELS CUT</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{avg_efficiency:.1f}%</div>
                    <div class="stat-label">AVG EFFICIENCY</div>
                </div>
            </div>
        </div>
        
        <h2>MATERIAL BREAKDOWN</h2>
        <table class="materials-table">
            <tr>
                <th>Material</th>
                <th>Sheets Required</th>
                <th>Total Area Used</th>
                <th>Efficiency</th>
                <th>Waste (sq ft)</th>
            </tr>"""
    
    for material, material_sheets in by_material.items():
        sheets_count = len(material_sheets)
        total_used = sum(s.used_area() for s in material_sheets)
        total_area = sheets_count * 48 * 96
        efficiency = (total_used / total_area * 100) if total_area > 0 else 0
        waste = sum(s.waste_area() for s in material_sheets) / 144  # Convert to sq ft
        
        html += f"""
            <tr>
                <td>{material}</td>
                <td>{sheets_count}</td>
                <td>{total_used / 144:.1f} sq ft</td>
                <td>{efficiency:.1f}%</td>
                <td>{waste:.1f}</td>
            </tr>"""
    
    html += """
        </table>
        
        <div style="margin-top: 30px; font-size: 11px; color: #666;">
            <p><strong>Notes:</strong></p>
            <ul>
                <li>All dimensions include 1/8" kerf (blade width)</li>
                <li>Grain direction is preserved where specified</li>
                <li>Panels are optimized for minimal waste</li>
                <li>Save larger waste pieces for drawer bottoms or test cuts</li>
            </ul>
        </div>
    </div>"""
    
    # Individual sheet pages
    for sheet in sheets:
        instructions = optimizer.generate_cutting_instructions(sheet)
        svg_diagram = optimizer.generate_svg_diagram(sheet)
        
        html += f"""
    <div class="page">
        <div class="header">
            <h1>SHEET #{sheet.sheet_number} - {sheet.material}</h1>
            <div class="info-grid">
                <div><strong>Sheet Size:</strong> {sheet.width}" x {sheet.height}" (4' x 8')</div>
                <div><strong>Panels on Sheet:</strong> {len(sheet.panels)}</div>
                <div><strong>Material Usage:</strong> {sheet.efficiency:.1f}%</div>
                <div><strong>Waste:</strong> {sheet.waste_area() / 144:.2f} sq ft</div>
            </div>
        </div>
        
        <h2>CUTTING DIAGRAM</h2>
        <div class="diagram">
            {svg_diagram}
        </div>
        
        <div class="instructions">
            <h2 style="background: #666;">CUTTING INSTRUCTIONS</h2>
            <ol>"""
        
        for instruction in instructions:
            if instruction.startswith("    →"):
                html += f"""
                <ul style="margin-top: 5px;">
                    <li style="list-style: none;">{instruction}</li>
                </ul>"""
            elif instruction.startswith("(") or instruction.startswith("Usable"):
                html += f"""
            </ol>
            <p style="margin-top: 10px; font-style: italic;">{instruction}</p>
            <ol start="{len([i for i in instructions if not i.startswith(' ') and not i.startswith('(') and not i.startswith('Usable')]) + 1}">"""
            else:
                html += f"""
                <li>{instruction}</li>"""
        
        html += """
            </ol>
        </div>
    </div>"""
    
    # Footer
    html += f"""
    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ccc; text-align: center; font-size: 12px; color: #666;">
        <p>anyDoor Panel Optimizer - Generated: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}</p>
    </div>
</body>
</html>"""
    
    return html


def extract_panels_from_doors(door_items: List[Dict], door_specs: Dict = None) -> List[Panel]:
    """Extract panel information from door items"""
    panels = []
    
    # Determine if it's cope & stick or mitre cut from specs
    is_cope_and_stick = False
    is_mitre_cut = True  # Default
    sticking = 0.5
    stile_width = 3.0  # Default
    
    if door_specs and door_specs.get('raw_text'):
        specs_text = door_specs['raw_text'].lower()
        if 'cope and stick' in specs_text:
            is_cope_and_stick = True
            is_mitre_cut = False
            stile_width = 2.375  # From style 103
        elif 'mitre cut' in specs_text:
            is_mitre_cut = True
            stile_width = 3.0  # From style 231
    
    # Process each door item
    for item in door_items:
        if item['type'] != 'door':
            continue  # Only process doors (they have panels)
        
        # Parse dimensions
        width = fraction_to_decimal(item['width'])
        height = fraction_to_decimal(item['height'])
        
        # Calculate panel dimensions based on door type
        if is_cope_and_stick:
            panel_width = width - (stile_width + ((sticking * 2) - 0.125))
            panel_height = height - (stile_width + ((sticking * 2) - 0.125))
        else:  # Mitre cut
            panel_width = width - (2 * stile_width) + (sticking - 0.125)
            panel_height = height - (2 * stile_width) + (sticking - 0.125)
        
        # Determine grain direction (typically vertical for doors)
        grain = "vertical" if panel_height > panel_width else "horizontal"
        
        # Create panel object
        panel = Panel(
            cabinet=str(item['cabinet']),
            width=panel_width,
            height=panel_height,
            material=item.get('material', 'Unknown'),
            quantity=item['qty'],
            grain_direction=grain,
            label=f"Cab{item['cabinet']}"
        )
        panels.append(panel)
    
    return panels


def fraction_to_decimal(fraction_str: str) -> float:
    """Convert fraction string to decimal"""
    if not fraction_str:
        return 0.0
    
    # Remove quotes and extra spaces
    fraction_str = fraction_str.strip().strip('"').strip()
    
    # Handle whole number with fraction
    parts = fraction_str.split(' ')
    
    if len(parts) == 2:
        # Format: "14 3/8"
        whole = float(parts[0])
        frac_parts = parts[1].split('/')
        if len(frac_parts) == 2:
            decimal = whole + float(frac_parts[0]) / float(frac_parts[1])
        else:
            decimal = whole
    elif '/' in fraction_str:
        # Format: "3/8"
        frac_parts = fraction_str.split('/')
        decimal = float(frac_parts[0]) / float(frac_parts[1])
    else:
        # Format: "14"
        decimal = float(fraction_str)
    
    return decimal


# Testing function
if __name__ == "__main__":
    # Test with sample data
    sample_customer = {
        'name': 'Test Customer',
        'job_name': 'Test Kitchen',
        'job_number': '999',
        'date': '01/08/2025'
    }
    
    sample_doors = [
        {'cabinet': 1, 'qty': 2, 'width': '14 1/2', 'height': '30 1/4', 'type': 'door', 'material': 'Paint Grade'},
        {'cabinet': 2, 'qty': 2, 'width': '15 3/4', 'height': '30 1/4', 'type': 'door', 'material': 'Paint Grade'},
        {'cabinet': 3, 'qty': 1, 'width': '23 7/8', 'height': '30 1/4', 'type': 'door', 'material': 'Stain Grade'},
    ]
    
    # Generate report
    html = generate_optimizer_report(sample_customer, sample_doors, '103')
    
    # Save test output
    with open('test_optimizer.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("Test optimizer report generated: test_optimizer.html")