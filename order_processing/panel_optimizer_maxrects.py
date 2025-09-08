"""
Panel Optimizer Module with Maxrects Algorithm
Advanced 2D bin packing for improved efficiency
Targets 70-80% material utilization
"""

import json
import math
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy

@dataclass
class Panel:
    """Represents a panel to be cut"""
    cabinet: str
    width: float
    height: float
    material: str
    quantity: int = 1
    grain_direction: str = "vertical"  # vertical, horizontal, or none
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
class Rectangle:
    """Represents a free rectangle in the sheet"""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def x2(self) -> float:
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        return self.y + self.height
    
    def area(self) -> float:
        return self.width * self.height
    
    def can_fit(self, width: float, height: float) -> bool:
        """Check if a panel of given dimensions can fit"""
        return self.width >= width and self.height >= height
    
    def overlaps(self, other: 'Rectangle') -> bool:
        """Check if this rectangle overlaps with another"""
        return not (self.x2 <= other.x or self.x >= other.x2 or 
                   self.y2 <= other.y or self.y >= other.y2)

@dataclass
class Sheet:
    """Represents a sheet with placed panels"""
    width: float = 48.0
    height: float = 96.0
    panels: List[PlacedPanel] = field(default_factory=list)
    free_rects: List[Rectangle] = field(default_factory=list)
    material: str = ""
    sheet_number: int = 1
    
    def __post_init__(self):
        if not self.free_rects:
            # Initialize with the entire sheet as free space
            self.free_rects = [Rectangle(0, 0, self.width, self.height)]
    
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

class MaxrectsOptimizer:
    """
    Advanced panel optimizer using Maxrects algorithm
    Maintains list of maximal free rectangles for efficient packing
    """
    
    def __init__(self, kerf: float = 0.125, sheet_width: float = 48.0, sheet_height: float = 96.0):
        self.kerf = kerf  # Blade width
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.min_waste_size = 6.0  # Minimum size to save waste pieces
        self.enable_rotation = False  # Default to no rotation (grain matters)
        
    def optimize(self, panels: List[Panel]) -> List[Sheet]:
        """
        Optimize panel placement using Maxrects algorithm
        Returns list of sheets with placed panels
        """
        # Expand panels by quantity
        expanded_panels = []
        for panel in panels:
            for i in range(panel.quantity):
                p = Panel(
                    cabinet=panel.cabinet,
                    width=panel.width + self.kerf,  # Add kerf to panel dimensions
                    height=panel.height + self.kerf,
                    material=panel.material,
                    quantity=1,
                    grain_direction=panel.grain_direction,
                    label=f"{panel.cabinet}-{i+1}" if panel.quantity > 1 else panel.cabinet
                )
                expanded_panels.append(p)
        
        # Sort panels by area (largest first) for better packing
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
            sheets = self._pack_panels_maxrects(material_panels, material, sheet_number)
            all_sheets.extend(sheets)
            sheet_number += len(sheets)
        
        return all_sheets
    
    def _pack_panels_maxrects(self, panels: List[Panel], material: str, start_sheet_num: int) -> List[Sheet]:
        """Pack panels using Maxrects algorithm"""
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
            
            # Pack as many panels as possible into this sheet
            packed_count = 0
            i = 0
            while i < len(remaining_panels):
                panel = remaining_panels[i]
                placement = self._find_best_placement(sheet, panel)
                
                if placement:
                    # Place the panel
                    placed_panel = PlacedPanel(
                        panel=Panel(
                            cabinet=panel.cabinet,
                            width=panel.width - self.kerf,  # Remove kerf for display
                            height=panel.height - self.kerf,
                            material=panel.material,
                            grain_direction=panel.grain_direction,
                            label=panel.label
                        ),
                        x=placement[0],
                        y=placement[1],
                        rotated=placement[2]
                    )
                    sheet.panels.append(placed_panel)
                    
                    # Update free rectangles
                    self._update_free_rects(sheet, placed_panel)
                    
                    remaining_panels.pop(i)
                    packed_count += 1
                else:
                    i += 1
            
            if packed_count > 0:
                sheets.append(sheet)
                sheet_num += 1
            else:
                # Couldn't place any panel - panels might be too large
                if remaining_panels:
                    print(f"Warning: Panel too large for sheet: {remaining_panels[0]}")
                    remaining_panels.pop(0)
        
        return sheets
    
    def _find_best_placement(self, sheet: Sheet, panel: Panel) -> Optional[Tuple[float, float, bool]]:
        """
        Find the best placement for a panel using Best Area Fit
        Returns (x, y, rotated) or None if can't fit
        """
        best_placement = None
        best_waste = float('inf')
        
        # Try both orientations if rotation is allowed
        orientations = [(panel.width, panel.height, False)]
        if self.enable_rotation and panel.can_rotate():
            orientations.append((panel.height, panel.width, True))
        
        for width, height, rotated in orientations:
            for rect in sheet.free_rects:
                if rect.can_fit(width, height):
                    # Calculate waste (difference between free rect and panel)
                    waste = rect.area() - (width * height)
                    
                    # Prefer placements with less waste
                    if waste < best_waste:
                        best_waste = waste
                        best_placement = (rect.x, rect.y, rotated)
                        
                        # Perfect fit found
                        if waste == 0:
                            return best_placement
        
        return best_placement
    
    def _update_free_rects(self, sheet: Sheet, placed_panel: PlacedPanel):
        """
        Update free rectangles after placing a panel
        Uses the Maxrects split strategy
        """
        new_rects = []
        panel_rect = Rectangle(placed_panel.x, placed_panel.y, 
                               placed_panel.width, placed_panel.height)
        
        # Process each existing free rectangle
        for rect in sheet.free_rects[:]:
            if rect.overlaps(panel_rect):
                # Split the rectangle around the placed panel
                # Left split
                if rect.x < panel_rect.x:
                    new_rects.append(Rectangle(
                        rect.x, rect.y,
                        panel_rect.x - rect.x, rect.height
                    ))
                
                # Right split
                if rect.x2 > panel_rect.x2:
                    new_rects.append(Rectangle(
                        panel_rect.x2, rect.y,
                        rect.x2 - panel_rect.x2, rect.height
                    ))
                
                # Bottom split
                if rect.y < panel_rect.y:
                    new_rects.append(Rectangle(
                        rect.x, rect.y,
                        rect.width, panel_rect.y - rect.y
                    ))
                
                # Top split
                if rect.y2 > panel_rect.y2:
                    new_rects.append(Rectangle(
                        rect.x, panel_rect.y2,
                        rect.width, rect.y2 - panel_rect.y2
                    ))
                
                # Remove the original rectangle
                sheet.free_rects.remove(rect)
        
        # Add new rectangles
        sheet.free_rects.extend(new_rects)
        
        # Remove redundant rectangles (contained within others)
        self._prune_free_rects(sheet)
    
    def _prune_free_rects(self, sheet: Sheet):
        """Remove redundant free rectangles that are contained within others"""
        to_remove = []
        
        for i, rect1 in enumerate(sheet.free_rects):
            for j, rect2 in enumerate(sheet.free_rects):
                if i != j and self._contains(rect2, rect1):
                    to_remove.append(i)
                    break
        
        # Remove in reverse order to maintain indices
        for i in sorted(set(to_remove), reverse=True):
            if i < len(sheet.free_rects):
                sheet.free_rects.pop(i)
    
    def _contains(self, container: Rectangle, contained: Rectangle) -> bool:
        """Check if one rectangle contains another"""
        return (container.x <= contained.x and 
                container.y <= contained.y and
                container.x2 >= contained.x2 and
                container.y2 >= contained.y2)
    
    def generate_cutting_instructions(self, sheet: Sheet) -> List[str]:
        """Generate step-by-step cutting instructions for a sheet"""
        instructions = []
        
        if not sheet.panels:
            return ["No panels on this sheet"]
        
        # Sort panels for optimal cutting sequence
        # Group by Y position first (horizontal cuts), then X position
        panels = sorted(sheet.panels, key=lambda p: (round(p.y, 1), round(p.x, 1)))
        
        # Track cutting operations
        cut_num = 1
        current_y = 0
        
        # Process panels row by row
        y_groups = {}
        for panel in panels:
            y_key = round(panel.y, 1)
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append(panel)
        
        for y_pos in sorted(y_groups.keys()):
            # Make horizontal cut if needed
            if y_pos > current_y + 0.5:  # Allow small tolerance
                instructions.append(f"{cut_num}. Rip at {self._format_dimension(y_pos)} from bottom edge")
                cut_num += 1
                current_y = y_pos
            
            # Process panels in this row
            row_panels = sorted(y_groups[y_pos], key=lambda p: p.x)
            current_x = 0
            
            for panel in row_panels:
                # Make vertical cut if needed
                if panel.x > current_x + 0.5:  # Allow small tolerance
                    instructions.append(f"{cut_num}. Cross-cut at {self._format_dimension(panel.x)} from left edge")
                    cut_num += 1
                    current_x = panel.x
                
                # Describe the resulting piece
                piece_name = f"Cabinet #{panel.panel.cabinet}"
                dimensions = f"{self._format_dimension(panel.width)} x {self._format_dimension(panel.height)}"
                rotation_note = " [ROTATED]" if panel.rotated else ""
                instructions.append(f"    -> {piece_name}: {dimensions}{rotation_note}")
        
        # Add efficiency note
        instructions.append("")
        instructions.append(f"Sheet efficiency: {sheet.efficiency:.1f}%")
        
        # Note about usable waste
        if sheet.free_rects:
            large_waste = [r for r in sheet.free_rects 
                          if r.width >= self.min_waste_size and r.height >= self.min_waste_size]
            if large_waste:
                instructions.append(f"Usable waste pieces: {len(large_waste)}")
                instructions.append("(Save for drawer bottoms or small parts)")
        
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
        if fraction == 8:
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
    
    def generate_svg_diagram(self, sheet: Sheet) -> str:
        """Generate SVG diagram of sheet layout - rotated for portrait printing"""
        # Scale for 8.5x11 paper with margins
        scale = 0.078125  # Fits 96" width to 7.5" page width
        
        # Original dimensions
        orig_width = sheet.width * scale
        orig_height = sheet.height * scale
        
        # Rotated dimensions for portrait display
        display_width = orig_height  # 96" becomes width
        display_height = orig_width   # 48" becomes height
        
        # SVG dimensions in pixels (72 dpi)
        svg_width = display_width * 72
        svg_height = display_height * 72
        
        svg = f'''<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .sheet {{ fill: #f5f5f5; stroke: #333; stroke-width: 2; }}
            .panel {{ fill: #e8f4f8; stroke: #2196F3; stroke-width: 1; }}
            .panel-rotated {{ fill: #f0e8f8; stroke: #9C27B0; stroke-width: 1; }}
            .free-rect {{ fill: #ffebe8; stroke: #ff5722; stroke-width: 0.5; stroke-dasharray: 2,2; opacity: 0.3; }}
            .label {{ font-family: Arial; font-size: 10px; text-anchor: middle; }}
            .dimension {{ font-family: Arial; font-size: 8px; fill: #666; }}
            .efficiency {{ font-family: Arial; font-size: 14px; font-weight: bold; fill: #2196F3; }}
        </style>
        
        <!-- Apply rotation transform -->
        <g transform="rotate(90 {svg_width/2} {svg_height/2}) translate({(svg_width-svg_height)/2} {-(svg_width-svg_height)/2})">
            <!-- Sheet outline -->
            <rect x="10" y="10" width="{orig_width * 72 - 20}" height="{orig_height * 72 - 20}" class="sheet"/>
            '''
        
        # Calculate scale factors for drawing
        margin = 10
        sheet_width_pixels = orig_width * 72 - 2 * margin
        sheet_height_pixels = orig_height * 72 - 2 * margin
        x_scale = sheet_width_pixels / sheet.width
        y_scale = sheet_height_pixels / sheet.height
        
        # Draw free rectangles (waste areas) - optional visualization
        if len(sheet.free_rects) < 20:  # Don't draw if too many (cluttered)
            for rect in sheet.free_rects:
                if rect.width >= self.min_waste_size and rect.height >= self.min_waste_size:
                    rx = margin + rect.x * x_scale
                    ry = margin + (sheet.height - rect.y - rect.height) * y_scale
                    rw = rect.width * x_scale
                    rh = rect.height * y_scale
                    
                    svg += f'''
            <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" class="free-rect"/>'''
        
        # Draw placed panels
        for placed in sheet.panels:
            px = margin + placed.x * x_scale
            py = margin + (sheet.height - placed.y - placed.height) * y_scale
            pw = placed.width * x_scale
            ph = placed.height * y_scale
            
            panel_class = "panel-rotated" if placed.rotated else "panel"
            
            svg += f'''
            <!-- Panel {placed.panel.label} -->
            <rect x="{px}" y="{py}" width="{pw}" height="{ph}" class="{panel_class}"/>
            <g transform="rotate(-90 {px + pw/2} {py + ph/2})">
                <text x="{px + pw/2}" y="{py + ph/2 - 4}" class="label">
                    {placed.panel.label}
                </text>
                <text x="{px + pw/2}" y="{py + ph/2 + 8}" class="dimension">
                    {self._format_dimension(placed.width)} x {self._format_dimension(placed.height)}
                </text>
            </g>'''
        
        # Add efficiency indicator
        efficiency_color = "#4CAF50" if sheet.efficiency > 70 else "#FF9800" if sheet.efficiency > 50 else "#F44336"
        svg += f'''
            <!-- Efficiency indicator -->
            <text x="{orig_width * 72 - 60}" y="{orig_height * 72 - 15}" 
                  class="efficiency" fill="{efficiency_color}">
                {sheet.efficiency:.1f}%
            </text>'''
        
        # Close rotation group
        svg += '''
        </g>
        </svg>'''
        
        return svg


def generate_optimizer_report_maxrects(customer_info: Dict, door_items: List[Dict], 
                                       door_style: str, door_specs: Dict = None,
                                       settings: Dict = None) -> str:
    """Generate optimizer report using Maxrects algorithm"""
    
    from datetime import datetime
    
    # Default settings
    default_settings = {
        'kerf': 0.125,
        'sheet_width': 48.0,
        'sheet_height': 96.0,
        'enable_rotation': False  # Default to no rotation (grain matters)
    }
    
    if settings:
        default_settings.update(settings)
    
    # Extract panels from door items
    panels = extract_panels_from_doors(door_items, door_specs)
    
    # Initialize optimizer with settings
    optimizer = MaxrectsOptimizer(
        kerf=default_settings['kerf'],
        sheet_width=default_settings['sheet_width'],
        sheet_height=default_settings['sheet_height']
    )
    optimizer.enable_rotation = default_settings['enable_rotation']
    
    # Optimize panel placement
    sheets = optimizer.optimize(panels)
    
    # Calculate statistics
    total_sheets = len(sheets)
    total_panels = sum(len(s.panels) for s in sheets)
    avg_efficiency = sum(s.efficiency for s in sheets) / len(sheets) if sheets else 0
    
    # Group sheets by material
    by_material = {}
    for sheet in sheets:
        if sheet.material not in by_material:
            by_material[sheet.material] = []
        by_material[sheet.material].append(sheet)
    
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
        
        .efficiency-high {{ color: #4CAF50; }}
        .efficiency-med {{ color: #FF9800; }}
        .efficiency-low {{ color: #F44336; }}
        
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
        
        .improvement-note {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            margin-top: 20px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
"""
    
    # Determine efficiency class
    efficiency_class = "efficiency-high" if avg_efficiency > 70 else "efficiency-med" if avg_efficiency > 50 else "efficiency-low"
    
    # Summary page
    html += f"""
    <div class="page">
        <div class="header">
            <h1>PANEL OPTIMIZATION REPORT - MAXRECTS ALGORITHM</h1>
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
                    <div class="stat-value {efficiency_class}">{avg_efficiency:.1f}%</div>
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
        total_used = sum(s.used_area for s in material_sheets)
        total_area = sheets_count * default_settings['sheet_width'] * default_settings['sheet_height']
        efficiency = (total_used / total_area * 100) if total_area > 0 else 0
        waste = sum(s.waste_area for s in material_sheets) / 144  # Convert to sq ft
        
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
        
        <div class="improvement-note">
            <strong>Algorithm: Maxrects with Best Area Fit</strong><br>
            This advanced algorithm maintains maximal free rectangles for optimal placement.
            Features enabled: Panel rotation (when grain allows), gap filling, waste tracking.
        </div>
        
        <div style="margin-top: 30px; font-size: 11px; color: #666;">
            <p><strong>Settings:</strong></p>
            <ul>
                <li>Kerf (blade width): """ + str(default_settings['kerf']) + """\"</li>
                <li>Sheet size: """ + str(default_settings['sheet_width']) + """\" x """ + str(default_settings['sheet_height']) + """\"</li>
                <li>Rotation enabled: """ + str(default_settings['enable_rotation']) + """</li>
                <li>Minimum waste size to track: 6" x 6"</li>
            </ul>
        </div>
    </div>"""
    
    # Individual sheet pages
    for sheet in sheets:
        instructions = optimizer.generate_cutting_instructions(sheet)
        svg_diagram = optimizer.generate_svg_diagram(sheet)
        
        efficiency_indicator = "HIGH" if sheet.efficiency > 70 else "MEDIUM" if sheet.efficiency > 50 else "LOW"
        
        html += f"""
    <div class="page">
        <div class="header">
            <h1>SHEET #{sheet.sheet_number} - {sheet.material}</h1>
            <div class="info-grid">
                <div><strong>Sheet Size:</strong> {sheet.width}" x {sheet.height}"</div>
                <div><strong>Panels on Sheet:</strong> {len(sheet.panels)}</div>
                <div><strong>Material Usage:</strong> {sheet.efficiency:.1f}% ({efficiency_indicator})</div>
                <div><strong>Waste:</strong> {sheet.waste_area / 144:.2f} sq ft</div>
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
            if instruction.startswith("    ->"):
                html += f"""
                <ul style="margin: 5px 0 5px 20px;">
                    <li style="list-style: none;">{instruction}</li>
                </ul>"""
            elif instruction == "":
                html += """
            </ol>
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">"""
            elif instruction.startswith("Sheet efficiency") or instruction.startswith("Usable waste"):
                html += f"""
                <p style="font-style: italic; color: #666;">{instruction}</p>"""
            else:
                html += f"""
                <li>{instruction}</li>"""
        
        html += """
            </div>
        </div>
    </div>"""
    
    # Footer
    html += f"""
    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ccc; text-align: center; font-size: 12px; color: #666;">
        <p>anyDoor Panel Optimizer (Maxrects Algorithm) - Generated: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}</p>
    </div>
</body>
</html>"""
    
    return html


def extract_panels_from_doors(door_items: List[Dict], door_specs: Dict = None) -> List[Panel]:
    """Extract panel information from door items"""
    panels = []
    
    # Determine construction type from specs
    is_cope_and_stick = False
    is_mitre_cut = True  # Default
    sticking = 0.5
    stile_width = 3.0  # Default
    
    if door_specs and door_specs.get('raw_text'):
        specs_text = door_specs['raw_text'].lower()
        if 'cope and stick' in specs_text:
            is_cope_and_stick = True
            is_mitre_cut = False
            stile_width = 2.375
        elif 'mitre cut' in specs_text:
            is_mitre_cut = True
            stile_width = 3.0
    
    # Process each door item
    for item in door_items:
        if item['type'] != 'door':
            continue
        
        # Parse dimensions
        width = fraction_to_decimal(item['width'])
        height = fraction_to_decimal(item['height'])
        
        # Calculate panel dimensions
        if is_cope_and_stick:
            panel_width = width - (stile_width + ((sticking * 2) - 0.125))
            panel_height = height - (stile_width + ((sticking * 2) - 0.125))
        else:  # Mitre cut
            panel_width = width - (2 * stile_width) + (sticking - 0.125)
            panel_height = height - (2 * stile_width) + (sticking - 0.125)
        
        # Determine grain direction
        # Default to vertical grain (no rotation) unless specifically MDF
        material = item.get('material', 'Unknown')
        # Only allow rotation for MDF panels, not paint grade (which might still have grain)
        grain = "none" if "mdf" in material.lower() else "vertical"
        
        # Create panel object
        panel = Panel(
            cabinet=str(item['cabinet']),
            width=panel_width,
            height=panel_height,
            material=material,
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
        'date': '01/09/2025'
    }
    
    sample_doors = [
        {'cabinet': 1, 'qty': 2, 'width': '14 1/2', 'height': '30 1/4', 'type': 'door', 'material': 'Paint Grade'},
        {'cabinet': 2, 'qty': 2, 'width': '15 3/4', 'height': '30 1/4', 'type': 'door', 'material': 'Paint Grade'},
        {'cabinet': 3, 'qty': 1, 'width': '23 7/8', 'height': '30 1/4', 'type': 'door', 'material': 'Paint Grade'},
        {'cabinet': 4, 'qty': 2, 'width': '18', 'height': '36', 'type': 'door', 'material': 'Paint Grade'},
        {'cabinet': 5, 'qty': 1, 'width': '30', 'height': '15', 'type': 'door', 'material': 'Paint Grade'},
    ]
    
    # Test with custom settings
    settings = {
        'kerf': 0.125,
        'sheet_width': 48.0,
        'sheet_height': 96.0,
        'enable_rotation': True
    }
    
    # Generate report
    html = generate_optimizer_report_maxrects(sample_customer, sample_doors, '103', settings=settings)
    
    # Save test output
    with open('test_optimizer_maxrects.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("Test Maxrects optimizer report generated: test_optimizer_maxrects.html")
    
    # Compare efficiency
    panels = extract_panels_from_doors(sample_doors)
    
    # Test old algorithm
    from panel_optimizer import PanelOptimizer
    old_optimizer = PanelOptimizer()
    old_sheets = old_optimizer.optimize(panels)
    old_efficiency = sum(s.efficiency for s in old_sheets) / len(old_sheets) if old_sheets else 0
    
    # Test new algorithm
    new_optimizer = MaxrectsOptimizer()
    new_sheets = new_optimizer.optimize(panels)
    new_efficiency = sum(s.efficiency for s in new_sheets) / len(new_sheets) if new_sheets else 0
    
    print(f"\nEfficiency Comparison:")
    print(f"Old Algorithm (Shelf): {old_efficiency:.1f}%")
    print(f"New Algorithm (Maxrects): {new_efficiency:.1f}%")
    print(f"Improvement: {new_efficiency - old_efficiency:.1f}%")