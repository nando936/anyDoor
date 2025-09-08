"""
Debug script to understand why Maxrects isn't performing better
"""

from panel_optimizer_maxrects import MaxrectsOptimizer, Panel, Sheet

# Create simple test case
panels = [
    Panel(cabinet="1", width=20, height=30, material="MDF", grain_direction="none"),
    Panel(cabinet="2", width=15, height=25, material="MDF", grain_direction="none"),
    Panel(cabinet="3", width=10, height=20, material="MDF", grain_direction="none"),
    Panel(cabinet="4", width=12, height=18, material="MDF", grain_direction="none"),
]

print("Test Panels:")
for p in panels:
    print(f"  Cabinet {p.cabinet}: {p.width}\" x {p.height}\" (area: {p.area()} sq in)")

print(f"\nTotal area needed: {sum(p.area() for p in panels)} sq in")
print(f"Sheet area: {48 * 96} = 4608 sq in")
print(f"Theoretical efficiency: {sum(p.area() for p in panels) / 4608 * 100:.1f}%")

# Test with rotation enabled
optimizer = MaxrectsOptimizer(kerf=0.125)
optimizer.enable_rotation = True

sheets = optimizer.optimize(panels)

print(f"\nMaxrects Results (with rotation):")
print(f"Sheets used: {len(sheets)}")

for sheet in sheets:
    print(f"\nSheet {sheet.sheet_number}:")
    print(f"  Panels placed: {len(sheet.panels)}")
    print(f"  Efficiency: {sheet.efficiency:.1f}%")
    print(f"  Free rectangles: {len(sheet.free_rects)}")
    
    for panel in sheet.panels:
        rot_str = " (ROTATED)" if panel.rotated else ""
        print(f"    Panel {panel.panel.label} at ({panel.x:.1f}, {panel.y:.1f}) - {panel.width}x{panel.height}{rot_str}")
    
    # Show a few free rectangles
    if sheet.free_rects:
        print(f"  Sample free rects (first 3):")
        for rect in sheet.free_rects[:3]:
            print(f"    ({rect.x:.1f}, {rect.y:.1f}) - {rect.width:.1f}x{rect.height:.1f} (area: {rect.area():.0f})")

# Test shelf algorithm for comparison
from panel_optimizer import PanelOptimizer

old_optimizer = PanelOptimizer(kerf=0.125)
old_sheets = old_optimizer.optimize(panels)

print(f"\nShelf Algorithm Results:")
print(f"Sheets used: {len(old_sheets)}")
for sheet in old_sheets:
    print(f"  Sheet {sheet.sheet_number}: {len(sheet.panels)} panels, {sheet.efficiency:.1f}% efficiency")