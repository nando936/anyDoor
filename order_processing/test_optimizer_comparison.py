"""
Test script to compare old shelf algorithm vs new Maxrects algorithm
Tests with various panel configurations to find efficiency differences
"""

from panel_optimizer import PanelOptimizer, Panel
from panel_optimizer_maxrects import MaxrectsOptimizer, extract_panels_from_doors, generate_optimizer_report_maxrects

def test_algorithm_comparison():
    """Compare the two algorithms with different test cases"""
    
    test_cases = [
        {
            'name': 'Small Kitchen',
            'doors': [
                {'cabinet': 1, 'qty': 2, 'width': '14 1/2', 'height': '30 1/4', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 2, 'qty': 2, 'width': '15 3/4', 'height': '30 1/4', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 3, 'qty': 1, 'width': '23 7/8', 'height': '30 1/4', 'type': 'door', 'material': 'MDF'},
            ]
        },
        {
            'name': 'Large Kitchen with Mixed Sizes',
            'doors': [
                # Base cabinets
                {'cabinet': 1, 'qty': 2, 'width': '14 1/2', 'height': '30 1/4', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 2, 'qty': 2, 'width': '15 3/4', 'height': '30 1/4', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 3, 'qty': 1, 'width': '23 7/8', 'height': '30 1/4', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 4, 'qty': 2, 'width': '18', 'height': '30 1/4', 'type': 'door', 'material': 'MDF'},
                # Upper cabinets
                {'cabinet': 5, 'qty': 2, 'width': '14 1/2', 'height': '36', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 6, 'qty': 2, 'width': '15 3/4', 'height': '36', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 7, 'qty': 2, 'width': '18', 'height': '42', 'type': 'door', 'material': 'MDF'},
                # Pantry
                {'cabinet': 8, 'qty': 2, 'width': '23 7/8', 'height': '83 1/4', 'type': 'door', 'material': 'MDF'},
            ]
        },
        {
            'name': 'Many Small Panels',
            'doors': [
                {'cabinet': i, 'qty': 1, 'width': '12', 'height': '12', 'type': 'door', 'material': 'MDF'}
                for i in range(1, 21)  # 20 small panels
            ]
        },
        {
            'name': 'Mixed Sizes with Rotation Allowed',
            'doors': [
                {'cabinet': 1, 'qty': 2, 'width': '30', 'height': '15', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 2, 'qty': 2, 'width': '24', 'height': '18', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 3, 'qty': 3, 'width': '20', 'height': '20', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 4, 'qty': 2, 'width': '36', 'height': '12', 'type': 'door', 'material': 'MDF'},
                {'cabinet': 5, 'qty': 1, 'width': '40', 'height': '30', 'type': 'door', 'material': 'MDF'},
            ]
        }
    ]
    
    print("=" * 80)
    print("PANEL OPTIMIZER ALGORITHM COMPARISON")
    print("=" * 80)
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print("-" * 40)
        
        # Extract panels
        panels = extract_panels_from_doors(test_case['doors'])
        total_panel_area = sum(p.area() * p.quantity for p in panels)
        
        # Test old algorithm
        old_optimizer = PanelOptimizer(kerf=0.125)
        old_sheets = old_optimizer.optimize(panels.copy())
        old_sheet_count = len(old_sheets)
        old_avg_efficiency = sum(s.efficiency for s in old_sheets) / len(old_sheets) if old_sheets else 0
        old_total_waste = sum(s.waste_area for s in old_sheets)
        
        # Test new algorithm (no rotation)
        new_optimizer = MaxrectsOptimizer(kerf=0.125)
        new_optimizer.enable_rotation = False
        new_sheets = new_optimizer.optimize(panels.copy())
        new_sheet_count = len(new_sheets)
        new_avg_efficiency = sum(s.efficiency for s in new_sheets) / len(new_sheets) if new_sheets else 0
        new_total_waste = sum(s.waste_area for s in new_sheets)
        
        # Test new algorithm with rotation for MDF
        new_optimizer_rot = MaxrectsOptimizer(kerf=0.125)
        new_optimizer_rot.enable_rotation = True
        new_sheets_rot = new_optimizer_rot.optimize(panels.copy())
        new_sheet_count_rot = len(new_sheets_rot)
        new_avg_efficiency_rot = sum(s.efficiency for s in new_sheets_rot) / len(new_sheets_rot) if new_sheets_rot else 0
        new_total_waste_rot = sum(s.waste_area for s in new_sheets_rot)
        
        print(f"Total Panel Area: {total_panel_area / 144:.1f} sq ft")
        print(f"Total Panels: {sum(p.quantity for p in panels)}")
        print()
        
        print("Old Algorithm (Shelf Packing):")
        print(f"  Sheets Used: {old_sheet_count}")
        print(f"  Average Efficiency: {old_avg_efficiency:.1f}%")
        print(f"  Total Waste: {old_total_waste / 144:.1f} sq ft")
        print()
        
        print("New Algorithm (Maxrects - No Rotation):")
        print(f"  Sheets Used: {new_sheet_count}")
        print(f"  Average Efficiency: {new_avg_efficiency:.1f}%")
        print(f"  Total Waste: {new_total_waste / 144:.1f} sq ft")
        print(f"  Improvement: {old_avg_efficiency - new_avg_efficiency:+.1f}% efficiency")
        print(f"  Sheets Saved: {old_sheet_count - new_sheet_count}")
        print()
        
        print("New Algorithm (Maxrects - With Rotation for MDF):")
        print(f"  Sheets Used: {new_sheet_count_rot}")
        print(f"  Average Efficiency: {new_avg_efficiency_rot:.1f}%")
        print(f"  Total Waste: {new_total_waste_rot / 144:.1f} sq ft")
        print(f"  Improvement: {old_avg_efficiency - new_avg_efficiency_rot:+.1f}% efficiency")
        print(f"  Sheets Saved: {old_sheet_count - new_sheet_count_rot}")
    
    # Generate sample HTML reports for visual comparison
    print("\n" + "=" * 80)
    print("Generating sample HTML reports for visual inspection...")
    
    customer_info = {
        'name': 'Test Customer',
        'job_name': 'Algorithm Comparison',
        'job_number': '999',
        'date': '01/09/2025'
    }
    
    # Use the "Large Kitchen" test case for reports
    large_kitchen = test_cases[1]['doors']
    
    # Generate old algorithm report
    from panel_optimizer import generate_optimizer_report
    old_html = generate_optimizer_report(customer_info, large_kitchen, '103')
    with open('comparison_old_algorithm.html', 'w', encoding='utf-8') as f:
        f.write(old_html)
    
    # Generate new algorithm report (no rotation)
    settings_no_rot = {'enable_rotation': False}
    new_html = generate_optimizer_report_maxrects(customer_info, large_kitchen, '103', settings=settings_no_rot)
    with open('comparison_new_algorithm_no_rotation.html', 'w', encoding='utf-8') as f:
        f.write(new_html)
    
    # Generate new algorithm report (with rotation)
    settings_rot = {'enable_rotation': True}
    new_html_rot = generate_optimizer_report_maxrects(customer_info, large_kitchen, '103', settings=settings_rot)
    with open('comparison_new_algorithm_with_rotation.html', 'w', encoding='utf-8') as f:
        f.write(new_html_rot)
    
    print("Reports generated:")
    print("  - comparison_old_algorithm.html")
    print("  - comparison_new_algorithm_no_rotation.html")
    print("  - comparison_new_algorithm_with_rotation.html")
    print("\nOpen these files in a browser to visually compare the packing strategies.")

if __name__ == "__main__":
    test_algorithm_comparison()