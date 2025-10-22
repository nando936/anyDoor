"""
Master Script - Generate All Reports from Unified Door Order Format
Generates estimate PDF, stile & rail report, and panel report from a single unified JSON file
Works with both True Custom and Inhouse Door Order unified output
"""
import os
import sys
import subprocess

def main():
    """Generate all reports from unified door order JSON"""

    if len(sys.argv) < 2:
        print("[ERROR] Please specify unified door order JSON path")
        print("Usage: python generate_all_reports.py <unified_door_order.json> [estimate_number]")
        print("\nExample:")
        print("  python generate_all_reports.py page_1_unified_door_order.json")
        print("  python generate_all_reports.py page_1_unified_door_order.json EST-2025-001")
        sys.exit(1)

    unified_json_path = sys.argv[1]
    estimate_number = sys.argv[2] if len(sys.argv) > 2 else None

    # Verify file exists
    if not os.path.exists(unified_json_path):
        print(f"[ERROR] Unified door order JSON not found: {unified_json_path}")
        sys.exit(1)

    # Get script directory (where report generators live)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths to report generators
    estimate_pdf_script = os.path.join(script_dir, 'generate_estimate_pdf.py')
    stile_rail_script = os.path.join(script_dir, 'generate_stile_rail_report.py')
    panel_script = os.path.join(script_dir, 'generate_panel_report.py')

    print("=" * 80)
    print("GENERATING ALL REPORTS FROM UNIFIED DOOR ORDER")
    print("=" * 80)
    print(f"Input: {os.path.basename(unified_json_path)}")
    print()

    # Track success/failure
    results = []

    # 1. Generate Estimate PDF
    print("[1/3] Generating Estimate PDF...")
    try:
        cmd = [sys.executable, estimate_pdf_script, unified_json_path]
        if estimate_number:
            cmd.append(estimate_number)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        results.append(("Estimate PDF", True))
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to generate estimate PDF")
        print(e.stderr)
        results.append(("Estimate PDF", False))

    # 2. Generate Stile & Rail Report
    print("\n[2/3] Generating Stile & Rail Report...")
    try:
        result = subprocess.run(
            [sys.executable, stile_rail_script, unified_json_path],
            check=True, capture_output=True, text=True
        )
        print(result.stdout)
        results.append(("Stile & Rail Report", True))
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to generate stile & rail report")
        print(e.stderr)
        results.append(("Stile & Rail Report", False))

    # 3. Generate Panel Report
    print("\n[3/3] Generating Panel Report...")
    try:
        result = subprocess.run(
            [sys.executable, panel_script, unified_json_path],
            check=True, capture_output=True, text=True
        )
        print(result.stdout)
        results.append(("Panel Report", True))
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to generate panel report")
        print(e.stderr)
        results.append(("Panel Report", False))

    # Summary
    print("\n" + "=" * 80)
    print("REPORT GENERATION SUMMARY")
    print("=" * 80)

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    for report_name, success in results:
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {report_name}")

    print("=" * 80)
    print(f"COMPLETED: {success_count}/{total_count} reports generated successfully")
    print("=" * 80)

    # Exit with error code if any failed
    if success_count < total_count:
        sys.exit(1)

if __name__ == '__main__':
    main()
