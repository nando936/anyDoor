"""
Shared Utilities for Door Order Processing
Common functions used across all order types and report generators
"""
from fractions import Fraction
import json
import os
from datetime import datetime


def fraction_to_decimal(measurement):
    """
    Convert measurement string like '23 15/16' to decimal

    Args:
        measurement: String like "23 15/16" or "32" or "1/2"

    Returns:
        float: Decimal value (e.g., 23.9375)
    """
    if not measurement or measurement == '':
        return 0.0

    # Remove asterisks, dollar signs, and other notation symbols
    measurement = measurement.replace('*', '').replace('$', '').strip()

    parts = measurement.strip().split()

    if len(parts) == 0:
        return 0.0

    if len(parts) == 1:
        # Just a fraction or whole number
        if '/' in parts[0]:
            frac = Fraction(parts[0])
            return float(frac)
        else:
            return float(parts[0])

    # Whole number + fraction
    whole = int(parts[0])
    frac = Fraction(parts[1])
    return whole + float(frac)


def decimal_to_fraction(decimal_value, max_denominator=16):
    """
    Convert decimal to fraction string

    Args:
        decimal_value: Decimal number (e.g., 23.9375)
        max_denominator: Maximum denominator for fraction (default 16)

    Returns:
        str: Fractional measurement (e.g., "23 15/16")
    """
    frac = Fraction(decimal_value).limit_denominator(max_denominator)

    if frac.numerator >= frac.denominator:
        whole = frac.numerator // frac.denominator
        remainder = frac.numerator % frac.denominator

        if remainder == 0:
            return str(whole)
        else:
            return f"{whole} {remainder}/{frac.denominator}"
    else:
        return str(frac)


def calculate_sqft(width, height):
    """
    Calculate square footage from width and height

    Args:
        width: Width in inches (decimal or string)
        height: Height in inches (decimal or string)

    Returns:
        float: Square footage
    """
    if isinstance(width, str):
        width = fraction_to_decimal(width)
    if isinstance(height, str):
        height = fraction_to_decimal(height)

    # Convert square inches to square feet
    return (width * height) / 144.0


def convert_opening_to_finished(opening_size, overlay_value):
    """
    Convert opening size to finished size by adding overlay

    Args:
        opening_size: Opening measurement (string or decimal)
        overlay_value: Overlay to add per side (decimal inches, e.g., 0.625 for 5/8")

    Returns:
        tuple: (finished_size_string, finished_size_decimal)
    """
    if isinstance(opening_size, str):
        opening_decimal = fraction_to_decimal(opening_size)
    else:
        opening_decimal = opening_size

    # Add overlay to both sides
    finished_decimal = opening_decimal + (overlay_value * 2)
    finished_string = decimal_to_fraction(finished_decimal)

    return finished_string, finished_decimal


def parse_overlay_spec(overlay_spec):
    """
    Parse overlay specification to decimal value

    Args:
        overlay_spec: String like "5/8 OL" or "1/2 OL"

    Returns:
        float: Overlay value in decimal inches (e.g., 0.625)
    """
    if not overlay_spec or overlay_spec == "":
        return 0.0

    # Extract fraction part
    parts = overlay_spec.split()
    if len(parts) > 0:
        fraction_part = parts[0]
        return fraction_to_decimal(fraction_part)

    return 0.0


def validate_unified_format(data):
    """
    Validate that data conforms to unified door order schema

    Args:
        data: Dictionary to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    required_top_level = ["schema_version", "source", "order_info", "specifications", "doors", "drawers", "summary"]

    for field in required_top_level:
        if field not in data:
            return False, f"Missing required field: {field}"

    # Validate source
    if "type" not in data["source"]:
        return False, "Missing source.type"
    if data["source"]["type"] not in ["true_custom", "inhouse", "other"]:
        return False, f"Invalid source.type: {data['source']['type']}"

    # Validate order_info
    if "jobsite" not in data["order_info"]:
        return False, "Missing order_info.jobsite"

    # Validate specifications
    if "wood_type" not in data["specifications"]:
        return False, "Missing specifications.wood_type"
    if "door_style" not in data["specifications"]:
        return False, "Missing specifications.door_style"

    # Validate doors array
    if not isinstance(data["doors"], list):
        return False, "doors must be an array"

    # Validate drawers array
    if not isinstance(data["drawers"], list):
        return False, "drawers must be an array"

    return True, "Valid"


def save_unified_json(data, output_path):
    """
    Save unified door order JSON with validation

    Args:
        data: Unified door order data
        output_path: Path to save JSON file

    Returns:
        bool: True if successful
    """
    is_valid, error_msg = validate_unified_format(data)

    if not is_valid:
        print(f"[ERROR] Invalid unified format: {error_msg}")
        return False

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Saved unified door order: {output_path}")
    return True


def load_unified_json(json_path):
    """
    Load and validate unified door order JSON

    Args:
        json_path: Path to JSON file

    Returns:
        dict: Unified door order data, or None if invalid
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] File not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    is_valid, error_msg = validate_unified_format(data)

    if not is_valid:
        print(f"[ERROR] Invalid unified format: {error_msg}")
        return None

    return data


def calculate_summary(doors, drawers):
    """
    Calculate summary statistics for doors and drawers

    Args:
        doors: List of door items
        drawers: List of drawer items

    Returns:
        dict: Summary statistics
    """
    total_door_units = sum(door['qty'] for door in doors)
    total_drawer_units = sum(drawer['qty'] for drawer in drawers)

    return {
        "total_doors": len(doors),
        "total_door_units": total_door_units,
        "total_drawers": len(drawers),
        "total_drawer_units": total_drawer_units,
        "total_units": total_door_units + total_drawer_units
    }


def get_current_date():
    """Get current date in YYYY-MM-DD format"""
    return datetime.now().strftime("%Y-%m-%d")


if __name__ == "__main__":
    # Test functions
    print("Testing fraction_to_decimal:")
    print(f"  '23 15/16' -> {fraction_to_decimal('23 15/16')}")
    print(f"  '32' -> {fraction_to_decimal('32')}")
    print(f"  '1/2' -> {fraction_to_decimal('1/2')}")

    print("\nTesting decimal_to_fraction:")
    print(f"  23.9375 -> '{decimal_to_fraction(23.9375)}'")
    print(f"  32.0 -> '{decimal_to_fraction(32.0)}'")
    print(f"  0.5 -> '{decimal_to_fraction(0.5)}'")

    print("\nTesting calculate_sqft:")
    print(f"  24\" x 32\" -> {calculate_sqft(24, 32):.2f} sqft")

    print("\nTesting convert_opening_to_finished:")
    finished_str, finished_dec = convert_opening_to_finished("23", 0.625)
    print(f"  23\" opening + 5/8\" overlay -> {finished_str} ({finished_dec})")

    print("\nTesting parse_overlay_spec:")
    print(f"  '5/8 OL' -> {parse_overlay_spec('5/8 OL')}")
    print(f"  '1/2 OL' -> {parse_overlay_spec('1/2 OL')}")
