#!/usr/bin/env python3
"""
Shared utility functions for door order processing
Used by both Inhouse Door Order and other processing modules
"""

import json
from datetime import datetime
from typing import Tuple, List, Dict, Any


def fraction_to_decimal(fraction_str: str) -> float:
    """
    Convert fraction string to decimal

    Examples:
        "14 3/8" -> 14.375
        "3/8" -> 0.375
        "14" -> 14.0
    """
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


def decimal_to_fraction(decimal: float) -> str:
    """
    Convert decimal to fraction string (inches with 16ths precision)

    Examples:
        14.375 -> "14 3/8"
        0.5625 -> "9/16"
        14.0 -> "14"
    """
    whole = int(decimal)
    remainder = decimal - whole

    if remainder < 0.001:  # Essentially zero
        return str(whole)

    # Convert to 16ths
    sixteenths = round(remainder * 16)

    # Simplify fraction
    if sixteenths == 0:
        return str(whole)
    elif sixteenths == 16:
        return str(whole + 1)
    elif sixteenths % 8 == 0:
        numerator = sixteenths // 8
        denominator = 2
    elif sixteenths % 4 == 0:
        numerator = sixteenths // 4
        denominator = 4
    elif sixteenths % 2 == 0:
        numerator = sixteenths // 2
        denominator = 8
    else:
        numerator = sixteenths
        denominator = 16

    if whole == 0:
        return f"{numerator}/{denominator}"
    else:
        return f"{whole} {numerator}/{denominator}"


def parse_overlay_spec(overlay_info: str) -> float:
    """
    Parse overlay specification to decimal value

    Examples:
        "5/8 OL" -> 0.625
        "5/8" -> 0.625
        "3/4 OL" -> 0.75
    """
    if not overlay_info:
        return 0.625  # Default 5/8"

    # Remove "OL" and extra spaces
    overlay_str = overlay_info.upper().replace('OL', '').strip()

    return fraction_to_decimal(overlay_str)


def convert_opening_to_finished(opening_size: str, overlay_decimal: float) -> Tuple[str, float]:
    """
    Convert opening size to finished size by subtracting overlay

    Args:
        opening_size: Opening size as fraction string (e.g., "14 3/8")
        overlay_decimal: Overlay amount in decimal (e.g., 0.625 for 5/8")

    Returns:
        Tuple of (finished_size_string, finished_size_decimal)

    Examples:
        ("14 3/8", 0.625) -> ("13 3/4", 13.75)
    """
    opening_decimal = fraction_to_decimal(opening_size)
    finished_decimal = opening_decimal - overlay_decimal
    finished_string = decimal_to_fraction(finished_decimal)

    return finished_string, finished_decimal


def calculate_sqft(width_decimal: float, height_decimal: float) -> float:
    """
    Calculate square footage from width and height in inches

    Args:
        width_decimal: Width in inches
        height_decimal: Height in inches

    Returns:
        Square footage rounded to 2 decimal places
    """
    sqft = (width_decimal * height_decimal) / 144
    return round(sqft, 2)


def calculate_summary(doors: List[Dict], drawers: List[Dict]) -> Dict[str, Any]:
    """
    Calculate summary statistics for doors and drawers

    Args:
        doors: List of door items with 'qty' and 'sqft' fields
        drawers: List of drawer items with 'qty' field

    Returns:
        Dictionary with summary statistics
    """
    total_doors = sum(item.get('qty', 0) for item in doors)
    total_drawers = sum(item.get('qty', 0) for item in drawers)
    total_sqft = sum(item.get('sqft', 0) * item.get('qty', 0) for item in doors)

    summary = {
        "total_doors": total_doors,
        "total_drawers": total_drawers,
        "total_sqft": round(total_sqft, 2),
        "total_items": total_doors + total_drawers
    }

    return summary


def save_unified_json(data: Dict, file_path: str) -> None:
    """
    Save unified JSON data to file

    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Unified door order: {file_path}")


def get_current_date() -> str:
    """
    Get current date in YYYY-MM-DD format

    Returns:
        Date string in YYYY-MM-DD format
    """
    return datetime.now().strftime('%Y-%m-%d')


def get_current_datetime() -> str:
    """
    Get current date and time in ISO format

    Returns:
        DateTime string in ISO format
    """
    return datetime.now().isoformat()
