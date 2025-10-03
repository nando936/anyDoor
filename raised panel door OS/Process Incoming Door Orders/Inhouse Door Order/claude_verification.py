#!/usr/bin/env python3
"""
Claude-based OCR verification for suspicious measurements.
Uses Claude CLI to verify measurements that look incorrect.
"""

import subprocess
import os
import re


def is_suspicious_measurement(text):
    """
    Detect if a measurement looks suspicious and might need Claude verification.

    Returns: (is_suspicious, reason)
    """
    if not text:
        return False, None

    # Pattern 1: Decimal point in a fraction (e.g., "6.5/8" should be "6 5/8")
    if re.search(r'\d+\.\d+/\d+', text):
        return True, "decimal_in_fraction"

    # Pattern 2: Missing space before fraction (e.g., "65/8" should be "6 5/8")
    # But allow valid fractions like "5/8" or "1/2" without a whole number
    if re.search(r'\d{2,}/\d+', text):
        # Check if numerator is larger than denominator (impossible)
        match = re.search(r'(\d+)/(\d+)', text)
        if match:
            num, denom = int(match.group(1)), int(match.group(2))
            if num >= denom:
                return True, "invalid_fraction"
        return True, "missing_space_before_fraction"

    # Pattern 3: Impossible fraction (numerator >= denominator, except 1/1 which shouldn't appear)
    # Check all fractions in the text
    for match in re.finditer(r'(\d+)/(\d+)', text):
        num, denom = int(match.group(1)), int(match.group(2))
        if num >= denom:
            return True, "impossible_fraction"

    # Pattern 4: Strange multi-digit numbers before slash (e.g., "13.5/8" or "345/8")
    if re.search(r'\d{3,}/\d+', text):
        return True, "multi_digit_numerator"

    # Pattern 5: Leading dash or minus (e.g., "-345/8")
    if text.startswith('-') or text.startswith('−'):
        return True, "leading_dash"

    return False, None


def verify_measurements_with_claude(suspicious_measurements, image_dir):
    """
    Verify suspicious measurements using Claude via CLI.

    Args:
        suspicious_measurements: List of dicts with keys:
            - 'text': The suspicious OCR text
            - 'debug_image': Path to the debug zoom image
            - 'index': Index in the measurements list
            - 'reason': Why it's suspicious
        image_dir: Directory containing the debug images (for --add-dir permission)

    Returns:
        dict mapping index -> corrected_text
    """
    if not suspicious_measurements:
        return {}

    print(f"\n=== CLAUDE VERIFICATION ===")
    print(f"Found {len(suspicious_measurements)} suspicious measurements to verify")

    # Build prompt for Claude
    prompt = """I need you to verify OCR readings from cabinet measurement images. The OCR has misread some measurements. I'll provide image file paths - please read each one and tell me the correct measurement.

Each image shows a measurement in green text. Measurements are in the format:
- Whole number + fraction: "33 5/8" or "15 1/2"
- Just fraction: "5/8" or "1/2"
- Just whole number: "44"

Respond with a numbered list in this EXACT format (just the measurement, nothing else):
1. [measurement]
2. [measurement]
3. [measurement]

Here are the images to verify:

"""

    for i, meas in enumerate(suspicious_measurements, 1):
        prompt += f"{i}. {meas['debug_image']}\n"
        prompt += f"   OCR read: {meas['text']} (suspicious: {meas['reason']})\n\n"

    # Call Claude via CLI
    claude_exe = r'C:\Users\nando\AppData\Roaming\npm\claude.cmd'

    try:
        print(f"Calling Claude to verify {len(suspicious_measurements)} measurements...")
        result = subprocess.run(
            f'{claude_exe} --print --add-dir "{image_dir}"',
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes for multiple images
            shell=True
        )

        if result.returncode != 0:
            print(f"[WARNING] Claude call failed with code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return {}

        # Parse Claude's response
        response = result.stdout.strip()
        print(f"\nClaude's response:")
        print(response)

        # Extract corrected measurements
        corrections = {}
        lines = response.split('\n')

        for i, line in enumerate(lines):
            # Match pattern like "1. 33 5/8" or "2. 15 1/2"
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                list_num = int(match.group(1)) - 1  # Convert to 0-based index
                corrected_text = match.group(2).strip()

                if list_num < len(suspicious_measurements):
                    original_index = suspicious_measurements[list_num]['index']
                    corrections[original_index] = corrected_text
                    print(f"  [{list_num+1}] Index {original_index}: '{suspicious_measurements[list_num]['text']}' → '{corrected_text}'")

        print(f"\nCorrected {len(corrections)} measurements")
        return corrections

    except subprocess.TimeoutExpired:
        print("[WARNING] Claude call timed out")
        return {}
    except Exception as e:
        print(f"[WARNING] Claude verification failed: {e}")
        return {}


def apply_claude_corrections(measurements_list, corrections):
    """
    Apply Claude's corrections to the measurements list.

    Args:
        measurements_list: List of measurement dicts
        corrections: Dict mapping index -> corrected_text

    Returns:
        Number of corrections applied
    """
    if not corrections:
        return 0

    count = 0
    for index, corrected_text in corrections.items():
        if index < len(measurements_list):
            old_text = measurements_list[index]['text']
            measurements_list[index]['text'] = corrected_text
            measurements_list[index]['claude_corrected'] = True
            measurements_list[index]['original_ocr'] = old_text
            count += 1
            print(f"  Applied correction at index {index}: '{old_text}' → '{corrected_text}'")

    return count
