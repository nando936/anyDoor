#!/usr/bin/env python3
"""
Claude-based OCR verification for suspicious measurements.
Uses Claude CLI to verify measurements that look incorrect.
"""

import subprocess
import os
import re


def is_suspicious_measurement(text, raw_ocr=None):
    """
    Detect if a measurement looks suspicious and might need Claude verification.

    Args:
        text: Cleaned measurement text
        raw_ocr: Raw OCR text before cleaning (optional)

    Returns: (is_suspicious, reason)
    """
    if not text:
        return False, None

    # Check raw OCR for suspicious patterns (before cleaning removed them)
    if raw_ocr:
        # Pattern 0: Leading dash or minus in RAW OCR (e.g., "-9-" which gets cleaned to "9")
        if raw_ocr.startswith('-') or raw_ocr.startswith('−'):
            # Only mark suspicious if cleaned text doesn't look like a valid measurement
            # If cleaning fixed it (e.g., "-45—" → "45"), trust the cleaned result
            if not re.match(r'^\d+(\s+\d+/\d+)?$', text.strip()):
                return True, "leading_dash_in_raw_ocr"
            # Otherwise, cleaning fixed it - not suspicious

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

    # Pattern 6: Measurement over 120 inches (likely OCR error - missing space/fraction)
    # Extract numeric value from text
    try:
        # Try to parse as a simple number first
        match = re.match(r'^(\d+)(?:\s+(\d+)/(\d+))?', text)
        if match:
            whole = int(match.group(1))
            if match.group(2) and match.group(3):
                # Has fraction - calculate total
                frac_num = int(match.group(2))
                frac_den = int(match.group(3))
                value = whole + (frac_num / frac_den)
            else:
                # Just whole number
                value = whole

            if value > 120:
                return True, "over_120_inches"
    except:
        pass

    # Pattern 7: Letters in measurement (except valid notations F, O, NH)
    # Valid notations: F (finished size), O (opening size), NH (no hinges)
    if re.search(r'[A-Za-z]', text):
        # Remove valid notations from check
        text_cleaned = text.upper()
        text_cleaned = re.sub(r'\bF\b', '', text_cleaned)  # Remove standalone F
        text_cleaned = re.sub(r'\bO\b', '', text_cleaned)  # Remove standalone O
        text_cleaned = re.sub(r'\bNH\b', '', text_cleaned)  # Remove NH
        # Check if any letters remain after removing valid notations
        if re.search(r'[A-Z]', text_cleaned):
            return True, "invalid_letters_in_measurement"

    # Pattern 8: Invalid fraction denominator (not in 1/16 increments)
    # Valid fractions: 1/2, 1/4, 3/4, 1/8, 3/8, 5/8, 7/8, and 1/16 through 15/16
    for match in re.finditer(r'(\d+)/(\d+)', text):
        num, denom = int(match.group(1)), int(match.group(2))

        # Check if denominator is valid (2, 4, 8, 16)
        valid_denoms = {2, 4, 8, 16}
        if denom not in valid_denoms:
            return True, "invalid_fraction_denominator"

        # Check valid numerator for each denominator
        # 1/2: only 1 is valid
        if denom == 2 and num != 1:
            return True, "invalid_fraction_numerator"
        # 1/4, 3/4: only 1 and 3 are valid
        if denom == 4 and num not in {1, 3}:
            return True, "invalid_fraction_numerator"
        # 1/8, 3/8, 5/8, 7/8: only odd numbers 1,3,5,7 are valid
        if denom == 8 and num not in {1, 3, 5, 7}:
            return True, "invalid_fraction_numerator"
        # 1/16 through 15/16: only odd numbers 1,3,5,7,9,11,13,15 are valid
        if denom == 16 and num not in {1, 3, 5, 7, 9, 11, 13, 15}:
            return True, "invalid_fraction_numerator"

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
    prompt = """I need you to read measurements from cabinet measurement images. Please look at each image and tell me what measurement you see.

CRITICAL INSTRUCTIONS:
- If the image is BLANK or completely empty: respond with "BLANK"
- If the text is too faint, blurry, or unreadable: respond with "UNREADABLE"
- Only provide a measurement if you can clearly read it

Each image should show a measurement in green text. Valid formats:
- Whole number + fraction: "33 5/8" or "15 1/2"
- Just fraction: "5/8" or "1/2"
- Just whole number: "44"

Respond with a numbered list in this EXACT format:
1. [measurement or BLANK or UNREADABLE]
2. [measurement or BLANK or UNREADABLE]
3. [measurement or BLANK or UNREADABLE]

Here are the images to read:

"""

    for i, meas in enumerate(suspicious_measurements, 1):
        prompt += f"{i}. {meas['debug_image']}\n\n"

    # Call Claude via CLI
    import platform
    if platform.system() == 'Windows':
        claude_exe = r'C:\Users\nando\AppData\Roaming\npm\claude.cmd'
    else:
        claude_exe = 'claude'  # Linux/Mac - assume in PATH

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
        Tuple: (number of corrections applied, list of invalid indices)
    """
    if not corrections:
        return 0, []

    # Phrases that indicate Claude couldn't read the measurement
    invalid_phrases = [
        'no visible',
        'blank',
        'washed out',
        'unclear',
        'cannot read',
        'illegible',
        'too blurry',
        'not readable',
        'no text',
        'unreadable',
        'too faint',
        'not visible'
    ]

    count = 0
    invalid_indices = []

    for index, corrected_text in corrections.items():
        if index < len(measurements_list):
            old_text = measurements_list[index]['text']

            # Check if Claude marked this measurement as invalid/unreadable
            corrected_lower = corrected_text.lower()
            is_invalid = any(phrase in corrected_lower for phrase in invalid_phrases)

            if is_invalid:
                # Mark as invalid instead of updating text
                measurements_list[index]['claude_invalid'] = True
                measurements_list[index]['claude_reason'] = corrected_text
                invalid_indices.append(index)
                print(f"  Marked INVALID at index {index}: '{old_text}' - Claude says: '{corrected_text}'")
            else:
                # Apply correction normally
                measurements_list[index]['text'] = corrected_text
                measurements_list[index]['claude_corrected'] = True
                measurements_list[index]['original_ocr'] = old_text
                count += 1
                print(f"  Applied correction at index {index}: '{old_text}' → '{corrected_text}'")

    return count, invalid_indices
