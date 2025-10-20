#!/usr/bin/env python3
"""
Claude-based OCR verification for suspicious measurements.
Uses Claude CLI to verify measurements that look incorrect.
"""

import subprocess
import os
import re


def matches_valid_measurement_pattern(text):
    """
    Check if text matches the valid measurement pattern.

    Valid patterns:
    - Whole number: "30"
    - Whole + fraction: "30 1/4"
    - Whole + notation: "30 F", "30 O", "30 NH"
    - Whole + fraction + notation: "30 1/4 F", "30 1/4 O", "30 1/4 NH"

    Args:
        text: Measurement text to validate

    Returns: True if matches valid pattern, False otherwise
    """
    if not text:
        return False

    # Pattern: whole number, optional fraction, optional notation (F, O, NH)
    # ^\d+(\s+\d+/\d+)?(\s+(F|O|NH))?$
    pattern = r'^\d+(\s+\d+/\d+)?(\s+(F|O|NH))?$'
    return re.match(pattern, text.strip()) is not None


def is_partial_measurement(text, raw_ocr=None):
    """
    Detect if a measurement appears to be partially cropped/cut-off.

    Indicators of partial measurements:
    - Ends with "/" but no denominator (e.g., "6 1/")
    - Space + single digit at end without slash (e.g., "6 1", "20 3")
    - Raw OCR suggests incomplete fraction

    Args:
        text: Cleaned measurement text
        raw_ocr: Raw OCR text before cleaning (optional)

    Returns: (is_partial, recommended_expansion_pixels)
    """
    if not text:
        return False, 0

    # Pattern 1: Ends with "/" but no denominator (e.g., "6 1/")
    if re.search(r'/\s*$', text):
        return True, 150  # Expand to catch denominator + possible notation

    # Check raw OCR as well
    if raw_ocr and re.search(r'/\s*$', raw_ocr):
        return True, 150

    # Pattern 2: Space + single digit at end without slash (e.g., "6 1", "20 3")
    # This looks like an incomplete fraction (missing "1/16" or similar)
    if re.search(r'\d+\s+\d$', text):
        return True, 150  # Expand to catch "/denominator" + possible notation

    # Pattern 3: Very short text that doesn't match valid pattern
    # (already caught by is_suspicious_measurement, but check here too)
    if len(text.strip()) < 3 and not re.match(r'^\d+$', text):
        return True, 100

    return False, 0


def is_suspicious_measurement(text, raw_ocr=None):
    """
    Detect if a measurement looks suspicious and might need Claude verification.

    Uses two-stage validation:
    1. Check raw OCR against valid pattern (if provided)
    2. If raw fails, check cleaned text against valid pattern
    3. If both fail, mark as suspicious
    4. If cleaned text passes, run additional fraction validation checks

    Args:
        text: Cleaned measurement text
        raw_ocr: Raw OCR text before cleaning (optional)

    Returns: (is_suspicious, reason)
    """
    if not text:
        return False, None

    # TWO-STAGE VALIDATION
    # Stage 1: Check raw OCR against valid pattern (if provided)
    raw_ocr_valid = False
    if raw_ocr:
        raw_ocr_valid = matches_valid_measurement_pattern(raw_ocr)

    # Stage 2: Check cleaned text against valid pattern
    cleaned_text_valid = matches_valid_measurement_pattern(text)

    # If cleaned text doesn't match valid pattern, it's suspicious
    # (even if raw OCR was valid, cleaning may have broken it)
    if not cleaned_text_valid:
        return True, "invalid_measurement_format"

    # If cleaned text is valid, continue with additional fraction validation checks below

    # Check raw OCR for suspicious patterns (before cleaning removed them)
    if raw_ocr and not raw_ocr_valid:
        # Pattern 0: Leading dash or minus in RAW OCR (e.g., "-9-" which gets cleaned to "9")
        # If raw OCR had issues but cleaning fixed it (cleaned text passed validation),
        # we still want to note it for logging purposes
        if raw_ocr.startswith('-') or raw_ocr.startswith('−'):
            # Cleaned text already passed validation (line 62), so cleaning fixed it
            # This is just for information - not marking as suspicious
            pass

    # Additional validation for measurements that passed the basic pattern check
    # These checks catch logically invalid measurements (impossible fractions, too large, etc.)

    # Pattern 1: Impossible fraction (numerator >= denominator, except 1/1 which shouldn't appear)
    # Check all fractions in the text
    for match in re.finditer(r'(\d+)/(\d+)', text):
        num, denom = int(match.group(1)), int(match.group(2))
        if num >= denom:
            return True, "impossible_fraction"

    # Pattern 2: Measurement over 120 inches (likely OCR error - missing space/fraction)
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

    # Pattern 3: Invalid fraction denominator (not in 1/16 increments)
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
- Look carefully for notation letters (F, O, NH) after the measurement
- If you see any notation in the image, include it in your response

NOTATION MEANINGS:
- F = Finished size (the final size of the door)
- O = Opening size (the size of the cabinet opening)
- NH = No hinges (door has no hinge holes)

Each image should show a measurement in green text. Valid formats:
- Whole number: "44"
- Whole number + fraction: "33 5/8" or "15 1/2"
- Whole number + notation: "30 F" or "24 O" or "18 NH"
- Whole number + fraction + notation: "30 1/4 F" or "18 1/2 O" or "24 3/4 NH"

IMPORTANT: If you see a notation letter in the image, include it in your response.
Example: if image shows "6 1/2 F", return "6 1/2 F" (not just "6 1/2")

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
        Tuple: (number of corrections applied, list of invalid indices, list of corrected indices)
    """
    if not corrections:
        return 0, [], []

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
    corrected_indices = []  # Track which measurements were actually corrected

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

                # Re-check for F notation after Claude correction
                if corrected_text.endswith(' F'):
                    measurements_list[index]['is_finished_size'] = True
                    # Remove ' F' from display text to avoid duplication
                    measurements_list[index]['text'] = corrected_text[:-2]
                    print(f"  Re-detected F notation after Claude correction")

                count += 1
                corrected_indices.append(index)  # Track this correction
                print(f"  Applied correction at index {index}: '{old_text}' → '{corrected_text}'")

    return count, invalid_indices, corrected_indices
