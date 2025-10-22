#!/usr/bin/env python3
"""
Test arrow detection using both template matching and Claude verification.
Tests against M4's left ROI.
"""
import cv2
import os
import subprocess
import platform

# Paths
ROI_PATH = "/home/debian/projects/anyDoor/m11_left_roi.png"  # M11 has confirmed arrow
TEMPLATE_PATH = os.path.expanduser("~/Pictures/Left arrow.png")
OUTPUT_PATH = "/home/debian/projects/anyDoor/arrow_match_result.png"


def ask_claude_about_arrow(roi_path):
    """
    Ask Claude via CLI if there's a left arrow in the ROI image.
    Uses Claude CLI with user's Max account (no API needed).
    """
    print("\n" + "="*60)
    print("ASKING CLAUDE ABOUT LEFT ARROW")
    print("="*60)

    # Build prompt
    prompt = f"""Look at this image:

{roi_path}

This is a cropped region from a cabinet measurement diagram showing a horizontal strip where we expect to find a left-pointing arrow (←) that marks the start of a dimension line.

Question: Do you see a LEFT-POINTING ARROW in this image?

Look for a green arrow head pointing LEFT (<) on a horizontal dimension line. The arrow may be small and at the left edge of the image.

If you see a left arrow, provide its approximate location as pixel coordinates (X, Y) where:
- X is the horizontal position from the left edge (0 = left, 192 = right edge)
- Y is the vertical position from the top (0 = top, 52 = bottom)

Answer in this EXACT format:
Line 1: YES or NO
Line 2 (only if YES): X,Y (coordinates of the arrow tip)

Example answer if arrow found at position 50 pixels from left, 25 pixels from top:
YES
50,25

Your answer:"""

    # Call Claude via CLI
    system = platform.system()
    image_dir = os.path.dirname(roi_path)

    if system == 'Windows':
        claude_exe = r'C:\Users\nando\AppData\Roaming\npm\claude.cmd'
        claude_cmd = f'{claude_exe} --print --add-dir "{image_dir}"'
    else:
        # Linux
        claude_exe = 'claude'
        claude_cmd = f'{claude_exe} --print --add-dir "{image_dir}"'

    try:
        print(f"Calling Claude CLI...")
        result = subprocess.run(
            claude_cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
            shell=True
        )

        if result.returncode != 0:
            print(f"ERROR: Claude call failed with code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return None

        # Parse response
        response = result.stdout.strip()
        print(f"\nClaude's response:")
        print(response)

        lines = response.split('\n')

        # Check first line for YES/NO
        if not lines:
            print("? Empty response from Claude")
            return None

        first_line = lines[0].strip().upper()

        if 'YES' in first_line:
            print("✓ Claude detected LEFT ARROW")

            # Try to extract coordinates from second line
            if len(lines) > 1:
                coord_line = lines[1].strip()
                # Parse X,Y format
                import re
                match = re.search(r'(\d+)\s*,\s*(\d+)', coord_line)
                if match:
                    x = int(match.group(1))
                    y = int(match.group(2))
                    print(f"  Arrow location: X={x}, Y={y}")
                    return {'detected': True, 'x': x, 'y': y}
                else:
                    print(f"  (No coordinates provided)")
                    return {'detected': True, 'x': None, 'y': None}
            else:
                return {'detected': True, 'x': None, 'y': None}

        elif 'NO' in first_line:
            print("✗ Claude did NOT detect left arrow")
            return {'detected': False, 'x': None, 'y': None}
        else:
            print(f"? Could not parse Claude's response")
            return None

    except subprocess.TimeoutExpired:
        print("ERROR: Claude call timed out")
        return None
    except Exception as e:
        print(f"ERROR: Claude verification failed: {e}")
        return None


def test_arrow_template(threshold=0.6):
    """Test arrow template matching on M4 left ROI"""

    print("\n" + "="*60)
    print(f"TESTING TEMPLATE MATCHING (threshold={threshold})")
    print("="*60)

    # Load ROI
    roi = cv2.imread(ROI_PATH)
    if roi is None:
        print(f"ERROR: Could not load ROI from {ROI_PATH}")
        return

    print(f"Loaded ROI: {roi.shape}")

    # Load template
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        print(f"ERROR: Could not load template from {TEMPLATE_PATH}")
        return

    print(f"Loaded template: {template.shape}")

    # Check if template fits in ROI
    if template.shape[0] >= roi.shape[0] or template.shape[1] >= roi.shape[1]:
        print(f"ERROR: Template too large for ROI!")
        print(f"  Template: {template.shape[0]}h x {template.shape[1]}w")
        print(f"  ROI: {roi.shape[0]}h x {roi.shape[1]}w")
        return False

    # Perform template matching
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"\nResults:")
    print(f"  Best match confidence: {max_val:.3f}")
    print(f"  Match location: {max_loc}")

    if max_val >= threshold:
        print(f"  ✓ ARROW DETECTED (confidence >= threshold)")
    else:
        print(f"  ✗ NO ARROW (confidence < threshold)")

    # Draw rectangle around match location
    result_img = roi.copy()
    h, w = template.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Use green if match found, red if not
    color = (0, 255, 0) if max_val >= threshold else (0, 0, 255)
    cv2.rectangle(result_img, top_left, bottom_right, color, 2)

    # Add confidence text
    text = f"Conf: {max_val:.3f}"
    cv2.putText(result_img, text, (top_left[0], top_left[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save result
    cv2.imwrite(OUTPUT_PATH, result_img)
    print(f"\nSaved result visualization: {OUTPUT_PATH}")

    return max_val >= threshold


if __name__ == "__main__":
    print("="*60)
    print("ARROW DETECTION TEST - M4 LEFT ROI")
    print("="*60)
    print(f"ROI: {ROI_PATH}")
    print(f"Template: {TEMPLATE_PATH}")

    # Test 1: Ask Claude
    claude_result = ask_claude_about_arrow(ROI_PATH)

    # Test 2: Template matching
    template_result = test_arrow_template(threshold=0.6)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if claude_result and isinstance(claude_result, dict):
        print(f"Claude detection: {claude_result['detected']}")
        if claude_result['detected'] and claude_result['x'] is not None:
            print(f"  Arrow coordinates: ({claude_result['x']}, {claude_result['y']})")
    else:
        print(f"Claude detection: {claude_result}")
    print(f"Template matching (0.6): {template_result}")
    print("="*60)
