# Claude Vision API Arrow Verification - Implementation Summary

## Overview

Added Claude Vision API verification to the `detect_down_arrow()` function in `line_detection.py` to handle cases where only one arrow leg is detected by OpenCV.

## Problem Being Solved

When OpenCV detects lines in a down arrow ROI, it looks for two types of lines:
- **Down-right lines**: 20-88 degrees (right leg of the V)
- **Down-left lines**: 285-360 degrees (left leg of the V)

Sometimes, only one leg type is detected (e.g., only down-right lines, no down-left lines), causing the arrow detection to fail even though a human (or Claude) can clearly see the arrow is present.

## Solution

When exactly one leg type is detected:
1. Call Claude Vision API with the ROI debug image
2. Ask: "Is there a down arrow (arrow pointing down) visible in this image? Answer only YES or NO."
3. If Claude says YES, override the result and return `has_down_arrow=True`
4. If Claude says NO, keep the original result (`has_down_arrow=False`)

## Files Modified

### `/home/nando/projects/anyDoor/raised panel door OS/Process Incoming Door Orders/Inhouse Door Order/line_detection.py`

#### 1. Added Imports (lines 7-14)
```python
import subprocess
import tempfile
import os
import platform
```

#### 2. Added Helper Function `verify_arrow_with_claude()` (lines 17-71)

**Purpose**: Call Claude Vision API to verify if an arrow is present in an ROI image.

**Parameters**:
- `roi_image`: The ROI image (numpy array)
- `direction`: Arrow direction ('up', 'down', 'left', 'right')

**Returns**:
- `bool`: True if Claude detects an arrow, False otherwise

**How it works**:
1. Saves ROI image to temporary file
2. Calls Claude CLI with prompt: "Is there a {direction} arrow visible in this image? Answer only YES or NO."
3. Parses response and returns True if response contains "YES"
4. Cleans up temporary file
5. Returns False on any errors

#### 3. Modified `detect_arrow_in_roi()` Function (lines 1177-1214)

Added logic after line 1175 (after calculating `has_correct_arrow` and `has_wrong_arrow`):

**When triggered**:
- No arrow found by OpenCV (`not has_correct_arrow`)
- Direction is 'down'
- At least 2 lines detected

**What it does**:
1. Counts lines in each leg direction:
   - Right leg: 20-88 degrees (adjusted for rotation)
   - Left leg: 285-360 degrees (adjusted for rotation)

2. Detects single-leg case:
   - `(right_leg_count > 0 and left_leg_count == 0)` OR
   - `(left_leg_count > 0 and right_leg_count == 0)`

3. Calls Claude verification:
   - `claude_result = verify_arrow_with_claude(roi_image, direction)`

4. Overrides result if Claude confirms:
   - If `claude_result == True`: sets `has_correct_arrow = True`
   - Otherwise: keeps `has_correct_arrow = False`

## Console Output

When single-leg case is detected, you'll see:

```
        [SINGLE LEG DETECTED] Found 2 down-right and 0 down-left lines
        [CLAUDE VERIFICATION] Calling Claude to verify down arrow presence...
        [CLAUDE VERIFICATION] Arrow detected - OVERRIDE: has_down_arrow=True
```

OR

```
        [SINGLE LEG DETECTED] Found 0 down-right and 3 down-left lines
        [CLAUDE VERIFICATION] Calling Claude to verify down arrow presence...
        [CLAUDE VERIFICATION] No arrow detected - keeping has_down_arrow=False
```

## Testing

Created test scripts to verify implementation:

1. **test_line_detection_import.py** - Verifies module imports without errors
2. **test_claude_verification_logic.py** - Tests the `verify_arrow_with_claude()` function

Both tests passed successfully.

## Limitations

1. **Only for 'down' direction**: Currently only triggers for down arrows. Can be extended to other directions if needed.

2. **Requires Claude CLI**: Must have Claude CLI installed and accessible:
   - Linux: `claude` in PATH
   - Windows: `C:\Users\nando\AppData\Roaming\npm\claude.cmd`

3. **Performance**: Adds ~1-2 seconds per single-leg detection (Claude API call overhead)

4. **Minimum lines requirement**: Requires at least 2 lines detected to trigger (prevents false positives on empty ROIs)

## Future Enhancements

Potential improvements:
- Extend to up/left/right arrows
- Cache Claude results for identical ROIs
- Add confidence threshold instead of YES/NO
- Save Claude-verified ROIs for review
- Add fallback strategies if Claude API is unavailable

## Edge Cases Handled

1. **Claude CLI not available**: Returns False, keeps original result
2. **Claude timeout**: Returns False after 30 seconds
3. **Empty ROI**: Not triggered (requires >= 2 lines)
4. **Both legs present**: Not triggered (only for single-leg cases)
5. **No lines detected**: Not triggered (requires >= 2 lines)
6. **Temp file cleanup**: Always attempts cleanup, silently fails if already deleted

## Integration Notes

- No changes required to calling code
- Transparent upgrade - existing behavior unchanged for normal cases
- Only activates for edge cases (single-leg detections)
- Debug logging shows when Claude is called and the result
- Compatible with existing debug image saving logic
