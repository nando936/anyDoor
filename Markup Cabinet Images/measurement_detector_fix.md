# Fix for Second "20 1/16" Detection Issue on Page 3

## Problem Summary
The second "20 1/16" measurement isn't being detected because:
1. Initial OCR reads it as garbled text: `['20', 'ラ', '17', '116']`
2. The group IS correctly centered on the entire bounding box
3. But `verify_measurement_with_zoom` only returns what's at the exact center point ("20") instead of the complete measurement spanning the group area

## Root Cause
In `verify_measurement_with_zoom` function (lines 92-185), when verifying grouped text:
- The function zooms to the correct area (centered on group bounds)
- The zoomed OCR correctly sees "20 1/16" in the full text
- BUT the function only returns text that's at the exact center pixel
- For a wide group, only "20" is at center, even though "20 1/16" spans the area

## Solution
Modify `verify_measurement_with_zoom` to:
1. **First priority**: Check if the full zoomed text contains a complete measurement pattern
2. **Second priority**: Look at all text in the group's area, not just center point
3. **Last resort**: Fall back to center-point checking

## Key Code Changes

In the `verify_measurement_with_zoom` function, after getting the full text (around line 139):

```python
# RIGHT AFTER getting full_text from annotations[0]
# ADD THIS BLOCK to check for complete measurements first:

# Priority 1: Check if a complete measurement exists in the full text
measurement_patterns = [
    r'\b(\d+\s+\d+/\d+)\b',      # "20 1/16"
    r'\b(\d+-\d+/\d+)\b',         # "20-1/16"
]

for pattern in measurement_patterns:
    match = re.search(pattern, full_text)
    if match:
        measurement = match.group(1).replace('-', ' ')
        # Sanity check
        first_num = int(re.match(r'^(\d+)', measurement).group(1))
        if 2 <= first_num <= 100:
            print(f"    Verification: Found complete measurement '{measurement}' in zoomed text")
            return measurement

# THEN continue with the existing center-point logic as fallback...
```

## Expected Result
With this fix, when the group `['20', 'ラ', '17', '116']` is verified:
- Zoom shows full text: "20 1/16"
- Pattern matching finds "20 1/16" as complete measurement
- Returns "20 1/16" instead of just "20"
- Second measurement is properly detected

## Test Case
On page 3, this will detect both:
1. First "20 1/16" at position (495, 1309) ✓ (already working)
2. Second "20 1/16" at position (1186, 1410) ✓ (will work with fix)