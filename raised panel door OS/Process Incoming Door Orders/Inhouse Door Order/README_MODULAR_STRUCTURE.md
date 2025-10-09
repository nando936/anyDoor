# Modular Cabinet Measurement Detection

## Overview

This is a **modular reorganization** of the original `measurement_detector_test.py` (3854 lines) into **8 separate modules** for better maintainability and reusability.

## Module Structure

```
measurement_config.py          - Configuration constants (56 lines)
image_preprocessing.py         - HSV filtering, OpenCV text detection (305 lines)
text_detection.py              - OCR, room/overlay extraction, grouping (442 lines)
measurement_verification.py    - Zoom verification with full bounds (637 lines)
line_detection.py              - Line detection, classification with rotation (752 lines)
measurement_pairing.py         - Pairing widths/heights into openings (364 lines)
visualization.py               - Complete visualization with markers/legends (864 lines)
main.py                        - Full 5-phase pipeline (275 lines)
```

## Module Descriptions

### 1. `measurement_config.py`
**All configuration in one place**
- `HSV_CONFIG` - Color range for green text
- `GROUPING_CONFIG` - Proximity thresholds
- `ZOOM_CONFIG` - Zoom parameters
- `VALIDATION_CONFIG` - Min/max measurement values
- `ROOM_PATTERNS` - Room name regex patterns
- `EXCLUDE_PATTERNS` - Non-measurement text
- `OVERLAY_PATTERN` - Overlay notation regex

### 2. `image_preprocessing.py`
**Image processing and OpenCV text detection**
- `apply_hsv_preprocessing()` - HSV color filtering
- `find_opencv_supplemental_regions_better()` - Better text detection
- `find_opencv_supplemental_regions()` - Advanced text detection with line filtering

### 3. `text_detection.py`
**OCR and text extraction**
- `find_room_and_overlay()` - Extract room names and overlay info
- `extract_measurements_from_text()` - Parse measurement patterns
- `find_interest_areas()` - Phase 1: Find text regions via OCR
- `merge_close_centers()` - Merge duplicate detections

### 4. `measurement_verification.py`
**Zoom verification**
- `verify_measurement_at_center()` - Simple zoom verification
- `verify_measurement_at_center_with_logic()` - Zoom with detailed logic tracking
- Returns precise measurements with bounds

### 5. `line_detection.py`
**Line detection and classification**
- `find_lines_near_measurement()` - Find dimension lines (H/V) near measurements with ROI rotation
- `classify_measurements_by_lines()` - Classify as WIDTH/HEIGHT/UNCLASSIFIED
- Arrow detection for dimension markers
- Rotated ROI extraction for skewed images
- Comprehensive line filtering and validation

### 6. `measurement_pairing.py`
**Pairing logic**
- `pair_measurements_by_proximity()` - Pair widths/heights into openings
- `add_fraction_to_measurement()` - Fraction arithmetic for overlay calculations
- `find_clear_position_for_marker()` - Intelligent marker placement avoiding overlaps
- Handles both stacked and side-by-side cabinet arrangements
- Tracks unpaired measurements

### 7. `visualization.py`
**Complete visualization system**
- `create_visualization()` - Draw all visual elements with full control
- Classification labels (WIDTH/HEIGHT/UNCLASSIFIED)
- ROI visualization for line detection regions
- Opening markers with numbered circles and dimension text
- Legend panel with finish sizes and overlay calculations
- Special notation support (NO HINGES, etc.)
- Timestamp and page numbering
- Intelligent label placement avoiding overlaps

### 8. `main.py`
**Complete pipeline orchestration**
- `main()` - Full 5-phase detection pipeline
- Phase 1: Find interest areas via OCR
- Phase 2: Zoom verification of measurements
- Phase 3: Line-based classification (WIDTH/HEIGHT)
- Phase 4: Pairing into cabinet openings
- Phase 5: Visualization with all annotations
- Command-line interface with page numbering support
- JSON output of results

## Implementation Status

### ✅ **ALL MODULES FULLY IMPLEMENTED**

All 8 modules are complete and production-ready:

- ✅ `measurement_config.py` - All configuration constants
- ✅ `image_preprocessing.py` - HSV filtering and OpenCV text detection
- ✅ `text_detection.py` - Complete OCR and text extraction
- ✅ `measurement_verification.py` - Zoom verification with precise bounds
- ✅ `line_detection.py` - Full line detection with rotation support
- ✅ `measurement_pairing.py` - Complete pairing logic
- ✅ `visualization.py` - Full visualization system
- ✅ `main.py` - Complete 5-phase pipeline

### 🎯 **Ready for Production Use**

The modular system is fully functional and can process cabinet measurement images end-to-end.

## Benefits of This Structure

### ✅ **Maintainability**
- Each module has a clear, single responsibility
- Easy to find and fix bugs in specific areas
- Changes to config don't require touching code

### ✅ **Reusability**
- Other projects can import just what they need
- `visualization.py` can be used across all 17 scripts
- Configuration can be shared

### ✅ **Testability**
- Each module can be tested independently
- Mock dependencies easily for unit tests
- Easier to debug specific pipeline stages

### ✅ **Readability**
- 8 files of ~100-600 lines each
- vs. 1 file of 3854 lines
- Clear module boundaries

## Usage

```bash
# Process a cabinet measurement image:
python main.py "path/to/image.png"

# Start opening numbers at a specific value:
python main.py "path/to/image.png" 5  # Start numbering at opening 5

# Example with network path:
python main.py "//vmware-host/Shared Folders/project/page_1.png"
```

### Output Files

The script generates:
- `*_test_viz.png` - Annotated visualization with all detections and markers
- `*_cabinet_openings.json` - Structured data with all measurements and openings
- `*_debug.txt` - Complete debug log with all classification decisions

## Debugging Measurement Classifications

### Debug Log Analysis Protocol

**When asked "why was measurement X classified as WIDTH/HEIGHT/UNCLASSIFIED?":**

Provide ONLY the actual data from the `*_debug.txt` file for that measurement:

1. **For each rotation attempt (0°, +22.5°, -22.5°):**
   - What lines were detected in each ROI (Left H-ROI, Right H-ROI, Top V-ROI, Bottom V-ROI)
   - The exact angles found for each line
   - How many lines passed/failed filtering (with reasons)
   - Whether arrows were detected

2. **The exact logic that led to classification:**
   - Which ROIs had indicators (L-horiz, R-horiz, T-vert, B-vert)
   - The final classification decision and why

3. **NO speculation, NO "likely" responses** - only facts from the debug log

### Example Debug Log Response:

```
M11 '18 1/16' at (630, 2000):

Attempt 1 (0° rotation):
- Left H-ROI: 0 lines
- Right H-ROI: 1 line (angle=86.3°)
- Top V-ROI: 5 lines
- Bottom V-ROI: 9 lines (all filtered - angles 0-38°, not in 55-125° range)
- Result: UNCLASSIFIED (L-horiz:False R-horiz:False)

Attempt 2 (+22.5° rotation):
- Top V-ROI: 8 lines, up-arrow=True
- Result: UNCLASSIFIED (T-vert:True B-vert:False)

Attempt 3 (-22.5° rotation):
- Left H-ROI: 0 lines
- Right H-ROI: 1 line (angle=-13.8°)
- Top V-ROI: 1 v-line (angle=-69.1°, adjusted=-91.6°)
- Bottom V-ROI: 1 v-line (angle=-68.6°, adjusted=-91.1°)
- Result: "Found vertical indicators: top(1 v-lines) + bottom(1 v-lines) → HEIGHT"
- Final: HEIGHT (via -22.5° rotation)
```

This data-driven approach ensures accurate troubleshooting based on what the script actually detected.

## Original File Reference

All code comes from: `Markup Cabinet Images/measurement_detector_test.py`

**Line Mapping (Original → Modular):**
```
Lines 27-82     → measurement_config.py (56 lines)
Lines 85-436    → image_preprocessing.py (305 lines)
Lines 438-790   → text_detection.py (442 lines)
Lines 792-1445  → measurement_verification.py (637 lines)
Lines 1446-2178 → line_detection.py (752 lines)
Lines 2180-2746 → measurement_pairing.py (364 lines)
Lines 2747-3576 → visualization.py (864 lines)
Lines 3576-3854 → main.py (275 lines)
```

**Total**: 3854 lines → 3695 lines across 8 modules (modularization improved organization with similar line count)

## Features

### Advanced Capabilities
- **HSV Color Filtering** - Isolates green dimension text on brown backgrounds
- **Dual OCR Sources** - Google Vision API + OpenCV text detection
- **Smart Grouping** - Proximity-based measurement grouping
- **Zoom Verification** - 3x magnification for accurate OCR
- **Line Classification** - Detects horizontal/vertical dimension lines
- **ROI Rotation** - Handles skewed images up to 22.5°
- **Arrow Detection** - Recognizes dimension arrow markers
- **Intelligent Pairing** - Matches widths with heights by proximity
- **Overlay Calculations** - Automatic finish size calculations
- **Smart Marker Placement** - Avoids overlapping labels
- **Special Notations** - Handles NH (No Hinges), etc.
- **JSON Export** - Structured output for downstream processing
