# Inhouse Door Order - Debug Files Guide

## Overview

When processing cabinet measurement images, the pipeline creates **4 separate debug files** (one per phase) that contain the complete stdout log for that phase. These files are essential for debugging and understanding what happened during processing.

## Debug File Structure

### File Naming Convention

```
{base_name}_phase_{phase_number}_{phase_label}.txt
```

**Example for page_8.png:**
- `page_8_phase_1-2.5_finding_and_verification.txt`
- `page_8_phase_3_classification.txt`
- `page_8_phase_4_pairing.txt`
- `page_8_phase_5_visualization.txt`

## What's In Each Phase File

### Phase 1-2.5: Finding and Verification

**File:** `{base_name}_phase_1-2.5_finding_and_verification.txt`

**Contents:**

#### PHASE 1: Finding Interest Areas
- HSV preprocessing results
- Overlay detection (e.g., "11/16 OL")
- Room name detection (e.g., "Bath #4")
- All text items found by Google Vision OCR
- Grouping and merging of text areas
- Center coordinates for each interest area

#### PHASE 2: Zoom Verification
For each measurement candidate:
- OCR re-verification at 3x zoom
- Calculated text bounds (left, right, top, bottom)
- Text width and height in pixels
- Whether measurement is finished size (F) or opening size
- Verified measurement value
- Special notation detection (F, O, NH, etc.)

#### PHASE 2.5: Claude Verification
- Suspicious measurement detection
- Claude API corrections for OCR errors
- Applied corrections summary

**Use this file when:**
- Measurements are missing from detection
- OCR is reading wrong values
- Room name or overlay not detected correctly
- Text grouping issues

---

### Phase 3: Classification

**File:** `{base_name}_phase_3_classification.txt`

**Contents:**

For each verified measurement:

1. **Text Position and Bounds**
   - Measurement ID (M1, M2, etc.)
   - Text value and pixel coordinates
   - Bounding box dimensions

2. **ROI Setup**
   - H-ROI coordinates (horizontal region of interest)
   - V-ROI coordinates (vertical region of interest)
   - ROI dimensions and clearance calculations

3. **Line Detection**
   - Lines found in H-ROI
   - Lines found in V-ROI
   - Line angles and positions

4. **Arrow Detection**
   - Left arrow search results
   - Right arrow search results
   - Arrow confidence scores

5. **Rotation Attempts**
   - Tests at multiple angles: 0°, ±22.5°, ±30°, ±45°
   - For each rotation:
     - Lines found at that angle
     - Angle deviation calculation
     - Score calculation: `score = abs(deviation)`
   - **BEST ROTATION** selection (lowest score)

6. **Classification Decision**
   - Final classification: WIDTH / HEIGHT / UNCLASSIFIED
   - Reasoning for classification
   - Extent calculation results (line endpoints)

7. **Summary**
   - Total WIDTH measurements
   - Total HEIGHT measurements
   - Total UNCLASSIFIED measurements

**Use this file when:**
- Measurement classified incorrectly (width vs height)
- Need to see rotation scoring details
- Line detection not working
- Arrow detection failing
- Understanding why measurement is UNCLASSIFIED

---

### Phase 4: Pairing

**File:** `{base_name}_phase_4_pairing.txt`

**Contents:**

#### Pairing Strategy V2
- Algorithm description and approach

#### Down Arrow Scanning
- Down arrow detection results
- Scan region details
- Arrow positions found

#### Width-Height Pairing
For each opening:
- Width measurement and position
- Height measurement and position
- Pairing distance calculation
- Pairing logic explanation

#### Drawer Configuration Analysis
- Bottom width detection
- Drawer height threshold (default 10.0")
- Multi-drawer configuration detection
- Bottom width special handling

#### Unpaired Heights
- Heights that couldn't be paired
- Attempted pairing with widths above
- Reasons for pairing failures

#### Cabinet Opening Specifications
- Final list of paired openings
- Opening numbers (sequential)
- Width × Height specifications
- Position coordinates
- Pairing distances

#### Summary Statistics
- Total measurements found
- Total openings paired
- Measurement counts by type

**Use this file when:**
- Openings not pairing correctly
- Width and height matched incorrectly
- Drawer configuration issues
- Missing openings in final output
- Understanding pairing logic

---

### Phase 5: Visualization

**File:** `{base_name}_phase_5_visualization.txt`

**Contents:**

#### Visualization Toggles
- Which visualization layers are enabled
- Panel display settings
- ROI display settings

#### Files Saved
- Annotated image path
- JSON output paths
- Unified format output

#### Pipeline Completion
- Success/failure status
- Final summary

**Use this file when:**
- Visualization not generating
- Missing output files
- Understanding what was saved

---

## Debugging Workflow

### Common Debugging Questions

#### "Why was M11 classified as WIDTH instead of HEIGHT?"

```bash
# Look in Phase 3 file
grep -A 100 "Analyzing measurement 11:" page_8_phase_3_classification.txt
```

Look for:
- Rotation scores at each angle
- Which angle had the best (lowest) score
- What lines were found
- Final classification decision

#### "What was the rotation score at -30° for M11?"

```bash
# Search Phase 3 file
grep -A 100 "Analyzing measurement 11:" page_8_phase_3_classification.txt | grep "angle=-30"
```

#### "Why didn't opening #5 get paired?"

```bash
# Look in Phase 4 file
grep -A 50 "WIDTH measurements:" page_8_phase_4_pairing.txt
grep -A 50 "HEIGHT measurements:" page_8_phase_4_pairing.txt
grep -A 100 "Pairing" page_8_phase_4_pairing.txt
```

Look for:
- Available widths and heights
- Pairing attempts
- Distance calculations
- Unpaired measurements section

#### "Why is '28 1/16' not being detected?"

```bash
# Look in Phase 1-2.5 file
grep "28" page_8_phase_1-2.5_finding_and_verification.txt
```

Look for:
- Whether it appears in initial text items
- Whether it got filtered out
- OCR verification results
- Claude correction attempts

#### "What measurements are on the page?"

```bash
# Look in Phase 1-2.5 file
grep "✓ Verified:" page_8_phase_1-2.5_finding_and_verification.txt
```

---

## Debug Protocol

### When User Asks ANY Question About Processing:

1. **ALWAYS check the appropriate phase file FIRST**
2. **DO NOT re-run the page** unless files are missing/outdated
3. **Provide ONLY facts from debug logs** - NO speculation
4. **Quote exact output** from the relevant phase file

### Example Response Format:

```
M11 '18 1/16' classification (from phase_3_classification.txt):

Rotation scores:
- 0°:     score=45.2 (h-lines at angle=-45.2°)
- -22.5°: score=8.0  (h-lines at angle=-14.5°) ← BEST
- -30.0°: score=25.2 (h-lines at angle=-4.8°)
- -45.0°: score=12.3 (h-lines at angle=-32.7°)

M11 chose -22.5° rotation because it had the lowest score (8.0).

Classification: WIDTH (based on horizontal lines found at -22.5°)
```

---

## File Locations

**Debug files are saved in the SAME directory as the input image.**

Example:
```
Input:  /home/nando/onedrive/customers/raised-panel/page_8.png
Output: /home/nando/onedrive/customers/raised-panel/page_8_phase_*.txt
```

---

## Tips

- Phase files are created sequentially as pipeline runs
- If pipeline crashes in Phase 3, only phases 1-2.5 and 3 files exist
- Each phase file is independent and complete for that phase
- All console output also appears in the terminal (files are mirrored output)
- Files use UTF-8 encoding to handle fractions and special characters

---

## Quick Reference: What File to Check

| Question | Check This File |
|----------|----------------|
| Measurement not detected? | Phase 1-2.5 |
| OCR reading wrong? | Phase 1-2.5 |
| Wrong classification (W vs H)? | Phase 3 |
| Rotation scores? | Phase 3 |
| Line detection issues? | Phase 3 |
| Pairing problems? | Phase 4 |
| Missing openings? | Phase 4 |
| Drawer configuration? | Phase 4 |
| Visualization issues? | Phase 5 |
| What files were saved? | Phase 5 |
