# Cabinet Opening Detection Pipeline

## Overview
This pipeline processes cabinet construction images to automatically detect and mark cabinet openings that need doors or drawer fronts. It uses Google Vision API for OCR and custom algorithms for measurement pairing.

## Pipeline Scripts (in order)

### 0. `extract_any_page.py` (Preprocessing)
**Purpose**: Extracts specific pages from PDF files for processing.

**What it does**:
- Extracts a single page from a PDF as a PNG image
- Maintains high resolution for accurate OCR
- Saves to current working directory as `page_XX.png`

**IMPORTANT - Proper Extraction Process**:
1. First navigate to the target page folder
2. Then run the extraction script from that location
3. The script saves to the current working directory

**Correct Usage**:
```bash
# Navigate to the page folder FIRST
cd "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/page_15/"

# Then extract (using relative path to PDF)
python "C:/Users/nando/Projects/anyDoor/Markup Cabinet Images/extract_any_page.py" "../Measures-2025-09-08(17-08).pdf" 15
```

**Result**: Creates `page_15.png` in the `page_15/` folder

## Main Pipeline Scripts

### 1. `measurement_based_detector.py`
**Purpose**: Detects all measurements in the image and classifies them as horizontal (widths) or vertical (heights).

**What it does**:
- Uses Google Vision API to extract text from the image
- Applies OCR error corrections (removes negative signs, fixes decimal points)
- Reconstructs split measurements (e.g., "16 5" + "/" + "8" → "16 5/8")
- Identifies measurement lines to classify as horizontal or vertical
- Outputs: `page_XX_measurements_data.json` and `page_XX_measurements_detected.png`

**Key features**:
- Generic OCR error handling
- 3-item group reconstruction for split fractions
- Automatic skew detection (but not overly aggressive)
- Does NOT create pairings - only detects and classifies

### 2. `proximity_pairing_detector.py`
**Purpose**: Pairs width and height measurements into actual cabinet openings based on spatial proximity.

**What it does**:
- Loads measurements from the JSON file
- Uses proximity-based logic to pair widths with heights
- Two-pass algorithm:
  - First pass: Each width finds its closest height
  - Second pass: Unpaired heights find their closest width
- Stores the actual positions of paired measurements (crucial for duplicates)
- Outputs: `page_XX_openings_data.json`

**Key features**:
- Smart pairing based on spatial relationships
- Handles duplicate measurements by storing exact positions
- Creates actual cabinet openings, not all combinations

### 3. `mark_opening_intersections.py`
**Purpose**: Creates the final marked image showing cabinet openings with numbered markers.

**What it does**:
- Uses Width's X position × Height's Y position to find intersection points
- Places numbered markers (#1, #2, etc.) inside each opening
- Avoids overlapping with measurement text
- Creates finish size table at bottom
- Outputs: `page_XX_ROOMNAME_openings_1-N_marked.png`

**Key features**:
- Uses saved positions from pairing (not text lookup)
- Smart marker placement to avoid text overlap
- Professional output with room name and overlay info

### 4. `process_cabinet_page.py` (Master Script)
**Purpose**: Runs the complete pipeline in sequence.

**What it does**:
- Executes all three scripts in order
- Cleans up temporary files after processing
- Keeps only the final marked image

## Usage

### Run Complete Pipeline:

**IMPORTANT - Master Script Process**:
1. Extract page to its folder (see extraction instructions above)
2. Run master script from that folder location

```bash
# Navigate to the page folder where you extracted the image
cd "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/page_15/"

# Run the master script (it will find the scripts it needs)
python "C:/Users/nando/Projects/anyDoor/Markup Cabinet Images/process_cabinet_page.py" page_15.png
```

**Note**: The master script has been updated to use absolute paths to find the detector scripts regardless of working directory.

### Run Individual Scripts:
```bash
# Step 1: Detect measurements
python measurement_based_detector.py "path/to/page_XX.png"

# Step 2: Pair into openings
python proximity_pairing_detector.py "path/to/folder/"

# Step 3: Create marked image
python mark_opening_intersections.py "path/to/page_XX.png"
```

## Input Requirements
- Clear image of cabinet with measurements marked
- Measurements should be in format: "XX X/X" (e.g., "16 5/8", "27 3/16")
- Room name and overlay info (e.g., "5/8 OL") should be visible in image

## Output Files
- `page_XX_ROOMNAME_openings_1-N_marked.png` - Final marked image with numbered openings
- Temporary files are automatically cleaned up by the master script

## Common Issues and Solutions

### Issue: Duplicate measurements marked incorrectly
**Solution**: The system now stores exact positions in JSON to distinguish between duplicate measurements (e.g., two "16 5/8" measurements).

### Issue: OCR errors (negative signs, decimal points)
**Solution**: Automatic OCR error correction is applied:
- "-32" → "32"
- "16.5/8" → "16 5/8"
- "16.5 8" → "16 5/8"

### Issue: Split measurements not detected
**Solution**: The system reconstructs measurements from multiple pieces:
- "16 5" + "/" + "8" → "16 5/8"
- "16" + "5/8" → "16 5/8"

### Issue: Wrong openings created
**Solution**: Proximity pairing creates actual cabinet openings based on spatial relationships, not all mathematical combinations.

## Environment Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
- Google Cloud Vision API credentials
- Set GOOGLE_CLOUD_API_KEY environment variable

## How the Pairing Logic Works
1. Measurements are classified as horizontal (widths) or vertical (heights) based on nearby lines
2. The proximity pairing detector analyzes spatial arrangement:
   - If dimensions are stacked vertically → Heights are above their widths
   - If dimensions are side-by-side → Uses closest proximity matching
3. Creates actual cabinet openings, typically 3-4 per cabinet, not all combinations

## Notes
- The system is designed to be generic and work across different pages and jobs
- No hardcoded fixes for specific pages
- Handles hand-written measurements with reasonable tolerance
- Automatically detects room name and overlay information from the image