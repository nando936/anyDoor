# Order Processing System

This system standardizes door orders from various user-submitted formats into our consistent finish door list structure.

## Purpose
Every customer submits orders in different formats - different PDFs, forms, layouts, and structures. 
This system takes ANY user-submitted order and converts it to OUR STANDARD finish door list format.

## Workflow Overview

1. **Extract data from PDF** → 2. **Create standard door list** → 3. **Generate shop report** → 4. **Generate cut list** → 5. **Convert to PDFs**

## Main Scripts

### 1. `1_process_new_order.py`
Main processing script that generates all reports from extracted data.
- Creates finish door list (HTML with pictures and specifications)
- Handles opening size to finish size conversion (adds 2x overlay)
- Loads door specifications from specs files
- Generates shop report with technical specifications
- Generates cut list
- Converts all to PDFs
- Saves JSON data for records

### 2. `2_extract_from_pdf.py`
Helper for extracting data from user-submitted PDFs.
- Creates extraction template
- Generates checklist for manual extraction
- Validates extracted data
- Since PDFs vary in format, manual extraction is often required

### 3. `3_convert_html_to_pdf.py`
Utility for converting HTML reports to PDFs using Selenium.

## Step-by-Step Process

### Step 1: Extract Data from PDF
```bash
python 2_extract_from_pdf.py
```
This creates:
- `extraction_template.json` - Fill this with data from PDF
- `extraction_checklist.txt` - Use as guide for manual extraction

### Step 2: Process the Order
```bash
python 1_process_new_order.py
```
This generates:
- `[customer]_[job]_door_list.html` - Editable finish door list with pictures
- `[customer]_[job]_shop_report.html` - Production shop report
- `[customer]_[job]_cut_list.html` - Cut list for materials
- All corresponding PDFs

⚠️ **CRITICAL**: After door list generation, you MUST run `python critical_verification.py`
   - If it fails, FIX the issues and REGENERATE the door list
   - Keep iterating until verification passes with ZERO errors
   - This is NOT optional - verification must pass before any production

## Our Standardized Finish Door List Format

Regardless of how the user submits their order, our finish door list ALWAYS includes:
- Customer information (name, address, phone, email)
- Job details (job name, number, date)
- Door specifications:
  * Door Style number
  * Wood Species (can be mixed)
  * Bore/Door Prep (Yes/No)
  * Hinge Type (full description)
  * Overlay specification
  * Outside Edge details
  * Sticking (Inside) details
  * Panel Cut specification
  * Drawer/Front Type
  * Size Type (Opening or Finish)
- Standardized table with columns:
  * Cabinet # (converted from line numbers)
  * Quantity
  * Width (converted to finish if needed)
  * Height (converted to finish if needed)
  * Type (door/drawer/etc)
  * Material (wood species)
  * Style
  * Notes (critical instructions preserved)

## Important Conventions

1. **Line Numbers = Cabinet Numbers**
   - Line #1 from PDF becomes Cabinet #1 in all reports
   - This convention is maintained throughout the system

2. **Multiple Items per Cabinet**
   - A single cabinet can have multiple doors and/or drawers
   - All items for one cabinet share the same line/cabinet number

3. **Special Notes**
   - Notes referencing specific line numbers must stay with those items
   - Example: "Line #4 - no bore" must be attached to Line #4's entry

4. **Door Pictures and Specifications**
   - Pictures stored in: `order_processing/door pictures/`
   - Picture naming: `[door_number] door pic.JPG` and `[door_number] door profile.JPG`
   - Specifications naming: `[door_number] specs.txt`
   - Example: `231 door pic.JPG`, `231 door profile.JPG`, `231 specs.txt`
   - Place new door pictures and specs in this folder as you get new door styles
   - Pictures automatically display in HTML reports
   - Specifications automatically included in door lists and shop reports
   - Specs should include: stile/rail dimensions, material thickness, sticking size, panel info, etc.

5. **Size Conversion (Opening to Finish)**
   - If order specifies "Opening Sizes", system converts to finish sizes
   - Conversion formula: Finish Size = Opening Size + (2 × Overlay)
   - Example: 14" opening with 1/2" overlay = 15" finish size
   - If order specifies "Finish Sizes", sizes are used as-is

## File Structure

```
order_processing/
├── 1_process_new_order.py      # Main processing script
├── 2_extract_from_pdf.py       # PDF extraction helper
├── 3_convert_html_to_pdf.py    # HTML to PDF converter
├── README.md                    # This file
├── need to process/             # INPUT: Place new PDFs here
├── output/                      # OUTPUT: Finished products organized by job
│   ├── smith_kitchen_302/       # Example: Each job gets its own folder
│   │   ├── smith_kitchen_302_door_list.html
│   │   ├── smith_kitchen_302_door_list.pdf
│   │   ├── smith_kitchen_302_shop_report.html
│   │   ├── smith_kitchen_302_shop_report.pdf
│   │   ├── smith_kitchen_302_cut_list.html
│   │   ├── smith_kitchen_302_cut_list.pdf
│   │   └── smith_kitchen_302_data.json
│   └── [next_customer_job]/
├── processed/                   # ARCHIVE: Original PDFs after processing
├── door pictures/               # RESOURCES: Door style pictures and specifications
├── templates/                   # HTML templates (if needed)
├── scripts/                     # Additional utility scripts
├── examples/                    # Example orders
└── archive/                     # Old files and examples
```

## Workflow for New Orders

1. **Place new PDF in `need to process/` folder**
   - User places: `order_processing/need to process/smith_kitchen_order.pdf`

2. **Claude processes the order**
   - Extracts data from PDF
   - Creates subfolder in `output/` named after customer and job
   - Example: `output/smith_kitchen_302/`

3. **Generated files go to `output/[customer_job]/` folder**
   - All reports generated in the job-specific folder
   - Includes: door list, shop report, cut list (HTML & PDF)
   - Plus JSON data file for records

4. **🔴 CRITICAL VERIFICATION LOOP - MUST PASS BEFORE PROCEEDING 🔴**
   
   **Run the critical verification script:**
   ```bash
   python critical_verification.py
   ```
   
   This performs a COMPLETELY INDEPENDENT re-analysis:
   - **STARTS FROM SCRATCH** - Does not trust any previous processing
   - **RE-READS THE ORIGINAL ORDER** - Fresh extraction from the source
   - **VERIFIES EVERYTHING**:
     * Quantities - each cabinet's quantity must match exactly
     * Sizes - recalculates conversions independently:
       - If original says "Opening Sizes": verifies (2 × overlay) was added
       - If original says "Finish Sizes": verifies sizes match exactly
     * Materials - checks wood species for every item
     * Notes - ensures ALL special instructions transferred:
       - "no bore" / "no hinge boring" - CRITICAL for trash drawers
       - "trash drawer" notes
       - Cabinet number references
     * Cabinet numbers - confirms line numbers became cabinet numbers
   
   **IF VERIFICATION FAILS - DO NOT PROCEED:**
   1. **FIX THE EXTRACTION** - Go back to the data extraction step
   2. **FIX THE PROCESSING** - Update `1_process_new_order.py` if needed
   3. **REGENERATE DOOR LIST** - Create a new door list with fixes
   4. **RUN VERIFICATION AGAIN** - Repeat until it passes
   
   **THIS IS A LOOP PROCESS:**
   ```
   Extract Data → Generate Door List → Run Verification
         ↑                                    |
         |                                    |
         |← If Failed: Fix & Regenerate ←────|
   ```
   
   - **MISTAKES ARE EXTREMELY EXPENSIVE** after production
   - **NEVER SKIP VERIFICATION** - Must show "[OK] VERIFICATION COMPLETE"
   - **KEEP ITERATING** until all checks pass
   - Only proceed to production when verification shows ZERO errors

5. **Original PDF moves to `processed/` folder**
   - Keeps `need to process/` clean for new orders
   - `processed/` serves as archive of original PDFs

## Folder Purposes:
- **need to process/** = Inbox for new orders
- **output/** = Finished products, organized by job
- **processed/** = Archive of original PDFs
- **door pictures/** = Resource library of door styles

## Data Format

### Customer Info Structure
```json
{
  "name": "Customer Name",
  "address": "Customer Address",
  "phone": "(xxx) xxx-xxxx",
  "email": "customer@email.com",
  "job_name": "Kitchen Remodel",
  "job_number": "302",
  "date": "1/7/2025",
  "wood_species": "White Oak",
  "door_style": "231",
  "hinge_type": "Blum Soft Close",
  "overlay": "1/2\" Overlay"
}
```

### Door Items Structure
```json
[
  {
    "line": 1,
    "cabinet": 1,
    "qty": 2,
    "width": "14 3/8",
    "height": "24 3/4",
    "type": "door",
    "material": "White Oak",
    "notes": ""
  }
]
```

## Formulas Used

### Stiles Calculation
- Length = Door Height + 1/4"
- Width = 2 3/8"
- Quantity = 2 per door/drawer

### Rails Calculation
- Length = Door Width - 4 1/2" + 3/4"
- Width = 2 3/8"
- Quantity = 2 per door/drawer

### Material Calculation
- Linear inches = (Width + Height) × 2 × Quantity
- 8-foot pieces = Total inches ÷ 96" (rounded up)

## Testing

To test with sample data:
```bash
python 1_process_new_order.py
```
This will create sample output files to verify the system is working.

## Notes

- Always verify dimensions are in fraction format (e.g., "14 3/8")
- Door types: "door", "drawer", "false_front"
- False fronts don't require rails/stiles
- Each door requires 2 hinges
- All measurements in inches

---

For questions or issues, contact: fernando@theraisedpaneldoor.com