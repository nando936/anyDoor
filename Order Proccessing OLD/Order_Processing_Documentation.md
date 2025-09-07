# Cabinet Door Order Processing Documentation

## Overview
This document explains the complete process of taking a user-submitted order form PDF and generating production documents including a Shop Report and Cut List for The Raised Panel Door Factory.

## Input Document
**User Submitted Order Form** (`user_submited_ORDER_FORM_.pdf`)
- Contains customer information
- Lists door/drawer specifications
- Includes quantities, dimensions, and material specs

## Process Workflow

### Step 1: Extract Data from Order Form (Visual/Manual)

#### Customer Information
- **Name**: Extract from "Name:" field
- **Address**: Extract from "Address:" field  
- **Phone**: Extract from "Phone #:" field
- **Email**: Extract from "Email:" field
- **Job Name**: Extract from "Job Name:" field
- **Date**: Extract from "Date:" field

#### Door Specifications
- **Wood Species**: Extract (e.g., "White Oak")
- **Door Style Number**: Extract door # (e.g., "308")
- **Edge Type**: Outside edge (e.g., "Special") and Inside/Sticky (e.g., "Bevel")
- **Panel Cut**: Extract specification (e.g., "3/8" Plywood (Flat Panel ONLY)")
- **Overlay Type**: Extract (e.g., "1/2" Overlay")
- **Bore/Door Prep**: Yes/No checkbox status
- **Hinge Type**: Extract specification (e.g., "Blum Soft Close Frameless 1/2"OL")
- **Door Sizes**: Opening Sizes vs Finish Sizes (checkbox)
- **Drawer/Front Type**: Extract (e.g., "5 piece")

#### Line Items
Extract each row from the order table:
- **Item #**: Sequential number
- **QTY**: Quantity of doors/drawers
- **Width**: Width dimension
- **Height**: Height dimension  
- **Type**: "doors" or "drawer"
- **Details**: Any additional specifications

### Step 1.5: Create Standard Door List (Manual Extraction)

Since user-submitted PDFs can vary significantly in format and complexity, manual extraction and standardization is required.

**IMPORTANT CONVENTIONS**: 
- Line numbers from the PDF order form become Cabinet numbers in our standard door list and all subsequent reports
- For example, Line #1 in the PDF becomes Cabinet #1 in the shop report and cut list
- **Multiple items per cabinet**: A single line/cabinet number can have multiple doors and/or drawers
- Some PDFs may list all items for one cabinet on a single line (e.g., "2 doors, 3 drawers" for Cabinet #5)
- These should all be grouped under the same cabinet number in our standard list
- **Special notes association**: Any special notes that reference a specific line number or cabinet number MUST be included with that line item in our standard door list (e.g., "Line #4 - no bore" must be attached to Line #4's entry)

#### Manual Data Extraction Process
1. **Examine the PDF visually** to identify all door/drawer entries
2. **Extract each line item** with its specifications
3. **Create an HTML file** with door pictures and editable fields for review
4. **Review and make manual corrections** directly in the HTML (contenteditable fields)
5. **Convert to JSON structure** for automated processing
6. **Pictures automatically included** from `C:\Users\nando\Projects\anyDoor\door pictures\` folder based on door style number

#### Standard Door List Formats

##### HTML Format (Primary - includes pictures and is editable)
Create an HTML file that includes door pictures and is both viewable and editable:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Standard Door List - [Job Name]</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 15px; margin-bottom: 20px; }
        .door-pictures { border: 2px solid #333; padding: 10px; margin-bottom: 20px; }
        .door-pictures img { max-width: 300px; margin: 10px; border: 1px solid #ccc; }
        .line-item { border: 1px solid #ccc; padding: 10px; margin-bottom: 15px; background: #f9f9f9; }
        .editable { background: white; border: 1px dashed #999; padding: 2px; min-width: 50px; display: inline-block; }
        .notes { background: #ffffcc; padding: 5px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>STANDARD DOOR LIST</h1>
        <p><strong>Job:</strong> <span class="editable" contenteditable="true">[Job Name]</span></p>
        <p><strong>Date:</strong> <span class="editable" contenteditable="true">[Date]</span></p>
        <p><strong>Customer:</strong> <span class="editable" contenteditable="true">[Customer Name]</span></p>
        <p><strong>Door Style:</strong> <span class="editable" contenteditable="true">#302</span></p>
    </div>
    
    <div class="door-pictures">
        <h2>Door Style Reference Pictures</h2>
        <img src="door pictures/302 door pic.JPG" alt="Door Style 302">
        <img src="door pictures/302 door profile.JPG" alt="Door Profile 302">
        <p><em>Pictures from: C:\Users\nando\Projects\anyDoor\door pictures\</em></p>
    </div>
    
    <div class="line-items">
        <h2>Line Items (Cabinet Numbers)</h2>
        
        <div class="line-item">
            <h3>Line #1 (Cabinet #1)</h3>
            <p>Qty: <span class="editable" contenteditable="true">2</span></p>
            <p>Width: <span class="editable" contenteditable="true">14 3/8</span></p>
            <p>Height: <span class="editable" contenteditable="true">24 3/4</span></p>
            <p>Type: <span class="editable" contenteditable="true">door</span></p>
            <div class="notes">Notes: <span class="editable" contenteditable="true"></span></div>
        </div>
    </div>
    
    <script>
        // Auto-save changes to local storage
        document.querySelectorAll('.editable').forEach(el => {
            el.addEventListener('blur', () => {
                localStorage.setItem('doorList', document.body.innerHTML);
            });
        });
    </script>
</body>
</html>
```

##### Text Format (for review and manual editing)
Alternative text format if HTML is not suitable:
```text
STANDARD DOOR LIST - Job: [Job Name]
Date: [Date]
Customer: [Customer Name]
=====================================
IMPORTANT: Line numbers from PDF become Cabinet numbers in all reports

Line #1 (Cabinet #1):
  Qty: 2
  Width: 14 3/8
  Height: 24 3/4
  Type: door
  Material: White Oak
  Style: 302
  Notes: 

Line #2 (Cabinet #2):
  Qty: 2
  Width: 14 7/16
  Height: 24 3/4
  Type: door
  Material: White Oak
  Style: 302
  Notes:

Line #3 (Cabinet #3):
  Qty: 1
  Width: 14 3/4
  Height: 24 3/4
  Type: door
  Material: White Oak
  Style: 302
  Notes: No hinge boring required - island cabinet
  
  Qty: 2
  Width: 14 3/4
  Height: 5 3/8
  Type: drawer
  Material: White Oak
  Style: 302
  Notes: 

Line #4 (Cabinet #4):
  Qty: 1
  Width: 13 3/4
  Height: 24 3/4
  Type: door
  Material: White Oak
  Style: 302
  Notes: Do not bore - trash drawer (Special note from PDF for Line #4)
```

##### JSON Format (for processing)
After review and any manual corrections, convert to JSON:
```json
[
  {
    "line_item": 1,
    "cabinet": "#1",  // Line #1 from PDF = Cabinet #1 in all reports
    "items": [
      {
        "qty": 2,
        "width": "14 3/8",
        "height": "24 3/4",
        "type": "door",
        "material": "White Oak",
        "style": "302",
        "notes": ""
      }
    ]
  },
  {
    "line_item": 3,
    "cabinet": "#3",  // Can have multiple item types per cabinet
    "items": [
      {
        "qty": 1,
        "width": "14 3/4",
        "height": "24 3/4",
        "type": "door",
        "material": "White Oak",
        "style": "302",
        "notes": "No hinge boring required - island cabinet"
      },
      {
        "qty": 2,
        "width": "14 3/4",
        "height": "5 3/8",
        "type": "drawer",
        "material": "White Oak",
        "style": "302",
        "notes": ""
      }
    ]
  }
]
```

#### Key Points for Manual Extraction:
- **Line numbers become cabinet numbers**: Line #1 from PDF becomes Cabinet #1 in all reports
- **Use line item numbers** from the PDF as cabinet references throughout the system
- **Preserve exact dimensions** as shown in the order form
- **Include special notes** (e.g., "Do not bore", "trash drawer")
- **Special notes must stay with their line**: Any notes referencing specific line/cabinet numbers must be included with that item in our standard list
- **Identify type correctly**: door vs drawer vs false front
- **Maintain order sequence** from the original form

#### Why Manual Extraction is Necessary:
- PDF forms vary in layout and structure
- Complex tables and formatting are difficult to parse programmatically
- Hand-written notes or corrections need human interpretation
- Ensures accuracy for production requirements

#### File Naming Convention:
- HTML file: `[customer_name]_[job_number]_door_list.html`
- JSON file: `[customer_name]_[job_number]_door_list.json`
- Example: `paul_revere_302_door_list.html` → `paul_revere_302_door_list.json`

#### Door Pictures Integration:
- Pictures stored in: `C:\Users\nando\Projects\anyDoor\door pictures\`
- Naming convention: `[door_number] door pic.JPG` and `[door_number] door profile.JPG`
- Examples: `302 door pic.JPG`, `302 door profile.JPG`, `231 door pic.JPG`
- Pictures automatically displayed at top of HTML door list based on door style number
- Provides visual reference for production team

### Step 2: Generate Shop Report (HTML with Pictures)

#### File Structure
Create HTML file with door pictures and these sections:

##### Header Section
```html
Date: [Order Date]
Order #: [Leave blank for manual entry]
Customer Name: [Extract from order]
Address: [Extract from order]
Start Date: [Leave blank]
Finish Date: [Leave blank]
```

##### Door Specifications
```html
Door Sizes: [Opening/Finish Sizes]
Wood Species: [From order]
Outside: [Edge type]
Sticky: [Inside edge]
Panel Cut: [Specification]
Bore/Door Prep: [Yes/No checkbox]
Hinge Type: [From order]
Thickness: [Leave blank for manual entry]
```

##### Totals Calculation
- **Total Doors & Drawers**: Sum all quantities
- **Total Hinges**: Count doors × 2 hinges per door

##### Items Table (Stiles Calculation)
For stiles needed for production:
1. Calculate linear inches needed:
   - Formula: (Width + Height) × 2 × Quantity
2. Convert to 8-foot pieces:
   - Total inches ÷ 96" = Number of 8' pieces (round up)
3. Group all stiles together:
   - Show as single line item with total pieces needed
   - Width: "2 3/8" × 8'"
   - Part Name: "STILES"
   - Material: [Wood Species]
   - Notes: Total in feet

##### Pickup Section (Page 2)
```html
Name: [Customer Name]
Contact Information: [Phone / Email]
Job#: [Job Name]
Hinges: [Total calculated]
Expected Pick-Up Date: [Leave blank]
# of Doors: [Total doors + drawers]
```

### Step 3: Generate Cut List (HTML with Pictures)

#### Calculation Formulas
For each door/drawer, calculate:

**Stiles** (2 per door/drawer):
- Length = Door Height + 1/4"
- Width = 2 3/8"

**Rails** (2 per door/drawer):
- Length = Door Width - 4 1/2" + 3/4"
- Width = 2 3/8"

#### Cut List Structure

##### Header
- Date, Job Name, Customer, Wood Species
- Total Doors & Drawers count

##### Cut Table (Sorted by Length - Descending)
Columns:
1. **QTY**: Number of pieces to cut
2. **LENGTH**: Cut length in fractions (e.g., "45 1/4")
3. **WIDTH**: Always "2 3/8"
4. **PART**: "STILES" or "RAILS"
5. **LABEL**: Order line # (e.g., "#1", "#2", "#3", "#4")
6. **DESCRIPTION**: Original door/drawer specs
7. **CHECK**: Checkbox for tracking

##### Summary Section
- Total Stiles: Count and lengths
- Total Rails: Count and lengths
- Total Pieces: Sum of all cuts
- Material Required: Total 8' pieces needed

### Step 4: Convert All HTML Reports to PDF

Since all reports are now HTML-based with pictures, use Selenium WebDriver with Chrome to convert them to PDF:

#### Reports to Convert:
1. **Standard Door List HTML** → Standard Door List PDF (with pictures)
2. **Shop Report HTML** → Shop Report PDF (with pictures)
3. **Cut List HTML** → Cut List PDF (with pictures)

#### Conversion Process:

```python
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--headless')
driver = webdriver.Chrome(options=chrome_options)

# Load HTML file
driver.get(f"file:///{html_path}")

# Convert to PDF
pdf_data = driver.execute_cdp_cmd('Page.printToPDF', {
    'landscape': False,
    'displayHeaderFooter': False,
    'printBackground': True,
    'scale': 1.0,
    'paperWidth': 8.5,
    'paperHeight': 11,
    'marginTop': 0.5,
    'marginBottom': 0.5,
    'marginLeft': 0.5,
    'marginRight': 0.5
})

# Save PDF
with open(pdf_path, 'wb') as f:
    f.write(base64.b64decode(pdf_data['data']))
```

## File Outputs

### HTML Files (Editable, with pictures):
1. **[customer]_[job]_door_list.html** - Standard door list with pictures and editable fields
2. **[customer]_[job]_shop_report.html** - Shop report with door pictures
3. **[customer]_[job]_cut_list.html** - Cut list with reference pictures

### PDF Files (Final, for production):
1. **[customer]_[job]_door_list.pdf** - Standard door list PDF with pictures
2. **[customer]_[job]_shop_report.pdf** - Shop report PDF with pictures
3. **[customer]_[job]_cut_list.pdf** - Cut list PDF with pictures

## Automation Opportunities

### Future Enhancements
1. **Batch Processing**: Process multiple order forms at once
2. **Database Integration**: Store order data for tracking
3. **Barcode Generation**: Add barcodes for part tracking
4. **Material Optimization**: Calculate optimal board usage
5. **Email Integration**: Auto-send reports to production

### Python Script Template
```python
def process_order(pdf_path):
    # 1. Manual extraction - AI/Human examines PDF
    print("Please examine the PDF and extract data manually")
    
    # 2. Create text file for review
    text_list = create_text_door_list(extracted_data)
    save_text(text_list, 'order_door_list.txt')
    print("Review order_door_list.txt and make any corrections")
    
    # 3. After review, convert to JSON
    door_list = text_to_json_door_list('order_door_list.txt')
    save_json(door_list, 'order_door_list.json')
    
    # 4. Calculate production requirements
    stiles = calculate_stiles(door_list)
    rails = calculate_rails(door_list)
    hinges = calculate_hinges(door_list)
    
    # 5. Generate shop report
    shop_report_html = generate_shop_report(door_list, stiles, hinges)
    
    # 6. Generate cut list
    cut_list_html = generate_cut_list(door_list, stiles, rails)
    
    # 7. Convert to PDFs
    convert_to_pdf(shop_report_html, 'shop_report.pdf')
    convert_to_pdf(cut_list_html, 'cut_list.pdf')
    
    return 'shop_report.pdf', 'cut_list.pdf'
```

## Key Business Rules

1. **Hinges**: Always 2 hinges per door (drawers have no hinges)
2. **Stiles Formula**: Height + 1/4" (2 per door/drawer)
3. **Rails Formula**: Width - 4.5" + 0.75" (2 per door/drawer)
4. **Stile Width**: Standard 2 3/8" for all pieces
5. **Material Lengths**: Calculate in 8-foot (96") pieces
6. **Measurements**: Display as fractions, not decimals

## Validation Checklist

Before generating reports, verify:
- [ ] All line items have quantities and dimensions
- [ ] Wood species is specified
- [ ] Door/drawer type is clear for each item
- [ ] Opening vs Finish sizes is selected
- [ ] Hinge type is specified if doors are present
- [ ] Customer contact information is complete

## Notes for Implementation

### Data Extraction Challenges
- PDF text may vary in formatting
- Handle missing or incomplete fields gracefully
- Validate dimensions are numeric

### Display Formatting
- Use fractions (1/4, 1/2, 3/4) not decimals
- Keep all measurements in inches
- Show totals in feet where appropriate
- Maintain consistent 2 3/8" width for all cuts

### Production Considerations
- Group similar cuts together when possible
- Sort by length for efficient cutting
- Include line item labels for tracking
- Add checkboxes for quality control

## Testing Process

1. **Test with Sample Orders**: Use various order configurations
2. **Verify Calculations**: Double-check all math formulas
3. **Check PDF Output**: Ensure readable and properly formatted
4. **Validate Single Page**: Cut list must fit on one page
5. **Print Test**: Verify physical printout is usable in shop

---

*This documentation created: January 2025*
*For: The Raised Panel Door Factory*
*Location: 209 Riggs St, Conroe TX 77301*