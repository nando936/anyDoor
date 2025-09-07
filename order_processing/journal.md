# Journal Entry - Door Order Processing System

**Date**: 2025-09-07
**Topic**: Building standardized door order processing system

## What We Tried
- Built system to convert various user-submitted order formats to standardized finish door list
- Implemented opening to finish size conversion (add 2 × overlay)
- Created critical verification loop to prevent expensive mistakes
- Added door style specifications support (231 specs.txt)
- Organized header into 3 columns: Customer Info, Hardware, Door Specs

## Key Problems Solved
- **Size conversion errors**: Original data was wrong (14 3/8 not 13 7/8) - fixed extraction
- **Missing details**: Added bore/door prep, full hinge name, all specs from original
- **Unicode issues**: Windows encoding errors - removed all Unicode symbols
- **Verification**: Built redundant verification that re-reads original from scratch
- **Pictures not showing**: Fixed with file:/// protocol paths

## What Worked
- Folder structure: need to process → output/[customer] → processed
- Line numbers = Cabinet numbers convention
- Critical verification catches all errors before production
- 3-column header layout with sections
- HTML-first approach, PDF generation optional

## Current Status
- Finish door list complete with all details
- Shop report generated (HTML only, no PDF)
- Verification passing with zero errors
- System ready for production use

## File Locations
- Main script: `1_process_new_order.py`
- Verification: `critical_verification.py`
- Test order: `archive/door_231_order_form.html`
- Output: `output/paul_revere_231/`
- Door pics: `order_processing/door pictures/`

## To Restart
1. Run: `cd order_processing && python process_paul_revere_order.py`
2. Verify: `python critical_verification.py`
3. Check output in `output/paul_revere_231/`

## Lesson Learned
Always perform redundant verification from scratch - don't trust previous processing. Every detail from original must appear in standardized output.

## Time Spent
~3 hours

---

# Journal Entry - Shop Report & Cut List Enhancements

**Date**: 2025-09-07  
**Topic**: Enhanced shop report with specs, added cut list with mitre cuts

## What We Tried
- Updated shop report to use door specs (3" stiles, 13/16" thickness)
- Separated materials by wood species (Paint Grade vs White Oak)
- Made hinges separate line item with full type description
- Added door/profile/hinge pictures to both door list and shop report
- Created cut list using mitre cut calculations from specs

## Key Problems  
- Shop report was using hardcoded 2 3/8" instead of specs dimensions
- Materials weren't separated by species for ordering
- Cut list needed mitre cut logic (exact dimensions, not cope & stick)

## What Worked
- Parsing specs.txt for dimensions and cut type
- Mitre cut: stiles = door height, rails = door width (exact)
- Material tracking by species throughout all reports
- Pictures display properly with file:/// protocol

## Current Status Complete
- **Door List**: Shows all items with hinge/door/profile pictures
- **Shop Report**: Materials by species, proper dimensions from specs, pictures
- **Cut List**: Mitre cut calculations, sorted by length, totals by material
- All 3 reports working with Paul Revere test order

## Generated Files
- `output/paul_revere_231/finish_door_list.html` - Complete with pictures
- `output/paul_revere_231/shop_report.html` - Stile sticks 3" x 13/16" x 8'
- `output/paul_revere_231/cut_list.html` - 60 pieces total (36 Paint, 24 Oak)

## To Generate Reports
```bash
cd order_processing
python process_paul_revere_order.py  # Door list only
python generate_shop_report.py       # Shop report only  
python generate_cut_list.py          # Cut list only
```

## Lesson Learned
Door specs drive everything - dimensions, cut type, calculations. Always read specs first.

## Time Spent
~1 hour