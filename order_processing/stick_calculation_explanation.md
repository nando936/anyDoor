# Stick Quantity Calculation Explanation

## How Stick Quantities are Calculated in the Shop Report

The stick quantity calculation happens in the `generate_shop_report_html()` function in `1_process_new_order.py`.

### Calculation Process:

1. **Linear Inches Calculation** (lines 356-375):
   ```python
   # For each door item:
   width = fraction_to_decimal(item['width'])
   height = fraction_to_decimal(item['height'])
   qty = item['qty']
   
   if is_cope_and_stick:
       # Cope and stick: stiles need extra length and width for trimming
       stile_length = height + stile_length_oversize  # Add 1/4" for trimming
       stile_cut_width = stile_width + stile_width_oversize  # Add 1/8" width
       rail_length = width
       materials_by_species[species]['linear_inches'] += qty * 2 * (stile_length + rail_length)
   else:
       # Mitre cut doors: exact dimensions
       materials_by_species[species]['linear_inches'] += qty * 2 * (width + height)
   ```

   - Each door needs 2 stiles (vertical pieces) and 2 rails (horizontal pieces)
   - **For cope & stick doors (e.g., style 103):**
     - Stiles are cut 1/4" longer for trimming
     - Stiles are cut 1/8" wider for trimming
     - Rails use door width
     - Total linear inches = qty × 2 × (stile_length + rail_length)
   - **For mitre cut doors (e.g., style 231):**
     - 2 stiles = 2 × height (exact)
     - 2 rails = 2 × width (exact)
     - Total linear inches = qty × 2 × (width + height)

2. **Converting to 8-foot Stick Pieces** (lines 449-459):
   ```python
   # Calculate 8-foot pieces needed
   eight_foot_pieces = int((counts['linear_inches'] / 96) + 0.999)  # Round up
   ```
   
   - Each stick is 8 feet long = 96 inches
   - Divide total linear inches by 96
   - Add 0.999 and convert to int to effectively round up
   - This ensures we always have enough material

### Example Calculation:

For Benjamin Franklin Order #103:
- Paint Grade doors: 11 doors of various sizes
- If total linear inches = 850 inches
- 850 ÷ 96 = 8.85 pieces
- Round up = 9 pieces of 8-foot sticks

### Key Points:

1. **Material Grouping**: Sticks are calculated separately for each material type (Paint Grade, Stain Grade Maple, etc.)

2. **Waste Factor**: The calculation rounds up to ensure sufficient material, accounting for cuts and potential waste

3. **Stick Dimensions**: Default is 3" wide × 13/16" thick × 8' long (can vary by door style specifications)

4. **No Bore/No Hinge Items**: These still require stick material for the frame, only the hinge count is affected

5. **Mitre Cut vs Cope & Stick**: For mitre cut doors, the dimensions are exact. For cope & stick, additional calculations may be needed for the joint overlap.

### Display in Shop Report:
The calculated stick quantities appear in the "Materials Required" table as:
```
Stile Sticks | [quantity] pcs | 3" x 13/16" x 8' | [Material Type]
```