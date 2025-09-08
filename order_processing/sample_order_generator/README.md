# Sample Order Generator

This folder contains scripts and templates for creating sample door orders for testing the order processing system.

## Purpose
- Generate realistic test orders without real customer data
- Test different door styles and specifications
- Validate the processing system with various order configurations
- Create reproducible test cases

## Files in This Folder

### Main Script
- `generate_sample_order.py` - Primary script for creating sample orders

### Templates
- `extraction_template.json` - Basic template structure for orders
- `extraction_template_paul_revere.json` - Sample order for Paul Revere (style 231)
- Other extraction templates for specific test cases

### Legacy Scripts
- `create_sample_order.py` - Original sample order creation script
- `create_door_231_order.py` - Creates orders for door style 231
- `create_door_231_order_v2.py` - Updated version for style 231

## How to Generate Sample Orders

### Quick Start
```bash
# Generate a basic sample order
python generate_sample_order.py

# Generate order for specific door style
python generate_sample_order.py 103 "John Smith"

# Generate order for style 231
python generate_sample_order.py 231 "Jane Doe"
```

### Processing the Sample Order
1. Generate the sample order:
   ```bash
   python generate_sample_order.py 103 "Test Customer"
   ```

2. Process it:
   ```bash
   cd ..
   python process_extracted_order.py sample_order_generator/sample_test_customer_XXX.json
   ```

## Sample Order Specifications

### Door Style 103 (Cope & Stick)
- Stile width: 2 3/8" + 1/8" oversize
- Rail width: 2 1/4"
- Requires 1/4" longer stiles for trimming
- Panel: 1/4" MDF Raised Panel
- Typical for traditional kitchen cabinets

### Door Style 231 (Mitre Cut)
- Stile width: 3" (exact, no oversize)
- Rail width: 3"
- No oversize required
- Panel: 3/8" Plywood (Flat Panel ONLY)
- Modern, clean-line design

## Important Notes

### For Testing Purposes Only
- These are NOT real customer orders
- Use realistic but fictional data
- Test various edge cases:
  * Mixed wood species
  * Different door sizes
  * Special notes (no bore, trash drawer, etc.)
  * Opening vs finish sizes

### Common Test Scenarios
1. **Basic Order**: 8-10 standard doors, single wood species
2. **Mixed Species**: Paint grade and stain grade in same order
3. **Complex Order**: 15+ items with drawers and special notes
4. **Horizontal Doors**: Include wide/short doors for testing
5. **Trash Drawer**: Test "no bore" notes handling

### Data Validation
Sample orders should test:
- Size conversion (opening to finish)
- Board feet calculations
- Panel sheet calculations
- Note preservation
- Cabinet numbering consistency

## Creating Custom Test Cases

To create specific test scenarios, modify the `generate_sample_order()` function:

```python
# Example: Test order with all Paint Grade
customer_info["wood_species"] = "Paint Grade"

# Example: Test with specific cabinet count
num_cabinets = 15  # Instead of random

# Example: Add specific notes
notes = "Custom testing note"
```

## Files Generated

Each sample order creates:
- `sample_[customer]_[job#].json` - Order data in JSON format

When processed, it creates in `../output/[customer]_[job#]/`:
- `finish_door_list.html` - Editable door list
- `shop_report.html` - Production report with materials
- `cut_list.html` - Cut list with dimensions
- All corresponding PDFs
- `order_data.json` - Processed data

## Troubleshooting

### Common Issues
1. **Import errors**: Make sure you're running from the `sample_order_generator` folder
2. **Path issues**: The script expects to be one folder below the main processing scripts
3. **Missing specs**: Ensure door spec files exist in `../door pictures/`

### Testing Workflow
1. Generate sample order
2. Process it
3. Run verification: `python ../critical_verification.py`
4. Check outputs in `../output/` folder
5. Original moves to `../processed/` automatically

---

For questions about the order processing system, see the main README in the parent directory.