# Journal Entry - Door Configuration System Architecture

**Date**: 2025-09-07
**Topic**: Modular JSON-based door configuration system

## What We Built
Created a comprehensive modular configuration system for door manufacturing that separates concerns into distinct, reusable components. This allows mixing and matching different door styles, construction methods, materials, and hardware.

## System Architecture

### 1. Construction Methods (`door_configuration/construction_methods/`)
Defines how doors are assembled:
- **cope_and_stick.json**: Traditional joinery with coped ends and profiled edges
  - Requires oversize stiles/rails for cope depth
  - Sticking depth of 0.5"
  - Specific tooling requirements
- **mitre_cut.json**: 45-degree mitered corners
  - No oversize needed
  - Simpler tooling
  - Modern appearance

### 2. Material Schedules (`door_configuration/material_schedules/`)
Standardizes material specifications:
- **paint_grade_standard.json**: MDF-based materials for painting
  - Stiles/rails: 3/4" MDF
  - Panels: 1/2" MDF raised or 1/4" flat
  - Primer requirements included
- **stain_grade_premium.json**: Solid wood for staining
  - Species options (Oak, Cherry, Maple, Walnut)
  - Grain matching requirements
  - Finish specifications

### 3. Hardware Schedules (`door_configuration/hardware_schedules/`)
Defines hinges and mounting specifications:
- **blum_soft_close_full_overlay.json**: 
  - Model numbers and specifications
  - Boring patterns (35mm cups)
  - Installation requirements
- **blum_soft_close_half_overlay.json**:
  - Different overlay calculations
  - Adjusted boring positions

### 4. Profile Library (`door_configuration/profile_library/`)
Reusable edge and panel profiles:

**Edges:**
- **square.json**: Simple square edge
- **quarter_round.json**: Classic rounded profile
- **ogee.json**: Decorative S-curve profile
- **chamfer.json**: 45-degree beveled edge

**Panels:**
- **raised_standard.json**: Traditional raised panel
  - 3/8" raise height
  - 15-degree bevel angle
  - Tooling specifications
- **flat_recessed.json**: Modern flat panel
  - 1/4" recess depth
  - No special tooling required

### 5. Door Styles (`door_configuration/door_styles/`)
Combines all elements into specific door products:
- **103.json**: 5-piece mitre door with ogee edge
  - References: mitre_cut method, paint_grade material, quarter_round edge
  - Specific dimensions and calculations
- **231.json**: Cope & stick traditional door
  - References: cope_and_stick method, square edge, raised panel
  - Oversize specifications included

## Key Design Decisions

### Modular Architecture
- Each component is independent and reusable
- Door styles reference components by ID
- Changes to one component automatically update all doors using it
- Easy to add new options without breaking existing configurations

### JSON Format
- Human-readable and editable
- Version control friendly
- Easy to validate and parse
- Can be edited without coding knowledge

### Calculation Formulas
Each construction method includes formulas for:
```javascript
stile_length = door_height + oversize
rail_length = door_width - (2 * stile_width) + (2 * cope_depth)
panel_width = door_width - (2 * stile_width) + tongue_allowance
panel_height = door_height - (2 * rail_width) + tongue_allowance
```

## Implementation in Code

Created `door_configuration.py` that:
1. Loads JSON configurations dynamically
2. Resolves references between components
3. Provides calculation methods based on construction type
4. Falls back to defaults when specs unavailable
5. Integrates with existing processing pipeline

## Benefits Achieved

1. **Flexibility**: Easy to create new door styles by combining existing components
2. **Consistency**: Standardized specifications across all doors
3. **Maintainability**: Update one profile, affects all doors using it
4. **Scalability**: Add new construction methods or materials without code changes
5. **Documentation**: JSON files serve as self-documenting specifications

## Integration Points

The configuration system integrates with:
- **Shop Report**: Uses construction method for oversize calculations
- **Cut List**: Applies material schedules for grouping
- **Panel Optimizer**: Considers panel specifications from profiles
- **Door Specs**: Overrides with existing .txt specs when available

## Next Steps

1. **Expand Profile Library**
   - Add more edge profiles (bead, cove, etc.)
   - Include glass panel options
   - Add frame-only configurations

2. **Enhanced Calculations**
   - Automatic hinge placement based on door height
   - Drawer box calculations from fronts
   - Pull/knob placement formulas

3. **Validation System**
   - Verify component references exist
   - Check calculation constraints
   - Warn about incompatible combinations

4. **User Interface**
   - Web-based configuration builder
   - Visual preview of combinations
   - Export to production files

## Challenges Overcome

- **Backward Compatibility**: System works alongside existing door specs
- **Reference Resolution**: Handles nested references between components
- **Default Handling**: Gracefully falls back when configurations missing
- **Formula Standardization**: Unified calculation methods across construction types

## Lesson Learned

Separating configuration from code makes the system infinitely more flexible. Shop workers can modify JSON files to add new door styles without programmer involvement. The modular approach means we can build complex doors from simple, reusable components.

## Time Spent

Approximately 2 hours - design, implementation, and testing