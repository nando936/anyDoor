#!/usr/bin/env python3
"""
Visualization Toggle Configuration for Cabinet Measurement Detection

This module contains all toggle constants that control which visualization layers
are displayed in the annotated output images. These toggles allow you to enable/disable
specific visualization groups for debugging and presentation purposes.

All toggles are boolean values (True/False) that can be modified to show or hide
different visualization layers.

Usage:
    In visualization.py or other modules:
        from visualization_toggles import *

    Or explicitly:
        from visualization_toggles import SHOW_CENTER_ZONES, SHOW_DETECTED_LINES
"""

# ============================================================================
# VISUALIZATION TOGGLES - Control which groups of visualizations to show
# ============================================================================

# BOTTOM WIDTH GROUP: Controls all bottom width-related visualizations
#   - Thick red extent lines with tick marks
#   - Arrow end circles and labels ("ARROW END")
#   - Magenta/orange arrow search zone boxes (left/right extent search)
#   - Cyan scan ROIs (drawer detection area above extent line)
#   - Down arrow circles and labels (A1, A2, A3, etc.)
#   - Green bottom width reference line connecting arrows
SHOW_BOTTOM_WIDTH_GROUP = False  # Set to True to show all bottom width visualizations

# NON-BOTTOM WIDTH EXTENT LINES: Show projected h-line extents for regular widths
#   - Shows the horizontal extent line used during pairing algorithm
#   - Each width projects its h-line at the detected angle across the image
#   - Heights must be "above" this projected line to qualify for pairing
#   - Different from SHOW_BOTTOM_WIDTH_GROUP (which is for drawer detection)
#   - Helps debug why certain widths and heights don't pair
#   - Line style: dashed blue line spanning the full image width
SHOW_NON_BOTTOM_WIDTH_EXTENT_LINES = True  # Set to True to show extent lines for non-bottom widths

# DEBUG FILTER: Show extent lines for specific measurements only (non-bottom widths)
#   - Set to None to show all non-bottom width extent lines
#   - Set to list of measurement numbers to show only those (e.g., [8] for M8 only)
#   - Set to [3, 8, 14] to show M3, M8, and M14 extent lines
#   - Only affects non-bottom width extent lines (blue dashed lines)
#   - Does NOT affect bottom width visualizations (SHOW_BOTTOM_WIDTH_GROUP)
DEBUG_EXTENT_LINES_FOR_MEASUREMENTS = [8]  # None = show all, or [8] = M8 only, [3,8,14] = multiple

# CENTER ZONES: Show X-tolerance zones used during pairing
#   - Light orange vertical bands for width center zones (±text_width/2)
#   - Light green vertical bands for unpaired height center zones (±50px)
#   - Dashed boundary lines showing zone edges
#   - Center line showing measurement X position
#   - Labels showing tolerance values
SHOW_CENTER_ZONES = False  # Set to True to show center zone visualizations

# DEBUG TEXT BOXES: Show OCR text with boxes below actual measurements
SHOW_DEBUG_TEXT_BOXES = False  # Set to True to show debug text boxes and OCR text

# DETECTED LINES: Show actual h-lines and v-lines detected by HoughLinesP
#   - Yellow lines for detected h-lines (left and right ROIs)
#   - Cyan lines for detected v-lines (top and bottom ROIs)
SHOW_DETECTED_LINES = True  # Set to True to show detected lines from HoughLinesP

# OPENING DIMENSIONS: Show dimension text below opening markers
#   - Dimension text like "19 7/16 W x 7 3/4 H"
#   - Notation text like "NO HINGES"
#   - Finished size notes
SHOW_OPENING_DIMENSIONS = False  # Set to True to show dimensions below opening markers

# ============================================================================
