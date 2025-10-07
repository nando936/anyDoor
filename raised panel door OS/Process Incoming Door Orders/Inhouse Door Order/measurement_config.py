#!/usr/bin/env python3
"""
Configuration settings for cabinet measurement detection.
All configurable parameters in one place.
"""

# HSV color range for green text on brown backgrounds
HSV_CONFIG = {
    'lower_green': [40, 40, 40],
    'upper_green': [80, 255, 255]
}

# Text grouping proximity thresholds
GROUPING_CONFIG = {
    'x_distance': 100,   # Max horizontal distance for grouping (increased to 100)
    'y_distance': 25,   # Max vertical distance for grouping (reduced to prevent vertical over-grouping)
    'merge_threshold': 160  # Distance threshold for merging centers
}

# Zoom verification parameters
ZOOM_CONFIG = {
    'padding': 30,           # Vertical padding (top/bottom)
    'padding_horizontal': 50, # Horizontal padding (left/right) - wider to capture full characters
    'zoom_factor': 3         # Magnification factor
}

# Measurement validation
VALIDATION_CONFIG = {
    'min_value': 2,     # Minimum valid measurement value
    'max_value': 100    # Maximum valid measurement value
}

# Room name patterns to exclude
ROOM_PATTERNS = [
    r'(MASTER\s+BATH)',
    r'(MASTER\s+CLOSET)',
    r'(GUEST\s+BATH)',
    r'(POWDER\s+ROOM)',
    r'(POWDER)',
    r'(UTILITY)',
    r'(KITCHEN)',
    r'(LAUNDRY)',
    r'(PANTRY)',
    r'((?:UPSTAIRS|DOWNSTAIRS|MAIN\s+FLOOR)\s+BATH\s*\d*)',  # Location prefix + BATH
    r'((?:UPSTAIRS|DOWNSTAIRS|MAIN\s+FLOOR)\s+BEDROOM\s*\d*)',  # Location prefix + BEDROOM
    r'((?:UPSTAIRS|DOWNSTAIRS|MAIN\s+FLOOR)\s+CLOSET\s*\d*)',  # Location prefix + CLOSET
    r'((?:BATH|BEDROOM|CLOSET)\s+#\s*\d+)',  # Room with # NUMBER format (e.g., "Bath # 3")
    r'(BATH\s*\d*)',
    r'(BEDROOM\s*\d*)',
    r'(CLOSET\s*\d*)',
    r'(UPSTAIRS\s+BATH)',
    r'(DOWNSTAIRS\s+BATH)',
]

# Non-measurement text to exclude
EXCLUDE_PATTERNS = ['H2', 'NH', 'C', 'UPPERS', 'BASE', 'OR']

# Overlay notation pattern
OVERLAY_PATTERN = r'(\d+/\d+\s+OL)'

# Customer quantity notation pattern (e.g., "1 or 2", "2 or 3")
QUANTITY_NOTATION_PATTERN = r'\d+\s+or\s+\d+'
