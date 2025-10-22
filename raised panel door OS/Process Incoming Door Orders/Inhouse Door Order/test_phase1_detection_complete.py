#!/usr/bin/env python3
"""
Test script that replicates the EXACT Phase 1 detection process:
1. Google Vision API with same parameters
2. OpenCV supplemental detection with same filters
"""
import cv2
import base64
import requests
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the actual detection functions
from image_preprocessing import find_opencv_supplemental_regions

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GOOGLE_VISION_API_KEY')
if not api_key:
    print("[ERROR] GOOGLE_VISION_API_KEY not found in environment")
    exit(1)

# Load the preprocessed image
image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_phase1_preprocessed.png"
preprocessed_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if preprocessed_image is None:
    print(f"ERROR: Could not load image from {image_path}")
    exit(1)

print("="*80)
print("PHASE 1 DETECTION TEST - EXACT REPLICATION")
print("="*80)
print(f"Loaded preprocessed image: {preprocessed_image.shape[1]}x{preprocessed_image.shape[0]} pixels")

# ============================================================================
# STEP 1: GOOGLE VISION API (same as text_detection.py lines 166-183)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: GOOGLE VISION API DETECTION")
print("="*80)

# Encode image to base64
_, buffer = cv2.imencode('.png', preprocessed_image)
content = base64.b64encode(buffer).decode('utf-8')

# Call Google Vision API
print("Calling Google Vision API...")
url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
payload = {
    'requests': [{
        'image': {'content': content},
        'features': [{'type': 'TEXT_DETECTION'}]
    }]
}

response = requests.post(url, json=payload)

if response.status_code != 200:
    print(f"[ERROR] API request failed: {response.status_code}")
    exit(1)

data = response.json()
annotations = data.get('responses', [{}])[0].get('textAnnotations', [])

print(f"Vision API found {len(annotations)} text annotations")
print(f"Individual text items: {len(annotations) - 1}")

# Process Vision API results
vision_items = []
for i, annotation in enumerate(annotations[1:], start=1):
    text = annotation.get('description', '')

    # Check if text contains digit
    has_digit = any(c.isdigit() for c in text)
    if not has_digit:
        continue

    vertices = annotation.get('boundingPoly', {}).get('vertices', [])
    if len(vertices) < 4:
        continue

    x_coords = [v.get('x', 0) for v in vertices]
    y_coords = [v.get('y', 0) for v in vertices]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    vision_items.append({
        'text': text,
        'center': (int(center_x), int(center_y)),
        'source': 'Vision API'
    })

print(f"\nVision API items with numbers: {len(vision_items)}")

# Check for "9"
found_9_vision = any(item['text'].strip() == '9' for item in vision_items)
print(f"Vision API found '9': {found_9_vision}")

# ============================================================================
# STEP 2: OPENCV SUPPLEMENTAL DETECTION (same as text_detection.py line 306)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: OPENCV SUPPLEMENTAL DETECTION")
print("="*80)

# Collect Vision API detection regions for exclusion (use x_min/x_max format expected by function)
vision_regions = []
for annotation in annotations[1:]:
    vertices = annotation.get('boundingPoly', {}).get('vertices', [])
    if len(vertices) >= 4:
        x_coords = [v.get('x', 0) for v in vertices]
        y_coords = [v.get('y', 0) for v in vertices]
        vision_regions.append({
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords)
        })

print(f"Excluding {len(vision_regions)} Vision API regions from OpenCV search...")

# Call the actual OpenCV supplemental detection function
opencv_regions = find_opencv_supplemental_regions(
    preprocessed_image,
    vision_regions
)

print(f"OpenCV found {len(opencv_regions)} additional regions")

opencv_items = []
for i, region in enumerate(opencv_regions):
    # OpenCV regions have 'x', 'y', 'center' keys
    opencv_items.append({
        'text': region.get('text', f"[Region {i+1}]"),
        'center': region['center'],
        'source': 'OpenCV'
    })

# ========================================================================
# SUPPLEMENTAL PASS: Detect small single digits (e.g., "9")
# ========================================================================
print("\n" + "="*80)
print("STEP 2B: SUPPLEMENTAL DIGIT DETECTION (TEST ONLY)")
print("="*80)
print("Looking for small digits with lenient parameters...")

# Threshold image for contour detection
_, binary = cv2.threshold(preprocessed_image, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} total contours")

# Debug counters
filter_counts = {
    'size': 0,
    'area': 0,
    'overlap': 0,
    'shape': 0,
    'passed': 0
}

digit_items = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    center_x = x + w // 2
    center_y = y + h // 2

    # More lenient size constraints for small digits
    if not (5 < w < 30 and 8 < h < 40):
        filter_counts['size'] += 1
        continue

    # More lenient area threshold for thin strokes
    if area < 25:
        filter_counts['area'] += 1
        continue

    # Check Vision API overlap to avoid duplicates
    is_covered = False
    for item in vision_regions:
        zoom_padding_h = 0  # NO PADDING - test raw Vision API bounds only
        zoom_padding_v = 0  # NO PADDING - test raw Vision API bounds only
        vision_left = item['x_min'] - zoom_padding_h
        vision_right = item['x_max'] + zoom_padding_h
        vision_top = item['y_min'] - zoom_padding_v
        vision_bottom = item['y_max'] + zoom_padding_v

        if not (x + w < vision_left or x > vision_right or
                y + h < vision_top or y > vision_bottom):
            is_covered = True
            break

    if is_covered:
        filter_counts['overlap'] += 1
        continue

    # ===== SHAPE VERIFICATION: Only accept "9" shapes =====
    # Extract ROI for "9" shape analysis
    roi = binary[y:y+h, x:x+w]

    # 1. Aspect ratio: "9" should be taller than wide (typical 0.4-0.9)
    aspect_ratio = w / h
    if not (0.4 < aspect_ratio < 0.9):
        filter_counts['shape'] += 1
        continue

    # 2. "9" has a circular/oval hole at top (detect inner contour)
    roi_contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    has_hole = hierarchy is not None and len(hierarchy[0]) > 1
    if not has_hole:
        filter_counts['shape'] += 1
        continue

    # 3. Vertical center of mass - "9" is top-heavy
    moments = cv2.moments(roi)
    if moments['m00'] > 0:
        cy = moments['m01'] / moments['m00']
        top_heavy = cy < (h * 0.55)  # Center of mass in top 55%
        if not top_heavy:
            filter_counts['shape'] += 1
            continue

    # Found a "9" shape!
    filter_counts['passed'] += 1
    digit_items.append({
        'text': '9 (shape verified)',
        'center': (int(center_x), int(center_y)),
        'bounds': {'left': x, 'right': x+w, 'top': y, 'bottom': y+h},
        'size': (w, h),
        'area': int(area),
        'source': 'OpenCV Digit'
    })
    print(f"  Found '9' shape at ({x}, {y}), size {w}x{h}px, area {area}px², aspect={aspect_ratio:.2f}")

print(f"\nFilter statistics:")
print(f"  Filtered by size: {filter_counts['size']}")
print(f"  Filtered by area: {filter_counts['area']}")
print(f"  Filtered by overlap: {filter_counts['overlap']}")
print(f"  Filtered by shape: {filter_counts['shape']}")
print(f"  Passed all filters: {filter_counts['passed']}")

print(f"\nDigit detection found {len(digit_items)} '9' shapes on entire page")

# ============================================================================
# STEP 3: COMBINED RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: COMBINED DETECTION RESULTS")
print("="*80)

all_items = vision_items + opencv_items + digit_items
print(f"Total items detected: {len(all_items)}")
print(f"  - From Vision API: {len(vision_items)}")
print(f"  - From OpenCV: {len(opencv_items)}")
print(f"  - From Digit Detection: {len(digit_items)}")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

if found_9_vision:
    print("✓ THE '9' WAS FOUND BY GOOGLE VISION API")
else:
    print("✗ THE '9' WAS NOT FOUND BY GOOGLE VISION API")

if len(digit_items) > 0:
    print(f"✓ DIGIT DETECTION FOUND {len(digit_items)} '9' SHAPE(S) ON PAGE")
else:
    print("✗ DIGIT DETECTION DID NOT FIND ANY '9' SHAPES")

print("\n" + "-"*80)
print("ALL DETECTED ITEMS (sorted by Y position):")
print("-"*80)
for item in sorted(all_items, key=lambda x: x['center'][1]):
    source_marker = f"[{item['source']}]"
    print(f"{source_marker:15s} '{item['text']:15s}' at {item['center']}")

if len(digit_items) > 0:
    print("\n" + "="*80)
    print("VERIFIED '9' DETECTIONS:")
    print("="*80)
    for item in digit_items:
        print(f"Position: {item['center']}")
        if 'bounds' in item:
            print(f"Bounds: left={item['bounds']['left']:.0f}, right={item['bounds']['right']:.0f}, "
                  f"top={item['bounds']['top']:.0f}, bottom={item['bounds']['bottom']:.0f}")
        if 'size' in item:
            print(f"Size: {item['size'][0]}x{item['size'][1]}px, area={item['area']}px²")
        print()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if len(digit_items) > 0:
    print(f"✓ Detection successful - Found {len(digit_items)} '9' shape(s) on entire page")
    print("\n  The supplemental digit detection route successfully finds '9's!")
    print("  This approach could be added to the production code.")
else:
    print("✗ Detection failed - No '9' shapes found on entire page")
    print("\nEither no '9's exist or they were filtered out by detection checks.")
print("="*80)

# ============================================================================
# VISUALIZATION: Show excluded zones
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATION OF EXCLUDED ZONES")
print("="*80)

# Load color version of preprocessed image
vis_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

# Draw Vision API exclusion zones (NO padding)
for item in vision_regions:
    zoom_padding_h = 0  # NO PADDING - show raw Vision API bounds only
    zoom_padding_v = 0  # NO PADDING - show raw Vision API bounds only
    vision_left = int(item['x_min'] - zoom_padding_h)
    vision_right = int(item['x_max'] + zoom_padding_h)
    vision_top = int(item['y_min'] - zoom_padding_v)
    vision_bottom = int(item['y_max'] + zoom_padding_v)

    # Draw exclusion zone in red
    cv2.rectangle(vis_image, (vision_left, vision_top), (vision_right, vision_bottom), (0, 0, 255), 2)

# Mark verified "9" detections with CYAN marker
for digit_item in digit_items:
    bounds = digit_item['bounds']
    center_x = digit_item['center'][0]
    center_y = digit_item['center'][1]
    x = bounds['left']
    y = bounds['top']
    w = bounds['right'] - bounds['left']
    h = bounds['bottom'] - bounds['top']

    # Draw cyan outline circle around detected "9" (not filled)
    radius = 30  # Bigger radius to circle around the "9"
    cv2.circle(vis_image, (center_x, center_y), radius, (255, 255, 0), 3)  # Outline only (thickness=3)
    # Add label
    cv2.putText(vis_image, "FOUND '9'", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# Save visualization
output_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-10-15(12-09)/all_pages/page-5_opencv_excluded_zones.png"
cv2.imwrite(output_path, vis_image)
print(f"[SAVED] Excluded zones visualization: {output_path}")
print("  Red boxes = Vision API detection bounds (NO padding)")
print("  Cyan circle = Verified '9' detection")
print("="*80)
