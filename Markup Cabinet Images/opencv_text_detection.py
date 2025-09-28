#!/usr/bin/env python3
"""
Use OpenCV to find potential text areas that Vision API missed.
"""

import cv2
import numpy as np
import os
from dotenv import load_dotenv
from google.cloud import vision
from google.api_core import client_options

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

def apply_hsv_preprocessing(image):
    """Apply HSV preprocessing to isolate green text"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    return mask_inv

def find_text_contours_opencv(image):
    """Use OpenCV to find potential text regions"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours that could be text
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Filter criteria for potential text
        # - Not too small or too large
        # - Reasonable aspect ratio for text
        if (10 < w < 200 and 10 < h < 100 and
            50 < area < 5000 and
            0.2 < w/h < 10):  # Aspect ratio for text

            text_regions.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'center': (x + w//2, y + h//2),
                'area': area
            })

    return text_regions

def get_vision_api_results(image):
    """Get text detection results from Vision API"""
    api_key = os.environ.get('GOOGLE_VISION_API_KEY') or os.environ.get('GOOGLE_CLOUD_API_KEY')
    if not api_key:
        print("Error: API key not set")
        return []

    options = client_options.ClientOptions(api_key=api_key)
    client = vision.ImageAnnotatorClient(client_options=options)

    # Encode image
    _, encoded = cv2.imencode('.png', image)
    content = encoded.tobytes()

    image_vision = vision.Image(content=content)

    # Detect text
    response = client.text_detection(image=image_vision)
    texts = response.text_annotations

    vision_regions = []
    if texts:
        # Skip first annotation (full text)
        for text in texts[1:]:
            vertices = text.bounding_poly.vertices
            x_coords = [v.x for v in vertices]
            y_coords = [v.y for v in vertices]

            vision_regions.append({
                'text': text.description.strip(),
                'x': min(x_coords),
                'y': min(y_coords),
                'w': max(x_coords) - min(x_coords),
                'h': max(y_coords) - min(y_coords),
                'center': (sum(x_coords)//4, sum(y_coords)//4)
            })

    return vision_regions

def find_missed_regions(opencv_regions, vision_regions, threshold=50):
    """Find regions OpenCV detected but Vision API missed"""
    missed = []

    for ocv in opencv_regions:
        found_match = False
        ocv_center = ocv['center']

        # Check if any Vision region is close to this OpenCV region
        for vis in vision_regions:
            vis_center = vis['center']
            distance = np.sqrt((ocv_center[0] - vis_center[0])**2 +
                             (ocv_center[1] - vis_center[1])**2)

            if distance < threshold:
                found_match = True
                break

        if not found_match:
            missed.append(ocv)

    return missed

def analyze_missed_region(image, region):
    """Analyze what's in a missed region"""
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    roi = image[y:y+h, x:x+w]

    # Check if it looks like text based on pixel patterns
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi

    # Calculate some statistics
    mean_val = np.mean(gray_roi)
    std_val = np.std(gray_roi)

    # Count transitions (edges) - text has many
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)

    return {
        'mean': mean_val,
        'std': std_val,
        'edge_density': edge_density,
        'likely_text': edge_density > 0.1 and std_val > 20
    }

def main():
    image_path = "page_3.png"

    print("OPENCV TEXT DETECTION ANALYSIS")
    print("="*60)

    # Load original image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not load {image_path}")
        return

    # Apply HSV preprocessing
    hsv_processed = apply_hsv_preprocessing(original)

    # Save for reference
    cv2.imwrite("opencv_hsv_processed.png", hsv_processed)

    print("\n1. Finding text regions with OpenCV...")
    opencv_regions = find_text_contours_opencv(hsv_processed)
    print(f"   Found {len(opencv_regions)} potential text regions")

    print("\n2. Getting Vision API results...")
    vision_regions = get_vision_api_results(hsv_processed)
    print(f"   Vision API found {len(vision_regions)} text regions")

    print("\n3. Finding regions OpenCV detected but Vision API missed...")
    missed_regions = find_missed_regions(opencv_regions, vision_regions)
    print(f"   Found {len(missed_regions)} missed regions")

    # Visualize results
    visualization = cv2.cvtColor(hsv_processed, cv2.COLOR_GRAY2BGR)

    # Draw Vision API detections in GREEN
    for region in vision_regions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if 'text' in region:
            cv2.putText(visualization, region['text'], (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw missed regions in RED and analyze them
    print("\n4. Analyzing missed regions...")
    interesting_missed = []

    for i, region in enumerate(missed_regions):
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Analyze the region
        analysis = analyze_missed_region(hsv_processed, region)

        # Draw rectangle
        color = (0, 0, 255) if analysis['likely_text'] else (128, 128, 255)
        cv2.rectangle(visualization, (x, y), (x+w, y+h), color, 2)

        if analysis['likely_text']:
            interesting_missed.append(region)
            cv2.putText(visualization, f"MISSED #{i+1}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Save individual missed region for inspection
            roi = hsv_processed[y:y+h, x:x+w]
            cv2.imwrite(f"missed_region_{i+1}_at_{x}x{y}.png", roi)

            print(f"\n   Missed Region #{i+1}:")
            print(f"     Position: ({x}, {y})")
            print(f"     Size: {w}x{h}")
            print(f"     Center: {region['center']}")
            print(f"     Edge density: {analysis['edge_density']:.3f}")
            print(f"     Likely text: {analysis['likely_text']}")

    # Save visualization
    cv2.imwrite("opencv_vision_comparison.png", visualization)
    print(f"\nSaved visualization to opencv_vision_comparison.png")
    print(f"  - GREEN boxes: Vision API detected")
    print(f"  - RED boxes: OpenCV detected, Vision API missed (likely text)")
    print(f"  - LIGHT RED boxes: OpenCV detected, Vision API missed (unlikely text)")

    # Focus on the area where the left "23" should be
    print("\n5. Checking specific area for left '23' (around x=300-350, y=750-800)...")
    left_23_area = []
    for region in missed_regions:
        cx, cy = region['center']
        if 250 < cx < 400 and 700 < cy < 850:
            left_23_area.append(region)
            print(f"   Found missed region at ({cx}, {cy}) - size {region['w']}x{region['h']}")

    if not left_23_area:
        print("   No missed regions found in the expected area for left '23'")

        # Check if there are ANY OpenCV regions in that area
        print("\n   Checking all OpenCV regions in that area...")
        for region in opencv_regions:
            cx, cy = region['center']
            if 250 < cx < 400 and 700 < cy < 850:
                print(f"     OpenCV found region at ({cx}, {cy}) - size {region['w']}x{region['h']}")

if __name__ == "__main__":
    main()