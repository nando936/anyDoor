"""
Test Tesseract OCR on the zoomed 914 image to see if it reads it correctly as "9 1/4"
"""
import cv2
import pytesseract
import numpy as np

def test_tesseract_on_914():
    # Load the exact zoomed image that Google Vision sees
    image_path = "debug_914_EXACT_2nd_pass_image.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load {image_path}")
        return

    print(f"Testing Tesseract OCR on: {image_path}")
    print(f"Image shape: {image.shape}")
    print("="*60)

    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Test 1: Basic Tesseract
    print("\n[1] Basic Tesseract OCR:")
    text = pytesseract.image_to_string(gray)
    print(f"Result: '{text.strip()}'")

    # Test 2: With custom config for better number/fraction detection
    print("\n[2] Tesseract with number-focused config:")
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/ '
    text = pytesseract.image_to_string(gray, config=custom_config)
    print(f"Result: '{text.strip()}'")

    # Test 3: Get individual boxes to see how it segments
    print("\n[3] Tesseract character boxes:")
    boxes = pytesseract.image_to_boxes(gray)
    chars = []
    for b in boxes.splitlines():
        parts = b.split(' ')
        if len(parts) >= 2:
            chars.append(parts[0])
    print(f"Characters detected: {' '.join(chars)}")

    # Test 4: Try with preprocessing - threshold
    print("\n[4] With binary threshold preprocessing:")
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh)
    print(f"Result: '{text.strip()}'")

    # Test 5: Try inverting (white text on black)
    print("\n[5] With inverted image:")
    inverted = cv2.bitwise_not(gray)
    text = pytesseract.image_to_string(inverted)
    print(f"Result: '{text.strip()}'")

    # Test 6: Get word-level data to see segmentation
    print("\n[6] Word-level detection:")
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data['text'])):
        if data['text'][i].strip():
            words.append(data['text'][i])
            print(f"  Word: '{data['text'][i]}' at x={data['left'][i]}, y={data['top'][i]}")

    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Google Vision sees: '914'")
    print(f"Tesseract variations: {set([w.strip() for w in words if w.strip()])}")

if __name__ == "__main__":
    try:
        test_tesseract_on_914()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure pytesseract is installed:")
        print("pip install pytesseract")
        print("\nAnd Tesseract OCR is installed on the system")