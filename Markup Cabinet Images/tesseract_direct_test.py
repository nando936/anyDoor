"""
Test Tesseract directly on the original image area where "9 1/4" appears
"""
import cv2
import pytesseract
import numpy as np

def extract_and_ocr_region(image_path, x, y, padding=50):
    """Extract a region and OCR it with Tesseract"""
    # Load the original page 3 image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load {image_path}")
        return

    h, w = image.shape[:2]

    # Extract region around the position
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(w, int(x + padding))
    y2 = min(h, int(y + padding))

    # Crop
    cropped = image[y1:y2, x1:x2]

    # Save the crop for inspection
    cv2.imwrite("tesseract_crop_914.png", cropped)
    print(f"Saved crop: tesseract_crop_914.png (size: {cropped.shape})")

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Try multiple Tesseract configurations
    configs = [
        ("Default", ""),
        ("PSM 6", "--psm 6"),
        ("PSM 7", "--psm 7"),
        ("PSM 8", "--psm 8"),
        ("PSM 11", "--psm 11"),
        ("Legacy OEM", "--oem 0"),
        ("LSTM only", "--oem 1"),
        ("Combined", "--oem 2"),
        ("Default + whitelist", "-c tessedit_char_whitelist='0123456789 /'"),
    ]

    print(f"\nTesseract results for position ({x}, {y}):")
    print("="*50)

    for name, config in configs:
        try:
            result = pytesseract.image_to_string(gray, config=config).strip()
            if result:
                print(f"{name:20} : '{result}'")
        except Exception as e:
            print(f"{name:20} : Error - {str(e)[:30]}")

    # Also try with different preprocessing
    print("\n\nWith preprocessing:")
    print("="*50)

    # Binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    result = pytesseract.image_to_string(binary).strip()
    if result:
        print(f"Binary threshold     : '{result}'")

    # Inverted
    inverted = cv2.bitwise_not(gray)
    result = pytesseract.image_to_string(inverted).strip()
    if result:
        print(f"Inverted            : '{result}'")

    # Blur then threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, blurred_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = pytesseract.image_to_string(blurred_thresh).strip()
    if result:
        print(f"Blur + Otsu         : '{result}'")

    # Save preprocessed versions
    cv2.imwrite("tesseract_binary.png", binary)
    cv2.imwrite("tesseract_inverted.png", inverted)
    cv2.imwrite("tesseract_blurred_otsu.png", blurred_thresh)

def main():
    print("TESSERACT DIRECT TEST ON ORIGINAL IMAGE")
    print("="*60)

    # Original page 3 image
    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3.png"

    # Test the "914" position
    print("\n[1] Testing position (701, 1264) where '914' is detected:")
    extract_and_ocr_region(image_path, 701, 1264, padding=50)

    # Test with different padding
    print("\n\n[2] Testing with larger region (padding=100):")
    extract_and_ocr_region(image_path, 701, 1264, padding=100)

    # For comparison, test the working "9 1/4"
    print("\n\n[3] Testing the working '9 1/4' at (715, 1413) for comparison:")
    extract_and_ocr_region(image_path, 715, 1413, padding=50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Tesseract is installed:")