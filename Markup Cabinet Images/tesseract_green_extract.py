"""
Extract green channel and process it for better Tesseract OCR
since the measurements are in green text
"""
import cv2
import pytesseract
import numpy as np

def extract_green_text(image_path, x, y, padding=50):
    """Extract green text from image for better OCR"""
    # Load the original page 3 image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load {image_path}")
        return

    h, w = image.shape[:2]

    # Extract region
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(w, int(x + padding))
    y2 = min(h, int(y + padding))

    cropped = image[y1:y2, x1:x2]

    # Split channels (B, G, R)
    b, g, r = cv2.split(cropped)

    # Since text is green, the green channel should have high values
    # where text is, and lower values in background
    # Invert it so text becomes dark on light background
    green_inverted = cv2.bitwise_not(g)

    # Apply threshold to get clean black text on white
    _, thresh = cv2.threshold(green_inverted, 200, 255, cv2.THRESH_BINARY)

    # Save for inspection
    cv2.imwrite("green_channel_inverted.png", green_inverted)
    cv2.imwrite("green_thresh.png", thresh)

    print(f"Testing Tesseract on green channel extraction:")
    print("="*50)

    # Test different configs on the processed image
    configs = [
        ("Default", ""),
        ("PSM 6", "--psm 6"),
        ("PSM 7 (single line)", "--psm 7"),
        ("PSM 8 (single word)", "--psm 8"),
        ("PSM 11 (sparse)", "--psm 11"),
        ("PSM 13 (raw line)", "--psm 13"),
        ("Whitelist nums", r"-c tessedit_char_whitelist=0123456789/\ "),
    ]

    for name, config in configs:
        try:
            result = pytesseract.image_to_string(thresh, config=config).strip()
            if result:
                print(f"{name:20} : '{result}'")
        except Exception as e:
            if "Error" not in str(e):
                print(f"{name:20} : {e}")

    # Also try with morphological operations to clean up
    print("\n\nWith morphological cleaning:")
    print("="*50)

    # Remove noise with morphological opening
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("green_cleaned.png", cleaned)

    result = pytesseract.image_to_string(cleaned, config="--psm 7").strip()
    if result:
        print(f"Cleaned + PSM 7     : '{result}'")

    # Try dilation to make text thicker
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imwrite("green_dilated.png", dilated)

    result = pytesseract.image_to_string(dilated, config="--psm 7").strip()
    if result:
        print(f"Dilated + PSM 7     : '{result}'")

    # Try HSV color extraction
    print("\n\nUsing HSV color space:")
    print("="*50)

    # Convert to HSV
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Green color range in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create mask for green
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert so text is black
    mask_inv = cv2.bitwise_not(mask)
    cv2.imwrite("hsv_mask.png", mask_inv)

    result = pytesseract.image_to_string(mask_inv, config="--psm 7").strip()
    if result:
        print(f"HSV mask + PSM 7    : '{result}'")

def main():
    print("TESSERACT WITH GREEN CHANNEL EXTRACTION")
    print("="*60)

    image_path = "//vmware-host/Shared Folders/suarez group qb/customers/raised panel/Measures-2025-09-08(17-08)/all_pages/page_3.png"

    print("\n[1] Testing '914' position (701, 1264):")
    extract_green_text(image_path, 701, 1264, padding=40)

    print("\n\n[2] Testing working '9 1/4' position (715, 1413) for comparison:")
    extract_green_text(image_path, 715, 1413, padding=40)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")