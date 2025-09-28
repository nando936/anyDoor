"""
Test Tesseract on a tighter crop of just the "9 1/4" text
"""
import cv2
import pytesseract
import numpy as np

def test_with_different_crops():
    # First, let's use the smaller debug image that clearly shows "9 1/4"
    image_path = "debug_914_position.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load {image_path}")
        # Try the other debug image
        image_path = "debug_914_EXACT_2nd_pass_image.png"
        image = cv2.imread(image_path)
        if image is None:
            print("No debug images found")
            return

    print(f"Testing Tesseract on: {image_path}")
    print(f"Image shape: {image.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try different preprocessing methods
    print("\n[1] Direct grayscale:")
    text = pytesseract.image_to_string(gray).strip()
    print(f"Result: '{text}'")

    # Adaptive threshold (good for varying lighting)
    print("\n[2] Adaptive threshold:")
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    text = pytesseract.image_to_string(adaptive).strip()
    print(f"Result: '{text}'")

    # Otsu's threshold
    print("\n[3] Otsu threshold:")
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(otsu).strip()
    print(f"Result: '{text}'")

    # Try different PSM modes
    print("\n[4] Different PSM modes:")
    psm_modes = {
        3: "Fully automatic page segmentation",
        6: "Uniform block of text",
        7: "Single text line",
        8: "Single word",
        11: "Sparse text",
        13: "Raw line"
    }

    for mode, desc in psm_modes.items():
        try:
            config = f'--psm {mode}'
            text = pytesseract.image_to_string(gray, config=config).strip()
            if text:
                print(f"  PSM {mode} ({desc}): '{text}'")
        except:
            pass

    # Try to extract just numbers and fractions
    print("\n[5] Numbers and fractions only:")
    config = r'-c tessedit_char_whitelist=0123456789/\ '
    text = pytesseract.image_to_string(gray, config=config).strip()
    print(f"Result: '{text}'")

    # Save preprocessed images for inspection
    cv2.imwrite("debug_914_adaptive.png", adaptive)
    cv2.imwrite("debug_914_otsu.png", otsu)
    print("\nSaved preprocessed images for inspection")

    # Also try to enhance contrast
    print("\n[6] Enhanced contrast:")
    # Increase contrast
    alpha = 2.0  # Contrast control
    beta = -100  # Brightness control
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    text = pytesseract.image_to_string(enhanced).strip()
    print(f"Result: '{text}'")
    cv2.imwrite("debug_914_enhanced.png", enhanced)

if __name__ == "__main__":
    try:
        test_with_different_crops()
    except Exception as e:
        print(f"Error: {e}")