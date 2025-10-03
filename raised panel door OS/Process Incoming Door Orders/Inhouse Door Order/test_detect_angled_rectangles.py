import cv2
import numpy as np
import sys

def detect_angled_rectangles(image_path):
    """Detect rectangles in images even when rotated/angled"""

    # Read image
    image = cv2.imread(image_path.replace('\\', '/'))
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Create copy for drawing
    output = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} contours")

    rectangles = []

    for i, contour in enumerate(contours):
        # Get minimum area rectangle (works for rotated rectangles)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Get rectangle properties
        center, (width, height), angle = rect

        # Filter by size (adjust these values based on your needs)
        min_size = 30  # minimum dimension
        max_size = image.shape[0] * 0.9  # not larger than 90% of image

        if width > min_size and height > min_size and width < max_size and height < max_size:
            # Calculate aspect ratio
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

            # Only keep rectangles that look reasonable (not too thin)
            if aspect_ratio < 10:
                rectangles.append({
                    'center': center,
                    'size': (width, height),
                    'angle': angle,
                    'box': box,
                    'area': width * height
                })

                # Draw the rectangle
                cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

                # Draw center point
                cv2.circle(output, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

                # Add label
                label = f"{int(width)}x{int(height)}"
                cv2.putText(output, label, (int(center[0])-30, int(center[1])-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    print(f"Detected {len(rectangles)} rectangles")

    # Sort by area (largest first)
    rectangles.sort(key=lambda x: x['area'], reverse=True)

    # Print details
    for i, rect in enumerate(rectangles[:10]):  # Show top 10
        print(f"Rectangle {i+1}:")
        print(f"  Center: ({rect['center'][0]:.1f}, {rect['center'][1]:.1f})")
        print(f"  Size: {rect['size'][0]:.1f} x {rect['size'][1]:.1f}")
        print(f"  Angle: {rect['angle']:.1f} degrees")
        print(f"  Area: {rect['area']:.1f}")

    # Save output
    output_path = image_path.replace('\\', '/').replace('.png', '_rectangles_detected.png')
    cv2.imwrite(output_path, output)
    print(f"\n[OK] Output saved to: {output_path}")

    # Also save the edges for debugging
    edges_path = output_path.replace('_rectangles_detected.png', '_edges.png')
    cv2.imwrite(edges_path, edges)
    print(f"[OK] Edges saved to: {edges_path}")

    return rectangles

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_detect_angled_rectangles.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_angled_rectangles(image_path)
