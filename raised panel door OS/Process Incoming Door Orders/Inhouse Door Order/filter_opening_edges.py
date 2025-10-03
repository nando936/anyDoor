import cv2
import numpy as np
import sys

def filter_opening_edges(image_path):
    """Filter edges to show only door/drawer openings"""

    # Read image
    image = cv2.imread(image_path.replace('\\', '/'))
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} total contours")

    # Create blank image for filtered edges
    filtered_edges = np.zeros_like(edges)

    # Filter criteria for door/drawer openings
    min_width = 80   # Minimum width for opening
    min_height = 60  # Minimum height for opening
    max_width = image.shape[1] * 0.6   # Not larger than 60% of image width
    max_height = image.shape[0] * 0.8  # Not larger than 80% of image height
    min_aspect = 0.3  # Minimum aspect ratio (height/width or width/height)
    max_aspect = 4.0  # Maximum aspect ratio

    kept_openings = []

    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio
        aspect = h / w if w > 0 else 0
        if aspect < 1:
            aspect = w / h if h > 0 else 0

        # Filter by size and shape
        if (w >= min_width and h >= min_height and
            w <= max_width and h <= max_height and
            aspect >= min_aspect and aspect <= max_aspect):

            # Draw this contour on filtered image
            cv2.drawContours(filtered_edges, [contour], -1, 255, 1)

            kept_openings.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'aspect': aspect
            })

    print(f"Kept {len(kept_openings)} opening-like contours")

    # Sort by position (top to bottom, left to right)
    kept_openings.sort(key=lambda o: (o['y'], o['x']))

    # Print details
    for i, opening in enumerate(kept_openings):
        print(f"Opening {i+1}: x={opening['x']}, y={opening['y']}, "
              f"w={opening['width']}, h={opening['height']}, "
              f"aspect={opening['aspect']:.2f}")

    # Save filtered edges
    output_path = image_path.replace('\\', '/').replace('.png', '_filtered_edges.png')
    cv2.imwrite(output_path, filtered_edges)
    print(f"\n[OK] Filtered edges saved to: {output_path}")

    # Also create version with rectangles drawn
    output_with_boxes = image.copy()
    for opening in kept_openings:
        cv2.rectangle(output_with_boxes,
                     (opening['x'], opening['y']),
                     (opening['x'] + opening['width'], opening['y'] + opening['height']),
                     (0, 255, 0), 2)
        # Add label
        label = f"{opening['width']}x{opening['height']}"
        cv2.putText(output_with_boxes, label,
                   (opening['x'], opening['y'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    boxes_path = output_path.replace('_filtered_edges.png', '_opening_boxes.png')
    cv2.imwrite(boxes_path, output_with_boxes)
    print(f"[OK] Boxes overlay saved to: {boxes_path}")

    return kept_openings

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_opening_edges.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    filter_opening_edges(image_path)
