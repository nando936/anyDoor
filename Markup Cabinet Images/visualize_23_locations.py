#!/usr/bin/env python3
"""
Visualize where the "23" measurements should be on page 3.
"""

import cv2
import numpy as np
import sys
import json

def main():
    # Load the image
    image_path = "page_3.png" if len(sys.argv) < 2 else sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    # Create a copy for annotation
    annotated = image.copy()

    # Old data had two "23" measurements
    old_23_positions = [
        (471, 1148, "Old 23 #1"),  # First 23 from old data
        (1253, 1250, "Old 23 #2")  # Second 23 from old data
    ]

    # New data has one "23" measurement
    new_23_position = (835, 834, "New 23")

    # Draw old positions in RED
    for x, y, label in old_23_positions:
        cv2.circle(annotated, (int(x), int(y)), 15, (0, 0, 255), 3)
        cv2.putText(annotated, label, (int(x) - 50, int(y) - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated, f"({x},{y})", (int(x) - 50, int(y) + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw new position in GREEN
    x, y, label = new_23_position
    cv2.circle(annotated, (int(x), int(y)), 15, (0, 255, 0), 3)
    cv2.putText(annotated, label, (int(x) - 50, int(y) - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"({x},{y})", (int(x) - 50, int(y) + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Also show where we expect measurements based on the new measurements file
    try:
        with open("page_3_measurements_data.json", "r") as f:
            data = json.load(f)

        # Draw all measurements in BLUE
        for m in data["measurements"]:
            text = m["text"]
            x, y = m["position"]
            cv2.circle(annotated, (int(x), int(y)), 8, (255, 128, 0), 2)
            cv2.putText(annotated, text, (int(x) + 15, int(y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
    except:
        pass

    # Save the annotated image
    output_path = "page_3_23_locations.png"
    cv2.imwrite(output_path, annotated)
    print(f"Saved visualization to {output_path}")

    # Print summary
    print("\nSummary:")
    print("RED circles = Old '23' positions from previous data")
    print("GREEN circle = New '23' position from current detection")
    print("BLUE circles = All measurements from current detection")
    print("\nThe issue: We're missing one '23' measurement!")
    print("Old data had two '23' at different Y positions (1148 and 1250)")
    print("New data only has one '23' at Y position 834")

if __name__ == "__main__":
    main()