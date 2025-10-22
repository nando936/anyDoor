"""Crop a section, upscale it, then have Claude read it"""
import sys
import os
import subprocess
import cv2
from pathlib import Path

if len(sys.argv) < 6:
    print("Usage: python crop_upscale_read.py <image_path> <section_name> <y_start> <y_end> <upscale_factor>")
    print("Example: python crop_upscale_read.py page1.jpg 'Living Room' 300 400 4")
    sys.exit(1)

image_path = sys.argv[1]
section_name = sys.argv[2]
y_start = int(sys.argv[3])
y_end = int(sys.argv[4])
upscale_factor = int(sys.argv[5])

if image_path.startswith('\\\\'):
    image_path = image_path.replace('\\', '/')

# Read image
print(f"[INFO] Reading {image_path}...")
image = cv2.imread(image_path)
if image is None:
    print(f"[ERROR] Could not read image")
    sys.exit(1)

height, width = image.shape[:2]
print(f"[INFO] Image size: {width}x{height}")

# Crop the section (full width, specified y range)
print(f"[INFO] Cropping {section_name} section (y={y_start} to {y_end})...")
cropped = image[y_start:y_end, :]

crop_height, crop_width = cropped.shape[:2]
print(f"[INFO] Crop size: {crop_width}x{crop_height}")

# Upscale
print(f"[INFO] Upscaling by {upscale_factor}x...")
upscaled = cv2.resize(cropped, (crop_width * upscale_factor, crop_height * upscale_factor),
                      interpolation=cv2.INTER_CUBIC)

upscale_height, upscale_width = upscaled.shape[:2]
print(f"[INFO] Upscaled size: {upscale_width}x{upscale_height}")

# Save upscaled crop
image_dir = os.path.dirname(image_path)
output_name = section_name.replace(' ', '_').lower() + '_upscaled.jpg'
output_path = os.path.join(image_dir, output_name)
cv2.imwrite(output_path, upscaled)
print(f"[OK] Saved upscaled crop: {output_path}")

# Now send to Claude to read
print(f"\n[INFO] Sending to Claude to read...")

prompt = f"""Please read all the handwritten measurements from this upscaled image: {output_path}

This is the {section_name} section. Each line has the format:
QTY - WIDTH x HEIGHT  NOTES

IMPORTANT: Pay very close attention to the fractions. Common fractions are:
- 1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16
- 1/8, 3/8, 5/8, 7/8
- 1/4, 1/2, 3/4

Look very carefully at each number and fraction.

Output format (one per line):
[qty]-[width] x [height]  [notes if any]

Example:
4-15 13/16 x 32 1/2
2-17 1/4 x 41 5/16  1/4 MDF

Just give me the measurements, nothing else."""

claude_exe = r'C:\Users\nando\AppData\Roaming\npm\claude.cmd'

try:
    result = subprocess.run(
        f'{claude_exe} --print --add-dir "{image_dir}"',
        input=prompt,
        capture_output=True,
        text=True,
        timeout=90,
        shell=True
    )

    if result.returncode != 0:
        print(f"[ERROR] Claude call failed")
        if result.stdout:
            print(result.stdout)
        sys.exit(1)

    print("\n=== CLAUDE'S READING ===")
    print(result.stdout.strip())

except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)
