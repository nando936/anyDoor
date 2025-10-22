"""OCR handwritten text using Claude CLI (Max plan)"""
import sys
import os
import subprocess
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python ocr_with_claude.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if image_path.startswith('\\\\'):
    image_path = image_path.replace('\\', '/')

if not os.path.exists(image_path):
    print(f"[ERROR] Image not found: {image_path}")
    sys.exit(1)

# Get image directory for --add-dir permission
image_dir = os.path.dirname(image_path)

prompt = f"""Please read all the handwritten measurements from this image: {image_path}

This is a table of door measurements. Each line has the format:
QTY - WIDTH x HEIGHT  NOTES

IMPORTANT: Pay very close attention to the fractions. Common fractions in these measurements are:
- 1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16
- 1/8, 3/8, 5/8, 7/8
- 1/4, 1/2, 3/4

Look carefully at each fraction - for example "13/16" should be read as exactly that, not simplified or changed.

Please extract ALL measurements you can see and output them in this exact format, one per line:
[qty]-[width] x [height]  [notes if any]

For example:
4-15 13/16 x 32 1/2
2-17 1/4 x 41 5/16  1/4 MDF

Just give me the measurements, nothing else. Read the fractions EXACTLY as written."""

# Call Claude via CLI
claude_exe = r'C:\Users\nando\AppData\Roaming\npm\claude.cmd'

try:
    print(f"[INFO] Calling Claude CLI to read handwritten text...")
    result = subprocess.run(
        f'{claude_exe} --print --add-dir "{image_dir}"',
        input=prompt,
        capture_output=True,
        text=True,
        timeout=60,
        shell=True
    )

    if result.returncode != 0:
        print(f"[ERROR] Claude call failed with code {result.returncode}")
        if result.stderr:
            print(f"  STDERR: {result.stderr}")
        if result.stdout:
            print(f"  STDOUT: {result.stdout}")
        sys.exit(1)

    # Output Claude's response
    response = result.stdout.strip()
    print(response)

except subprocess.TimeoutExpired:
    print("[ERROR] Claude call timed out")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Claude call failed: {e}")
    sys.exit(1)
