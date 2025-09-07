import shutil
import os
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

source_path = r"\\vmware-host\Shared Folders\d\OneDrive\customers\archived customers\Jobs\1-302 door order.pdf"
dest_path = "1-302_door_order.pdf"

try:
    if os.path.exists(source_path):
        shutil.copy2(source_path, dest_path)
        print(f"[OK] Successfully copied file to {dest_path}")
        print(f"File size: {os.path.getsize(dest_path)} bytes")
    else:
        print(f"[ERROR] Source file not found: {source_path}")
except Exception as e:
    print(f"[ERROR] {e}")