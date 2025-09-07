import os
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def search_files(search_path):
    """Search for order, shop, report files in E shared folder"""
    found_files = []
    
    if not os.path.exists(search_path):
        print(f"[ERROR] Path does not exist: {search_path}")
        return found_files
        
    print(f"[OK] Searching in: {search_path}")
    print("-" * 50)
    
    try:
        # First, list top-level directories
        print("Top-level directories:")
        for item in os.listdir(search_path):
            item_path = os.path.join(search_path, item)
            if os.path.isdir(item_path):
                print(f"  - {item}/")
        
        print("\n" + "-" * 50)
        print("Searching for order/shop/report files...")
        
        for root, dirs, files in os.walk(search_path):
            # Skip certain directories to speed up search
            dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', 'venv', '__pycache__']]
            
            for file in files:
                file_lower = file.lower()
                # Look for order, shop, report, form files
                if any(keyword in file_lower for keyword in ['order', 'shop', 'report', 'form']):
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
                    print(f"  Found: {full_path}")
                    
                    # Stop after finding 200 files to avoid too much output
                    if len(found_files) >= 200:
                        print("\n[INFO] Stopping search after 200 files...")
                        return found_files
                    
    except PermissionError as e:
        print(f"[ERROR] Permission denied: {root}")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    return found_files

# Search path
search_path = r"\\vmware-host\Shared Folders\E"

print("Searching for order forms and shop reports in E shared folder...")
print("=" * 50)

found = search_files(search_path)

print("-" * 50)
print(f"\nTotal relevant files found: {len(found)}")