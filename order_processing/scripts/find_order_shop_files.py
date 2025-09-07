import os
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def search_files(search_paths, patterns):
    """Search for files matching patterns in given paths"""
    found_files = []
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            print(f"[INFO] Path does not exist: {search_path}")
            continue
            
        print(f"[OK] Searching in: {search_path}")
        
        try:
            for root, dirs, files in os.walk(search_path):
                # Skip certain directories to speed up search
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', 'venv', '__pycache__']]
                
                for file in files:
                    file_lower = file.lower()
                    for pattern in patterns:
                        if pattern in file_lower:
                            full_path = os.path.join(root, file)
                            found_files.append(full_path)
                            print(f"  Found: {full_path}")
                            break
                            
        except PermissionError as e:
            print(f"[ERROR] Permission denied: {root}")
        except Exception as e:
            print(f"[ERROR] {e}")
    
    return found_files

# Search paths
search_paths = [
    r"D:\\",
    r"D:\door order",
    r"D:\suarez group qb",
    r"\\vmware-host\Shared Folders\d",
    r"\\vmware-host\Shared Folders\suarez group qb",
    r"C:\Users\nando\Projects\anyDoor"
]

# Patterns to search for
patterns = ['order', 'shop', 'report', 'form']

print("Starting search for order forms and shop reports...")
print("-" * 50)

found = search_files(search_paths, patterns)

print("-" * 50)
print(f"\nTotal files found: {len(found)}")

if found:
    print("\nSummary of found files:")
    for f in found:
        print(f"  - {f}")