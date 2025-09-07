import os
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def search_files(search_path):
    """Search for all files in the Suarez Group QB folder"""
    found_files = []
    
    if not os.path.exists(search_path):
        print(f"[ERROR] Path does not exist: {search_path}")
        return found_files
        
    print(f"[OK] Searching in: {search_path}")
    print("-" * 50)
    
    try:
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
                    
    except PermissionError as e:
        print(f"[ERROR] Permission denied: {root}")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    return found_files

# Search path
search_path = r"\\vmware-host\Shared Folders\suarez group qb"

print("Searching for order forms and shop reports in Suarez Group QB folder...")
print("=" * 50)

found = search_files(search_path)

print("-" * 50)
print(f"\nTotal relevant files found: {len(found)}")

# Also list all files to see what's there
print("\n" + "=" * 50)
print("Listing ALL files in the folder:")
print("-" * 50)

try:
    for root, dirs, files in os.walk(search_path):
        level = root.replace(search_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 2 * (level + 1)
        for file in files[:50]:  # Limit to first 50 files per folder
            print(f'{sub_indent}- {file}')
        if len(files) > 50:
            print(f'{sub_indent}... and {len(files) - 50} more files')
except Exception as e:
    print(f"[ERROR] Could not list all files: {e}")