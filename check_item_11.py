import pdfplumber
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = "1-302_door_order.pdf"

print("Checking original order for item #11 details...")
print("=" * 50)

try:
    with pdfplumber.open(pdf_path) as pdf:
        # Extract text from all pages
        all_text = ""
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                all_text += f"\n--- Page {i+1} ---\n{text}\n"
        
        # Look for lines containing drawer information
        lines = all_text.split('\n')
        
        print("Looking for drawer items around #11 range...")
        print("-" * 50)
        
        item_count = 0
        in_drawer_section = False
        
        for line in lines:
            # Look for drawer section
            if 'Solid Wood Drawer Front' in line or 'drawer' in line.lower():
                in_drawer_section = True
            
            # If we're in drawer section, look for items with dimensions
            if in_drawer_section and ('x' in line or 'X' in line or '×' in line):
                # Try to identify items with quantities and dimensions
                if any(char.isdigit() for char in line):
                    item_count += 1
                    if 9 <= item_count <= 13:  # Show items around #11
                        print(f"Item #{item_count}: {line.strip()}")
                        
        print("\n" + "=" * 50)
        print("Looking for specific patterns with 16 5/8 dimension...")
        print("-" * 50)
        
        for line in lines:
            if '16 5/8' in line or '16-5/8' in line or '16.625' in line:
                print(f"Found: {line.strip()}")
                
except Exception as e:
    print(f"[ERROR] {e}")
    print("\nTrying to display any text containing '16' and '5/8'...")
    try:
        with open(pdf_path, 'rb') as f:
            content = f.read()
            # This won't work well for PDF but might show something
            print("[Note: PDF is binary, text extraction may be incomplete]")
    except Exception as e2:
        print(f"[ERROR] Could not read file: {e2}")