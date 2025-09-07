import PyPDF2
from PyPDF2 import PdfReader
import pdfplumber
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def examine_with_pdfplumber():
    """Use pdfplumber to extract text and positions"""
    print("Examining PDF with pdfplumber...")
    print("-" * 60)
    
    try:
        with pdfplumber.open("door_231_order_form_copy.pdf") as pdf:
            page = pdf.pages[0]
            
            # Get page dimensions
            print(f"Page dimensions: {page.width} x {page.height}")
            print(f"Page bbox: {page.bbox}")
            print()
            
            # Extract tables
            tables = page.extract_tables()
            if tables:
                print(f"Found {len(tables)} tables")
                for i, table in enumerate(tables):
                    print(f"\nTable {i+1}:")
                    for row in table[:5]:  # Show first 5 rows
                        print(f"  {row}")
            
            # Extract text with positions
            print("\nText elements with positions (showing items with 'QTY', numbers, or 'doors'):")
            words = page.extract_words()
            for word in words:
                text = word['text']
                if any(x in text.upper() for x in ['QTY', 'WIDTH', 'HEIGHT', 'DOOR', 'DRAWER', '12', '36', '15', '45', '14']):
                    print(f"  '{text}' at x={word['x0']:.1f}-{word['x1']:.1f}, y={word['top']:.1f}-{word['bottom']:.1f}")
            
            # Look for form fields
            print("\nChecking for form fields...")
            if hasattr(page, 'annots'):
                annots = page.annots
                if annots:
                    print(f"Found {len(annots)} annotations")
                    for annot in annots[:5]:
                        print(f"  {annot}")
            
    except Exception as e:
        print(f"[ERROR] pdfplumber examination failed: {e}")

def examine_with_pypdf2():
    """Use PyPDF2 to check for form fields"""
    print("\n" + "=" * 60)
    print("Examining PDF with PyPDF2...")
    print("-" * 60)
    
    try:
        reader = PdfReader("door_231_order_form_copy.pdf")
        
        # Check basic info
        print(f"Number of pages: {len(reader.pages)}")
        print(f"PDF encrypted: {reader.is_encrypted}")
        
        # Check for forms
        if "/AcroForm" in reader.trailer["/Root"]:
            print("\nForm fields found!")
            fields = reader.get_fields()
            if fields:
                print(f"Number of fields: {len(fields)}")
                for name, field in list(fields.items())[:10]:
                    print(f"  Field: {name}")
                    if '/V' in field:
                        print(f"    Value: {field['/V']}")
        else:
            print("\nNo AcroForm fields found")
        
        # Get page info
        page = reader.pages[0]
        print(f"\nPage 0 MediaBox: {page.mediabox}")
        
        # Try to extract text
        text = page.extract_text()
        lines = text.split('\n')
        print(f"\nFirst 20 lines of extracted text:")
        for i, line in enumerate(lines[:20]):
            if line.strip():
                print(f"  {i:2}: {line[:80]}")
        
    except Exception as e:
        print(f"[ERROR] PyPDF2 examination failed: {e}")

if __name__ == "__main__":
    examine_with_pdfplumber()
    examine_with_pypdf2()