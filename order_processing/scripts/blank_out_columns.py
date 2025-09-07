import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def create_white_overlay():
    """Create white rectangles to blank out specific columns"""
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    
    # Set white color for rectangles
    can.setFillColorRGB(1, 1, 1)  # White
    can.setStrokeColorRGB(0, 0, 0)  # Black border to maintain table lines
    
    # Based on the PDF structure, blank out the columns:
    # QTY column (approximately x=85-115)
    # Width column (approximately x=140-190) 
    # Height column (approximately x=210-260)
    # Doors/Drawer Details column (approximately x=290-400)
    
    # Row heights - these are the y positions for each row in the table
    # Starting from row 1 going down
    row_positions = [
        540,  # Row 1
        523,  # Row 2
        506,  # Row 3
        489,  # Row 4
        472,  # Row 5 (empty in original)
        455,  # Row 6 (empty in original)
        438,  # Row 7 (empty in original)
        421,  # Row 8 (empty in original)
        404,  # Row 9 (empty in original)
        387,  # Row 10 (empty in original)
        370,  # Row 11 (empty in original)
        353,  # Row 12 (empty in original)
        336,  # Row 13 (empty in original)
        319,  # Row 14 (empty in original)
    ]
    
    # Blank out each cell for the first 4 rows (which have content)
    for i, y_pos in enumerate(row_positions[:4]):
        # QTY column
        can.rect(88, y_pos-2, 35, 14, fill=1, stroke=0)
        
        # Width column  
        can.rect(143, y_pos-2, 55, 14, fill=1, stroke=0)
        
        # Height column
        can.rect(213, y_pos-2, 55, 14, fill=1, stroke=0)
        
        # Doors/Drawer Details column
        can.rect(293, y_pos-2, 115, 14, fill=1, stroke=0)
    
    # Also blank out the customer info fields
    print("Blanking out customer information fields...")
    
    # Customer info section
    can.rect(65, 697, 200, 12, fill=1, stroke=0)  # Name
    can.rect(75, 684, 200, 12, fill=1, stroke=0)  # Address  
    can.rect(75, 671, 200, 12, fill=1, stroke=0)  # Phone
    can.rect(65, 658, 200, 12, fill=1, stroke=0)  # Email
    
    # Job name
    can.rect(90, 728, 200, 15, fill=1, stroke=0)
    
    # Door specs that need changing
    can.rect(110, 618, 120, 15, fill=1, stroke=0)  # Wood species
    can.rect(475, 618, 50, 15, fill=1, stroke=0)   # Door number
    can.rect(185, 598, 120, 12, fill=1, stroke=0)  # Outside edge
    can.rect(385, 598, 120, 12, fill=1, stroke=0)  # Sticky/Sticking
    
    can.save()
    packet.seek(0)
    return PdfReader(packet)

def blank_pdf_columns():
    """Blank out specific columns in the PDF"""
    
    print("Blanking out columns in the order form...")
    print("-" * 50)
    
    try:
        # Read the copied PDF
        existing_pdf = PdfReader(open("door_231_order_form_copy.pdf", "rb"))
        output = PdfWriter()
        
        # Get the first page
        page = existing_pdf.pages[0]
        
        # Create white overlay
        overlay_pdf = create_white_overlay()
        overlay_page = overlay_pdf.pages[0]
        
        # Merge the overlay with the original page
        page.merge_page(overlay_page)
        output.add_page(page)
        
        # Add remaining pages as-is
        for i in range(1, len(existing_pdf.pages)):
            output.add_page(existing_pdf.pages[i])
        
        # Write to new file
        output_file = "door_231_order_form_blanked.pdf"
        with open(output_file, "wb") as outputStream:
            output.write(outputStream)
        
        print(f"[OK] Created {output_file} with blanked columns")
        print("Blanked out:")
        print("  - QTY column")
        print("  - Width column") 
        print("  - Height column")
        print("  - Doors/Drawer Details column")
        print("  - Customer information fields")
        print("  - Job name")
        print("  - Wood species, Door #, edges")
        
    except Exception as e:
        print(f"[ERROR] Failed to blank columns: {e}")

if __name__ == "__main__":
    blank_pdf_columns()