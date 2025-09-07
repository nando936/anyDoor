import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def create_white_overlay():
    """Create white rectangles to blank out specific data"""
    packet = io.BytesIO()
    # PDF coordinates: origin at bottom-left, page is 612x792 points
    can = canvas.Canvas(packet, pagesize=(612, 792))
    
    # Set white color for rectangles
    can.setFillColorRGB(1, 1, 1)  # White
    
    # Convert y-coordinates (pdfplumber uses top-origin, reportlab uses bottom-origin)
    # Page height is 792, so reportlab_y = 792 - pdfplumber_y
    
    print("Creating white rectangles to blank out data...")
    
    # Based on pdfplumber coordinates:
    # QTY column: x=60.8-78.6, Items at y=375.9, 403.8, 427.6, 451.5
    # Width column: x=95.4-105.5 (the numbers)
    # Height column: x=144.9-155.0 (the numbers)
    # Doors/Drawer column: x=198.9-227.6
    
    # Table data rows (converting y coordinates)
    # Row 1 (5, 12, 36, doors): pdfplumber_y=375.9, reportlab_y=792-375.9-10=406
    can.rect(55, 406, 28, 12, fill=1, stroke=0)  # QTY "5"
    can.rect(92, 406, 18, 12, fill=1, stroke=0)  # Width "12"
    can.rect(142, 406, 18, 12, fill=1, stroke=0)  # Height "36"
    can.rect(195, 406, 35, 12, fill=1, stroke=0)  # "doors"
    
    # Row 2 (4, 36, 6, drawer): y=403.8, reportlab_y=792-403.8-10=378
    can.rect(55, 378, 28, 12, fill=1, stroke=0)  # QTY "4"
    can.rect(92, 378, 18, 12, fill=1, stroke=0)  # Width "36"
    can.rect(142, 378, 18, 12, fill=1, stroke=0)  # Height "6"
    can.rect(195, 378, 40, 12, fill=1, stroke=0)  # "drawer"
    
    # Row 3 (6, 15, 45, doors): y=427.6, reportlab_y=792-427.6-10=354
    can.rect(55, 354, 28, 12, fill=1, stroke=0)  # QTY "6"
    can.rect(92, 354, 18, 12, fill=1, stroke=0)  # Width "15"
    can.rect(142, 354, 18, 12, fill=1, stroke=0)  # Height "45"
    can.rect(195, 354, 35, 12, fill=1, stroke=0)  # "doors"
    
    # Row 4 (3, 14, 6, drawer): y=451.5, reportlab_y=792-451.5-10=330
    can.rect(55, 330, 28, 12, fill=1, stroke=0)  # QTY "3"
    can.rect(92, 330, 18, 12, fill=1, stroke=0)  # Width "14"
    can.rect(142, 330, 18, 12, fill=1, stroke=0)  # Height "6"
    can.rect(195, 330, 40, 12, fill=1, stroke=0)  # "drawer"
    
    # Also blank out the fields we want to change:
    # Customer name: "kevin fox" at approximately y=144.5, convert to 792-144.5-10=637
    can.rect(100, 637, 100, 12, fill=1, stroke=0)
    
    # Address: "12342 pander lane" at y=157.6, convert to 792-157.6-10=624
    can.rect(73, 624, 150, 12, fill=1, stroke=0)
    
    # Phone: "9367882762" at y=167.6, convert to 792-167.6-10=614
    can.rect(73, 614, 80, 12, fill=1, stroke=0)
    
    # Email: kevin@gmail.com (need to find exact position, approximately y=170)
    can.rect(60, 605, 120, 12, fill=1, stroke=0)
    
    # Job name: "wiggins job" (top of page, approximately y=750)
    can.rect(80, 750, 100, 12, fill=1, stroke=0)
    
    # Wood Species: "White Oak" 
    can.rect(68, 528, 70, 12, fill=1, stroke=0)
    
    # Door #: "308"
    can.rect(305, 528, 30, 12, fill=1, stroke=0)
    
    # Outside Edge: "Special"
    can.rect(95, 508, 50, 12, fill=1, stroke=0)
    
    # Sticky: "Bevel"
    can.rect(245, 508, 40, 12, fill=1, stroke=0)
    
    can.save()
    packet.seek(0)
    return PdfReader(packet)

def blank_pdf_data():
    """Blank out specific data in the PDF"""
    
    print("Blanking out data in the order form...")
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
        output_file = "door_231_blanked_v2.pdf"
        with open(output_file, "wb") as outputStream:
            output.write(outputStream)
        
        print(f"[OK] Created {output_file}")
        print("Blanked out table data and customer info")
        
    except Exception as e:
        print(f"[ERROR] Failed to blank data: {e}")

if __name__ == "__main__":
    blank_pdf_data()