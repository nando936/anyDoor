import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import sys

# Set encoding for Windows  
sys.stdout.reconfigure(encoding='utf-8')

def create_text_overlay():
    """Create text overlay with new data"""
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(612, 792))
    
    # Set black color for text
    can.setFillColorRGB(0, 0, 0)  # Black
    can.setFont("Helvetica", 10)
    
    print("Adding new data...")
    
    # Customer info
    can.drawString(100, 639, "Paul Revere")  # Name
    can.drawString(73, 626, "1776 Liberty Lane, Boston TX 77385")  # Address
    can.drawString(73, 616, "(281) 555-1775")  # Phone
    can.drawString(60, 607, "prevere@colonialcabinets.com")  # Email
    
    # Job name
    can.drawString(80, 752, "Revere Kitchen Remodel")
    
    # Wood Species and Door #
    can.drawString(68, 530, "White Oak")
    can.setFont("Helvetica-Bold", 11)
    can.setFillColorRGB(1, 0, 0)  # Red for door number
    can.drawString(305, 530, "231")
    
    # Outside Edge and Sticking
    can.setFillColorRGB(0, 0, 0)  # Back to black
    can.setFont("Helvetica", 9)
    can.drawString(95, 510, "Standard for #231")
    can.drawString(245, 510, "Standard for #231")
    
    # Table data - 8 door items
    can.setFont("Helvetica", 9)
    
    # Row 1: Cabinet #1
    can.drawString(58, 408, "2")  # QTY
    can.drawString(92, 408, "14 3/8")  # Width
    can.drawString(142, 408, "24 3/4")  # Height
    can.drawString(197, 408, "doors")  # Type
    can.setFont("Helvetica", 8)
    can.drawString(250, 408, "Cabinet #1")  # Notes
    
    # Row 2: Cabinet #2
    can.setFont("Helvetica", 9)
    can.drawString(58, 380, "2")  # QTY
    can.drawString(92, 380, "14 7/16")  # Width
    can.drawString(142, 380, "24 3/4")  # Height
    can.drawString(197, 380, "doors")  # Type
    can.setFont("Helvetica", 8)
    can.drawString(250, 380, "Cabinet #2")  # Notes
    
    # Row 3: Cabinet #4 (trash drawer)
    can.setFont("Helvetica", 9)
    can.drawString(58, 356, "1")  # QTY (reduced to 1)
    can.drawString(92, 356, "14 3/4")  # Width
    can.drawString(142, 356, "24 3/4")  # Height
    can.drawString(197, 356, "doors")  # Type
    can.setFont("Helvetica", 7)
    can.drawString(250, 356, "No hinge boring - trash #4")  # Notes
    
    # Row 4: Cabinet #6
    can.setFont("Helvetica", 9)
    can.drawString(58, 332, "2")  # QTY
    can.drawString(92, 332, "13 3/4")  # Width
    can.drawString(142, 332, "24 3/4")  # Height
    can.drawString(197, 332, "doors")  # Type
    can.setFont("Helvetica", 8)
    can.drawString(250, 332, "Cabinet #6")  # Notes
    
    # Continue on empty rows (5-8)
    can.setFont("Helvetica", 9)
    
    # Row 5: Cabinet #7
    can.drawString(30, 308, "5")  # Row number
    can.drawString(58, 308, "2")  # QTY
    can.drawString(92, 308, "14 3/4")  # Width
    can.drawString(142, 308, "24 3/4")  # Height
    can.drawString(197, 308, "doors")  # Type
    can.setFont("Helvetica", 8)
    can.drawString(250, 308, "Cabinet #7")  # Notes
    
    # Row 6: Cabinet #10
    can.setFont("Helvetica", 9)
    can.drawString(30, 284, "6")  # Row number
    can.drawString(58, 284, "2")  # QTY
    can.drawString(92, 284, "13 13/16")  # Width
    can.drawString(142, 284, "24 3/4")  # Height
    can.drawString(197, 284, "doors")  # Type
    can.setFont("Helvetica", 8)
    can.drawString(250, 284, "Cabinet #10")  # Notes
    
    # Row 7: Cabinet #23
    can.setFont("Helvetica", 9)
    can.drawString(30, 260, "7")  # Row number
    can.drawString(58, 260, "2")  # QTY
    can.drawString(92, 260, "16 1/2")  # Width
    can.drawString(142, 260, "41 7/8")  # Height
    can.drawString(197, 260, "doors")  # Type
    can.setFont("Helvetica", 8)
    can.drawString(250, 260, "Cabinet #23")  # Notes
    
    # Row 8: Cabinet #24
    can.setFont("Helvetica", 9)
    can.drawString(30, 236, "8")  # Row number
    can.drawString(58, 236, "2")  # QTY
    can.drawString(92, 236, "16 1/2")  # Width
    can.drawString(142, 236, "41 7/8")  # Height
    can.drawString(197, 236, "doors")  # Type
    can.setFont("Helvetica", 8)
    can.drawString(250, 236, "Cabinet #24")  # Notes
    
    can.save()
    packet.seek(0)
    return PdfReader(packet)

def fill_blanked_pdf():
    """Fill the blanked PDF with new data"""
    
    print("Filling blanked PDF with Door #231 data...")
    print("-" * 50)
    
    try:
        # Read the blanked PDF
        existing_pdf = PdfReader(open("door_231_blanked_v2.pdf", "rb"))
        output = PdfWriter()
        
        # Get the first page
        page = existing_pdf.pages[0]
        
        # Create text overlay
        overlay_pdf = create_text_overlay()
        overlay_page = overlay_pdf.pages[0]
        
        # Merge the overlay with the blanked page
        page.merge_page(overlay_page)
        output.add_page(page)
        
        # Add remaining pages as-is
        for i in range(1, len(existing_pdf.pages)):
            output.add_page(existing_pdf.pages[i])
        
        # Write to final file
        output_file = "door_231_order_final.pdf"
        with open(output_file, "wb") as outputStream:
            output.write(outputStream)
        
        print(f"[OK] Created {output_file}")
        print("Door #231 order form complete!")
        print("Total doors: 15 (Cabinet #4 reduced to 1)")
        
    except Exception as e:
        print(f"[ERROR] Failed to fill PDF: {e}")

if __name__ == "__main__":
    fill_blanked_pdf()