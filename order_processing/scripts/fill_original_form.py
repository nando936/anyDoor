import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def create_overlay_for_page1():
    """Create overlay with new data for page 1"""
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    
    # Cover old data with white rectangles first
    can.setFillColorRGB(1, 1, 1)  # White
    
    # Cover old date
    can.rect(45, 742, 100, 15, fill=1, stroke=0)
    # Cover old job name  
    can.rect(90, 728, 200, 15, fill=1, stroke=0)
    # Cover old customer name
    can.rect(65, 697, 200, 12, fill=1, stroke=0)
    # Cover old address
    can.rect(75, 684, 200, 12, fill=1, stroke=0)
    # Cover old phone
    can.rect(75, 671, 200, 12, fill=1, stroke=0)
    # Cover old email
    can.rect(65, 658, 200, 12, fill=1, stroke=0)
    
    # Cover door number
    can.rect(475, 618, 50, 15, fill=1, stroke=0)
    # Cover wood species
    can.rect(110, 618, 120, 15, fill=1, stroke=0)
    # Cover outside edge
    can.rect(185, 598, 120, 12, fill=1, stroke=0)
    # Cover sticky/sticking
    can.rect(385, 598, 120, 12, fill=1, stroke=0)
    
    # Now add new text
    can.setFillColorRGB(0, 0, 0)  # Black
    can.setFont("Helvetica", 10)
    
    # New customer info
    can.drawString(50, 745, "1/7/2025")  # Keep same date
    can.drawString(95, 731, "Revere Kitchen Remodel")
    
    can.drawString(70, 699, "Paul Revere")
    can.drawString(80, 686, "1776 Liberty Lane, Boston TX 77385")
    can.drawString(80, 673, "(281) 555-1775")
    can.drawString(70, 660, "prevere@colonialcabinets.com")
    
    # New door specs
    can.setFont("Helvetica-Bold", 11)
    can.drawString(115, 620, "White Oak")
    can.setFillColorRGB(1, 0, 0)  # Red for door number
    can.drawString(480, 620, "231")
    
    can.setFillColorRGB(0, 0, 0)  # Black
    can.setFont("Helvetica", 10)
    can.drawString(190, 600, "Standard for #231")
    can.drawString(390, 600, "Standard for #231")
    
    # Cover the old order items and add new ones
    # Clear the table rows
    y_positions = [540, 523, 506, 489, 472, 455, 438]  # First 7 rows
    
    for y in y_positions:
        can.setFillColorRGB(1, 1, 1)
        can.rect(45, y, 500, 15, fill=1, stroke=0)
    
    # Add new door items
    can.setFillColorRGB(0, 0, 0)
    can.setFont("Helvetica", 9)
    
    doors_data = [
        ("1", "2", "14 3/8", "24 3/4", "doors", "Cabinet #1"),
        ("2", "2", "14 7/16", "24 3/4", "doors", "Cabinet #2"),
        ("3", "1", "14 3/4", "24 3/4", "doors", "No hinge boring - trash drawer #4"),
        ("4", "2", "13 3/4", "24 3/4", "doors", "Cabinet #6"),
        ("5", "2", "14 3/4", "24 3/4", "doors", "Cabinet #7"),
        ("6", "2", "13 13/16", "24 3/4", "doors", "Cabinet #10"),
        ("7", "2", "16 1/2", "41 7/8", "doors", "Cabinet #23"),
    ]
    
    y_start = 542
    for i, (num, qty, width, height, type_str, notes) in enumerate(doors_data):
        y = y_start - (i * 17)
        can.drawString(55, y, num)
        can.drawString(95, y, qty)
        can.drawString(150, y, width)
        can.drawString(220, y, height)
        can.drawString(310, y, type_str)
        can.drawString(400, y, notes)
    
    # Add 8th item on next line
    can.drawString(55, 421, "8")
    can.drawString(95, 421, "2")
    can.drawString(150, 421, "16 1/2")
    can.drawString(220, 421, "41 7/8")
    can.drawString(310, 421, "doors")
    can.drawString(400, 421, "Cabinet #24")
    
    can.save()
    packet.seek(0)
    return PdfReader(packet)

def fill_pdf_form():
    """Fill the original PDF form with new data"""
    
    print("Filling original order form with Door #231 data...")
    print("-" * 50)
    
    try:
        # Read the original PDF
        existing_pdf = PdfReader(open("user_submited_ORDER_FORM_.pdf", "rb"))
        output = PdfWriter()
        
        # Get the first page
        page = existing_pdf.pages[0]
        
        # Create overlay with new data
        overlay_pdf = create_overlay_for_page1()
        overlay_page = overlay_pdf.pages[0]
        
        # Merge the overlay with the original page
        page.merge_page(overlay_page)
        output.add_page(page)
        
        # Add remaining pages as-is (they're mostly empty)
        for i in range(1, len(existing_pdf.pages)):
            output.add_page(existing_pdf.pages[i])
        
        # Write to new file
        output_file = "door_231_filled_order.pdf"
        with open(output_file, "wb") as outputStream:
            output.write(outputStream)
        
        print(f"[OK] Generated {output_file}")
        print("Total doors: 15 (Cabinet #4 reduced to 1 for trash drawer)")
        
    except Exception as e:
        print(f"[ERROR] Failed to fill PDF: {e}")
        print("\nTrying alternative method with PyPDF2...")
        
        # Alternative: Try using form fields if they exist
        try:
            reader = PdfReader("user_submited_ORDER_FORM_.pdf")
            writer = PdfWriter()
            
            # Check if there are form fields
            if "/AcroForm" in reader.trailer["/Root"]:
                writer.append_pages_from_reader(reader)
                writer.update_page_form_field_values(
                    writer.pages[0],
                    {
                        "Date": "1/7/2025",
                        "Job Name": "Revere Kitchen Remodel",
                        "Name": "Paul Revere",
                        "Address": "1776 Liberty Lane, Boston TX 77385",
                        "Phone": "(281) 555-1775",
                        "Email": "prevere@colonialcabinets.com",
                        "Wood Species": "White Oak",
                        "Door #": "231",
                        "Outside Edge": "Standard for #231",
                        "Sticky": "Standard for #231"
                    }
                )
                
                with open("door_231_filled_order.pdf", "wb") as output_file:
                    writer.write(output_file)
                print("[OK] Filled form using form fields")
            else:
                print("[INFO] No form fields found, overlay method required")
                
        except Exception as e2:
            print(f"[ERROR] Alternative method also failed: {e2}")

if __name__ == "__main__":
    fill_pdf_form()