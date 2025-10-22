"""
Extract logo image from the estimate template PDF
"""
import fitz  # PyMuPDF
import sys
import os

def extract_logo(pdf_path, output_dir):
    """Extract images from PDF"""
    doc = fitz.open(pdf_path)
    page = doc[0]  # First page

    # Get all images on the page
    image_list = page.get_images()

    print(f"Found {len(image_list)} images on page 1")

    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        output_path = os.path.join(output_dir, f"logo.{image_ext}")

        with open(output_path, "wb") as f:
            f.write(image_bytes)

        print(f"[OK] Saved logo to: {output_path}")
        print(f"    Format: {image_ext}")
        print(f"    Size: {len(image_bytes)} bytes")

    doc.close()

if __name__ == '__main__':
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "//vmware-host/Shared Folders/D/OneDrive/customers/raised panel/True Custom/all_pages/estimate.pdf"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/nando/Projects/anyDoor/raised panel door OS/Process Incoming Door Orders/True Custom Door Order"

    extract_logo(pdf_path, output_dir)
