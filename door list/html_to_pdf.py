import sys
import os
from playwright.sync_api import sync_playwright
import time

def html_to_pdf(html_file, pdf_file):
    """Convert HTML file to PDF using Playwright"""
    
    # Set UTF-8 encoding for Windows
    sys.stdout.reconfigure(encoding='utf-8')
    
    print(f"Converting {html_file} to PDF...")
    
    with sync_playwright() as p:
        # Launch browser in headless mode
        browser = p.chromium.launch(headless=True)
        
        try:
            # Create a new page
            page = browser.new_page()
            
            # Get absolute path for the HTML file
            abs_path = os.path.abspath(html_file)
            file_url = f"file:///{abs_path.replace(os.sep, '/')}"
            
            print(f"Loading HTML from: {file_url}")
            
            # Navigate to the HTML file
            page.goto(file_url, wait_until='networkidle')
            
            # Wait a bit for any JavaScript to execute
            time.sleep(1)
            
            # Generate PDF with appropriate settings
            pdf_options = {
                'path': pdf_file,
                'format': 'Letter',
                'print_background': True,
                'margin': {
                    'top': '0.5in',
                    'bottom': '0.5in',
                    'left': '0.5in',
                    'right': '0.5in'
                },
                'scale': 0.95
            }
            
            page.pdf(**pdf_options)
            
            print(f"[OK] PDF saved as: {pdf_file}")
            
        except Exception as e:
            print(f"[ERROR] Failed to convert: {str(e)}")
            return False
        
        finally:
            browser.close()
    
    return True

if __name__ == "__main__":
    # Convert Paul Revere form to PDF
    html_file = "paul_revere_door_order_302.html"
    pdf_file = "paul_revere_door_order_302.pdf"
    
    if os.path.exists(html_file):
        success = html_to_pdf(html_file, pdf_file)
        if success:
            print(f"\n[DONE] Successfully created {pdf_file}")
            print(f"File size: {os.path.getsize(pdf_file):,} bytes")
    else:
        print(f"[ERROR] HTML file not found: {html_file}")