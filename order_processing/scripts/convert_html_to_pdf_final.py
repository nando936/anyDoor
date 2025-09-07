import os
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sys

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def html_to_pdf(html_file, pdf_file):
    """Convert HTML file to PDF using Selenium"""
    print(f"Converting {html_file} to PDF...")
    
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Load the HTML file
        file_url = f"file:///{os.path.abspath(html_file).replace(os.sep, '/')}"
        print(f"Loading: {file_url}")
        driver.get(file_url)
        
        # Generate PDF with exact letter size
        pdf_data = driver.execute_cdp_cmd('Page.printToPDF', {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'scale': 1.0,
            'paperWidth': 8.5,
            'paperHeight': 11,
            'marginTop': 0.5,
            'marginBottom': 0.5,
            'marginLeft': 0.5,
            'marginRight': 0.5,
            'preferCSSPageSize': True
        })
        
        # Save PDF
        with open(pdf_file, 'wb') as f:
            f.write(base64.b64decode(pdf_data['data']))
        
        driver.quit()
        print(f"[OK] Successfully created {pdf_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] PDF conversion failed: {e}")
        return False

if __name__ == "__main__":
    # Convert the HTML form to PDF
    html_to_pdf("door_order_form_exact.html", "door_231_order_form_FINAL.pdf")
    print("\nDoor #231 order form is ready!")
    print("Customer: Paul Revere")
    print("Total doors: 15 (Cabinet #4 reduced to 1 for trash drawer)")