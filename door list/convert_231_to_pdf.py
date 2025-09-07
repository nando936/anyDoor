import base64
import sys
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def html_to_pdf(html_path, pdf_path):
    """Convert HTML file to PDF using Selenium"""
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Load the HTML file
        file_url = f"file:///{os.path.abspath(html_path).replace(os.sep, '/')}"
        print(f"Loading: {file_url}")
        driver.get(file_url)
        
        # Wait for page to render
        time.sleep(2)
        
        # Generate PDF
        pdf_data = driver.execute_cdp_cmd('Page.printToPDF', {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'scale': 0.9,
            'paperWidth': 8.5,
            'paperHeight': 11,
            'marginTop': 0.5,
            'marginBottom': 0.5,
            'marginLeft': 0.5,
            'marginRight': 0.5
        })
        
        # Save PDF
        with open(pdf_path, 'wb') as f:
            f.write(base64.b64decode(pdf_data['data']))
        
        driver.quit()
        print(f"[OK] PDF created: {pdf_path}")
        return True
    except Exception as e:
        print(f"[ERROR] PDF conversion failed: {e}")
        return False

# Main execution
if __name__ == "__main__":
    print("Converting Paul Revere Door Order 231 to PDF")
    print("=" * 60)
    
    html_file = r"C:\Users\nando\Projects\anyDoor\door list\paul_revere_door_order_231.html"
    pdf_file = r"C:\Users\nando\Projects\anyDoor\door list\paul_revere_door_order_231_FILLED.pdf"
    
    if os.path.exists(html_file):
        if html_to_pdf(html_file, pdf_file):
            print(f"Successfully created: {pdf_file}")
        else:
            print("Failed to create PDF")
    else:
        print(f"[ERROR] HTML file not found: {html_file}")