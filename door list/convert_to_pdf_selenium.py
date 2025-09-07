import sys
import os
import base64
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time

# Set UTF-8 encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def html_to_pdf_selenium(html_file, pdf_file):
    """Convert HTML to PDF using Selenium with Chrome"""
    
    print(f"Converting {html_file} to PDF using Selenium...")
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--allow-file-access-from-files')
    
    # Enable printing to PDF
    chrome_options.add_experimental_option('prefs', {
        'printing.print_preview_sticky_settings.appState': json.dumps({
            'recentDestinations': [{
                'id': 'Save as PDF',
                'origin': 'local',
                'account': ''
            }],
            'selectedDestinationId': 'Save as PDF',
            'version': 2
        })
    })
    
    try:
        # Initialize Chrome driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Get absolute path and load the HTML file
        abs_path = os.path.abspath(html_file)
        file_url = f"file:///{abs_path.replace(os.sep, '/')}"
        
        print(f"Loading: {file_url}")
        driver.get(file_url)
        
        # Wait for page to load
        time.sleep(2)
        
        # Execute print to PDF
        print_options = {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'preferCSSPageSize': True,
            'paperWidth': 8.5,
            'paperHeight': 11,
            'marginTop': 0.5,
            'marginBottom': 0.5,
            'marginLeft': 0.5,
            'marginRight': 0.5,
            'scale': 0.95
        }
        
        # Use Chrome DevTools Protocol to print to PDF
        result = driver.execute_cdp_cmd('Page.printToPDF', print_options)
        
        # Save the PDF
        pdf_data = base64.b64decode(result['data'])
        with open(pdf_file, 'wb') as f:
            f.write(pdf_data)
        
        print(f"[OK] PDF saved as: {pdf_file}")
        print(f"File size: {len(pdf_data):,} bytes")
        
        driver.quit()
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to convert: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        return False

if __name__ == "__main__":
    # Convert Paul Revere form to PDF
    html_file = "paul_revere_door_order_302.html"
    pdf_file = "paul_revere_door_order_302_FILLED.pdf"
    
    if os.path.exists(html_file):
        success = html_to_pdf_selenium(html_file, pdf_file)
        if success:
            print(f"\n[DONE] Successfully created {pdf_file}")
    else:
        print(f"[ERROR] HTML file not found: {html_file}")