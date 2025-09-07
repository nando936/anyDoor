import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import base64

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import json
    
    # Setup Chrome options for printing to PDF
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Create driver
    print("Initializing Chrome browser...")
    driver = webdriver.Chrome(options=chrome_options)
    
    # Load the HTML file
    html_path = os.path.abspath("//vmware-host/Shared Folders/claude code screen captures/cutlist.html")
    file_url = f"file:///{html_path.replace(os.sep, '/')}"
    print(f"Loading HTML from: {file_url}")
    driver.get(file_url)
    
    # Wait for page to load
    time.sleep(2)
    
    # Execute print to PDF
    print("Converting to PDF...")
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
        'pageRanges': '',
        'preferCSSPageSize': True,
    })
    
    # Save PDF
    pdf_path = "//vmware-host/Shared Folders/claude code screen captures/cutlist.pdf"
    with open(pdf_path, 'wb') as f:
        f.write(base64.b64decode(pdf_data['data']))
    
    print(f"[OK] Successfully created: cutlist.pdf")
    driver.quit()
    
except Exception as e:
    print(f"[ERROR] {str(e)}")
    print("\n[!] Please open cutlist.html in your browser")
    print("[!] Then press Ctrl+P and save as PDF")