"""
HTML to PDF Converter Utility
Converts HTML files to PDFs using Selenium WebDriver
"""

import base64
import sys
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Set encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def html_to_pdf(html_path, pdf_path=None, scale=1.0, landscape=False):
    """
    Convert HTML file to PDF using Selenium
    
    Args:
        html_path: Path to HTML file
        pdf_path: Output PDF path (optional, defaults to same name as HTML)
        scale: Scaling factor (0.5 to 2.0)
        landscape: True for landscape, False for portrait
    """
    
    if not os.path.exists(html_path):
        print(f"[ERROR] HTML file not found: {html_path}")
        return False
    
    if pdf_path is None:
        pdf_path = html_path.replace('.html', '.pdf')
    
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
            'landscape': landscape,
            'displayHeaderFooter': False,
            'printBackground': True,
            'scale': scale,
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

def batch_convert(html_files, output_dir=None):
    """
    Convert multiple HTML files to PDFs
    
    Args:
        html_files: List of HTML file paths
        output_dir: Directory for output PDFs (optional)
    """
    
    successful = 0
    failed = 0
    
    for html_file in html_files:
        if output_dir:
            filename = os.path.basename(html_file).replace('.html', '.pdf')
            pdf_path = os.path.join(output_dir, filename)
        else:
            pdf_path = None
        
        if html_to_pdf(html_file, pdf_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nBatch conversion complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

def convert_order_reports(customer_prefix):
    """
    Convert all reports for a specific customer order
    
    Args:
        customer_prefix: Prefix for the order files (e.g., "paul_revere_302")
    """
    
    reports = ['door_list', 'shop_report', 'cut_list']
    
    print(f"Converting reports for: {customer_prefix}")
    print("=" * 60)
    
    for report in reports:
        html_file = f"{customer_prefix}_{report}.html"
        pdf_file = f"{customer_prefix}_{report}.pdf"
        
        if os.path.exists(html_file):
            print(f"\nConverting {report}...")
            html_to_pdf(html_file, pdf_file)
        else:
            print(f"\n[WARNING] Not found: {html_file}")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert HTML files to PDFs')
    parser.add_argument('input', help='HTML file or customer prefix')
    parser.add_argument('-o', '--output', help='Output PDF path')
    parser.add_argument('-s', '--scale', type=float, default=1.0, 
                        help='Scale factor (0.5-2.0, default: 1.0)')
    parser.add_argument('-l', '--landscape', action='store_true',
                        help='Use landscape orientation')
    parser.add_argument('--order', action='store_true',
                        help='Convert all reports for an order (input is prefix)')
    
    args = parser.parse_args()
    
    if args.order:
        # Convert all reports for an order
        convert_order_reports(args.input)
    else:
        # Convert single file
        if args.input.endswith('.html'):
            html_to_pdf(args.input, args.output, args.scale, args.landscape)
        else:
            print("[ERROR] Input must be an HTML file or use --order flag")
            print("\nExamples:")
            print("  python 3_convert_html_to_pdf.py report.html")
            print("  python 3_convert_html_to_pdf.py report.html -o output.pdf")
            print("  python 3_convert_html_to_pdf.py paul_revere_302 --order")
            print("  python 3_convert_html_to_pdf.py report.html -s 0.9 -l")