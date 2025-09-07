#!/usr/bin/env python3
"""
Cabinet Door Order Processor
Processes order forms and generates shop reports and cut lists
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Tuple
import re

# For PDF processing
try:
    import PyPDF2
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("[!] PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("[!] pdfplumber not installed. Install with: pip install pdfplumber")

# For HTML to PDF conversion
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    print("[!] Selenium not installed. Install with: pip install selenium")


class OrderData:
    """Container for order form data"""
    def __init__(self):
        self.date = ""
        self.job_name = ""
        self.customer_name = ""
        self.address = ""
        self.phone = ""
        self.email = ""
        self.wood_species = ""
        self.door_number = ""
        self.outside_edge = ""
        self.sticky = ""
        self.panel_cut = ""
        self.overlay_type = ""
        self.bore_prep = False
        self.hinge_type = ""
        self.opening_sizes = True  # vs finish sizes
        self.drawer_type = ""
        self.line_items = []  # List of dicts with qty, width, height, type, notes

    def to_dict(self):
        return self.__dict__


class OrderProcessor:
    """Main processor for orders"""
    
    def __init__(self):
        self.stile_width = 2.375  # 2 3/8" in decimal
        self.rail_deduction = 4.5  # 2.25 * 2
        self.rail_addition = 0.75  # for sticking
        self.stile_addition = 0.25  # 1/4" for stiles
        
    def extract_from_pdf_pure_python(self, pdf_path: str) -> OrderData:
        """
        Attempt to extract data using pure Python PDF libraries.
        NOTE: This often fails with complex PDFs or form fields.
        """
        order = OrderData()
        
        if HAS_PDFPLUMBER:
            # pdfplumber is better for tables
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[0]
                text = page.extract_text()
                
                # Try to extract fields using regex patterns
                patterns = {
                    'date': r'Date:\s*([0-9/]+)',
                    'job_name': r'Job Name:\s*([^\n]+)',
                    'customer_name': r'Name:\s*([^\n]+)',
                    'address': r'Address:\s*([^\n]+)',
                    'phone': r'Phone #:\s*([^\n]+)',
                    'email': r'Email:\s*([^\n]+)',
                    'wood_species': r'Wood\s+Species:\s*([^\n]+)',
                    'door_number': r'Door #:\s*([0-9]+)',
                }
                
                for field, pattern in patterns.items():
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        setattr(order, field, match.group(1).strip())
                
                # Extract table data
                tables = page.extract_tables()
                if tables and len(tables) > 0:
                    # Look for the main order table
                    for table in tables:
                        for row in table:
                            if row and len(row) >= 5:
                                # Check if this looks like a data row
                                if row[1] and row[2] and row[3]:
                                    try:
                                        qty = int(row[1]) if row[1].isdigit() else 0
                                        if qty > 0:
                                            order.line_items.append({
                                                'qty': qty,
                                                'width': row[2],
                                                'height': row[3],
                                                'type': row[4],
                                                'notes': row[5] if len(row) > 5 else ''
                                            })
                                    except:
                                        pass
        
        elif HAS_PYPDF:
            # Fallback to PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page = reader.pages[0]
                text = page.extract_text()
                
                # Similar regex extraction as above
                # (simplified for brevity)
                
        return order
    
    def parse_order_manually(self, order_dict: dict) -> OrderData:
        """
        Manually parse order data from a dictionary.
        This is what you'd use after AI/model extraction.
        """
        order = OrderData()
        for key, value in order_dict.items():
            if hasattr(order, key):
                setattr(order, key, value)
        return order
    
    def calculate_stiles(self, width: float, height: float, qty: int) -> dict:
        """Calculate stile requirements"""
        stile_length = height + self.stile_addition
        total_length = stile_length * 2 * qty  # 2 stiles per door/drawer
        pieces_8ft = int((total_length + 95) / 96)  # Round up to nearest 8' piece
        
        return {
            'length': stile_length,
            'qty': qty * 2,
            'total_inches': total_length,
            'pieces_8ft': pieces_8ft
        }
    
    def calculate_rails(self, width: float, height: float, qty: int) -> dict:
        """Calculate rail requirements"""
        rail_length = width - self.rail_deduction + self.rail_addition
        total_length = rail_length * 2 * qty  # 2 rails per door/drawer
        pieces_8ft = int((total_length + 95) / 96)  # Round up to nearest 8' piece
        
        return {
            'length': rail_length,
            'qty': qty * 2,
            'total_inches': total_length,
            'pieces_8ft': pieces_8ft
        }
    
    def generate_shop_report_html(self, order: OrderData) -> str:
        """Generate shop report HTML"""
        
        # Calculate totals
        total_doors = sum(item['qty'] for item in order.line_items if 'door' in item['type'].lower())
        total_drawers = sum(item['qty'] for item in order.line_items if 'drawer' in item['type'].lower())
        total_items = total_doors + total_drawers
        total_hinges = total_doors * 2
        
        # Calculate total stiles needed
        total_stile_inches = 0
        for item in order.line_items:
            width = float(item['width'])
            height = float(item['height'])
            qty = int(item['qty'])
            total_stile_inches += (width + height) * 2 * qty
        
        total_8ft_pieces = int((total_stile_inches + 95) / 96)
        total_feet = int(total_stile_inches / 12)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Shop Report - {order.job_name}</title>
    <style>
        /* Styles from our template */
        body {{ font-family: Arial, sans-serif; font-size: 12pt; }}
        /* ... (include all styles) ... */
    </style>
</head>
<body>
    <h1>SHOP REPORT Order Info:</h1>
    
    <table class="header-table">
        <tr>
            <td>Date: {order.date}</td>
            <td>Order #: _______</td>
        </tr>
        <tr>
            <td>Customer Name: {order.customer_name.upper()}</td>
            <td>Address: {order.address.upper()}</td>
        </tr>
    </table>
    
    <table class="info-table">
        <tr>
            <td>Wood Species: {order.wood_species.upper()}</td>
            <td>Outside: {order.outside_edge.upper()}</td>
        </tr>
        <tr>
            <td>Sticky: {order.sticky.upper()}</td>
            <td>Panel Cut: {order.panel_cut}</td>
        </tr>
        <tr>
            <td>TOTAL DOORS & DRAWERS: {total_items}</td>
            <td>TOTAL HINGES: {total_hinges}</td>
        </tr>
    </table>
    
    <table class="items-table">
        <tr>
            <th>QTY</th>
            <th>WIDTH</th>
            <th>PART NAME</th>
            <th>MATERIAL</th>
            <th>NOTES</th>
        </tr>
        <tr>
            <td>{total_8ft_pieces}</td>
            <td>2 3/8" × 8'</td>
            <td>STILES</td>
            <td>{order.wood_species.upper()}</td>
            <td>All doors & drawers ({total_feet} ft total)</td>
        </tr>
    </table>
</body>
</html>"""
        
        return html
    
    def generate_cut_list_html(self, order: OrderData) -> str:
        """Generate cut list HTML"""
        
        cuts = []
        
        # Process each line item
        for idx, item in enumerate(order.line_items, 1):
            width = float(item['width'])
            height = float(item['height'])
            qty = int(item['qty'])
            item_type = item['type']
            
            # Calculate stiles
            stile_length = height + self.stile_addition
            cuts.append({
                'length': stile_length,
                'qty': qty * 2,
                'part': 'STILES',
                'label': f'#{idx}',
                'description': f'{qty} {item_type} {width}×{height}'
            })
            
            # Calculate rails
            rail_length = width - self.rail_deduction + self.rail_addition
            cuts.append({
                'length': rail_length,
                'qty': qty * 2,
                'part': 'RAILS',
                'label': f'#{idx}',
                'description': f'{qty} {item_type} {width}×{height}'
            })
        
        # Sort by length descending
        cuts.sort(key=lambda x: x['length'], reverse=True)
        
        # Generate HTML
        rows = ""
        for cut in cuts:
            length_frac = self.decimal_to_fraction(cut['length'])
            rows += f"""
        <tr>
            <td>{cut['qty']}</td>
            <td>{length_frac}"</td>
            <td>2 3/8"</td>
            <td>{cut['part']}</td>
            <td>{cut['label']}</td>
            <td>{cut['description']}</td>
            <td>☐</td>
        </tr>"""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Cut List - {order.job_name}</title>
    <style>
        /* Compact styles for single page */
        body {{ font-size: 10pt; }}
        /* ... (include all styles) ... */
    </style>
</head>
<body>
    <h1>CUT LIST</h1>
    
    <div class="header-info">
        <p>Date: {order.date} | Job: {order.job_name} | Customer: {order.customer_name}</p>
    </div>
    
    <table class="cut-table">
        <thead>
            <tr>
                <th>QTY</th>
                <th>LENGTH</th>
                <th>WIDTH</th>
                <th>PART</th>
                <th>LABEL</th>
                <th>DESCRIPTION</th>
                <th>✓</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>"""
        
        return html
    
    def decimal_to_fraction(self, decimal: float) -> str:
        """Convert decimal inches to fraction string"""
        whole = int(decimal)
        fraction = decimal - whole
        
        if fraction < 0.125:
            return str(whole)
        elif fraction < 0.375:
            return f"{whole} 1/4"
        elif fraction < 0.625:
            return f"{whole} 1/2"
        elif fraction < 0.875:
            return f"{whole} 3/4"
        else:
            return str(whole + 1)
    
    def html_to_pdf(self, html_content: str, output_pdf: str):
        """Convert HTML to PDF using Selenium"""
        if not HAS_SELENIUM:
            print("[ERROR] Selenium not available")
            return False
        
        # Save HTML to temp file
        temp_html = "temp_report.html"
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(f"file:///{os.path.abspath(temp_html)}")
            
            pdf_data = driver.execute_cdp_cmd('Page.printToPDF', {
                'landscape': False,
                'displayHeaderFooter': False,
                'printBackground': True,
                'scale': 1.0,
                'paperWidth': 8.5,
                'paperHeight': 11,
            })
            
            with open(output_pdf, 'wb') as f:
                f.write(base64.b64decode(pdf_data['data']))
            
            driver.quit()
            os.remove(temp_html)
            return True
            
        except Exception as e:
            print(f"[ERROR] PDF conversion failed: {e}")
            return False


# EXAMPLE USAGE - What you'd get from AI/Model extraction
def example_extracted_data():
    """
    This is what an AI model would extract from the PDF.
    In pure Python, this extraction is very difficult and unreliable.
    """
    return {
        'date': '1/7/2025',
        'job_name': 'wiggins job',
        'customer_name': 'kevin fox',
        'address': '12342 pander lane',
        'phone': '9367882762',
        'email': 'kevin@gmail.com',
        'wood_species': 'White Oak',
        'door_number': '308',
        'outside_edge': 'Special',
        'sticky': 'Bevel',
        'panel_cut': '3/8" Plywood (Flat Panel ONLY)',
        'overlay_type': '1/2" Overlay',
        'bore_prep': True,
        'hinge_type': 'Blum Soft Close Frameless 1/2"OL',
        'opening_sizes': True,
        'drawer_type': '5 piece',
        'line_items': [
            {'qty': 5, 'width': 12, 'height': 36, 'type': 'doors', 'notes': ''},
            {'qty': 4, 'width': 36, 'height': 6, 'type': 'drawer', 'notes': ''},
            {'qty': 6, 'width': 15, 'height': 45, 'type': 'doors', 'notes': ''},
            {'qty': 3, 'width': 14, 'height': 6, 'type': 'drawer', 'notes': ''},
        ]
    }


def main():
    """Main execution"""
    processor = OrderProcessor()
    
    print("Cabinet Door Order Processor")
    print("-" * 40)
    
    # Option 1: Try pure Python extraction (usually fails on complex PDFs)
    if os.path.exists("user_submited_ORDER_FORM_.pdf"):
        print("Attempting to extract from PDF...")
        order = processor.extract_from_pdf_pure_python("user_submited_ORDER_FORM_.pdf")
        
        if not order.line_items:
            print("[!] Pure Python extraction failed or incomplete")
            print("[!] Complex form PDFs usually require AI/OCR for reliable extraction")
            print()
    
    # Option 2: Use pre-extracted data (from AI model or manual entry)
    print("Using example extracted data (as if from AI model)...")
    extracted_data = example_extracted_data()
    order = processor.parse_order_manually(extracted_data)
    
    # Generate reports
    print(f"Processing order for: {order.customer_name}")
    print(f"Job: {order.job_name}")
    print(f"Items: {len(order.line_items)} line items")
    
    # Generate Shop Report
    shop_html = processor.generate_shop_report_html(order)
    with open("shop_report.html", 'w', encoding='utf-8') as f:
        f.write(shop_html)
    print("[OK] Generated shop_report.html")
    
    if processor.html_to_pdf(shop_html, "shop_report.pdf"):
        print("[OK] Generated shop_report.pdf")
    
    # Generate Cut List
    cut_html = processor.generate_cut_list_html(order)
    with open("cut_list.html", 'w', encoding='utf-8') as f:
        f.write(cut_html)
    print("[OK] Generated cut_list.html")
    
    if processor.html_to_pdf(cut_html, "cut_list.pdf"):
        print("[OK] Generated cut_list.pdf")
    
    print("\nProcess complete!")
    
    # Show what pure Python CAN do reliably:
    print("\n" + "=" * 40)
    print("PURE PYTHON CAPABILITIES:")
    print("=" * 40)
    print("[OK] Calculate stiles and rails from dimensions")
    print("[OK] Generate HTML reports")
    print("[OK] Convert HTML to PDF")
    print("[OK] Sort and organize cut lists")
    print("[OK] Calculate material requirements")
    print()
    print("REQUIRES AI/MODEL:")
    print("-" * 40)
    print("[X] Extract text from complex PDF forms")
    print("[X] Identify form fields reliably")
    print("[X] Handle varying PDF formats")
    print("[X] Extract table data from scanned PDFs")
    print()
    print("RECOMMENDATION:")
    print("Use AI model for PDF extraction, then Python for everything else")


if __name__ == "__main__":
    main()