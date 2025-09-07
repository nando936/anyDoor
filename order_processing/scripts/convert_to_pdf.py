import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    from weasyprint import HTML, CSS
    use_weasyprint = True
except ImportError:
    use_weasyprint = False
    try:
        import pdfkit
        use_pdfkit = True
    except ImportError:
        use_pdfkit = False

def convert_html_to_pdf(html_file, pdf_file):
    if use_weasyprint:
        print(f"Converting {html_file} to {pdf_file} using WeasyPrint...")
        HTML(filename=html_file).write_pdf(pdf_file)
        print(f"[DONE] Created {pdf_file}")
    elif use_pdfkit:
        print(f"Converting {html_file} to {pdf_file} using pdfkit...")
        pdfkit.from_file(html_file, pdf_file)
        print(f"[DONE] Created {pdf_file}")
    else:
        print("[ERROR] No PDF conversion library available.")
        print("Please install one of the following:")
        print("  pip install weasyprint")
        print("  pip install pdfkit (also requires wkhtmltopdf)")
        return False
    return True

if __name__ == "__main__":
    html_path = "//vmware-host/Shared Folders/claude code screen captures/shopreport_empty.html"
    pdf_path = "//vmware-host/Shared Folders/claude code screen captures/shopreportempty.pdf"
    
    if convert_html_to_pdf(html_path, pdf_path):
        print(f"\n[OK] Successfully created empty shop report: shopreportempty.pdf")
    else:
        print("\n[X] Failed to create PDF. Using HTML version instead.")
        print(f"HTML version saved as: shopreport_empty.html")