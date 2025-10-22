#!/usr/bin/env python3
"""Test Docling's schema extraction for measurements"""

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from pydantic import BaseModel, Field
from typing import List, Optional

class Measurement(BaseModel):
    """A single measurement value"""
    value: str = Field(description="The measurement value (e.g., '18 1/16', '6 3/8')")
    x_position: Optional[float] = Field(description="X coordinate of measurement", default=None)
    y_position: Optional[float] = Field(description="Y coordinate of measurement", default=None)

class CabinetMeasurements(BaseModel):
    """All measurements found on a cabinet drawing"""
    room_name: Optional[str] = Field(description="Name of the room (e.g., 'Bath #4')", default=None)
    overlay: Optional[str] = Field(description="Overlay specification (e.g., '11/16 OL')", default=None)
    measurements: List[Measurement] = Field(description="All measurements found on the drawing")

# Initialize converter
converter = DocumentConverter(
    allowed_formats=[InputFormat.IMAGE]
)

# Path to image
image_path = "/home/nando/onedrive/customers/raised-panel/Measures-2025-09-09(14-18)/all_pages/page_8.png"

print("Converting document...")
result = converter.convert(image_path)

print("\nDocument converted. Attempting schema extraction...")

# Try to extract with schema
try:
    from docling.document_extractor import DocumentExtractor

    extractor = DocumentExtractor(
        allowed_formats=[InputFormat.IMAGE]
    )

    # Extract using Pydantic model
    extracted = extractor.extract(image_path, template=CabinetMeasurements)

    print("\nExtracted data:")
    print(extracted)

except ImportError as e:
    print(f"\nDocumentExtractor not available: {e}")
    print("Schema extraction may require additional installation or different API")

except Exception as e:
    print(f"\nError during extraction: {e}")
    print(f"Type: {type(e)}")

print("\nBasic OCR extraction from DocumentConverter:")
doc = result.document
print(f"Found {len(doc.texts)} text items")
for i, text in enumerate(doc.texts[:13]):  # Show first 13
    bbox = text.prov[0].bbox if text.prov else None
    if bbox:
        print(f"{i+1}. '{text.text}' at ({int(bbox.l)}, {int(bbox.t)})")
    else:
        print(f"{i+1}. '{text.text}'")
