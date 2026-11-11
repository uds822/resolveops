import json
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolveops.ingestion.docling_pdf_parser import parse_pdf_with_docling
from resolveops.ingestion.models import BlockType

pdf_path = "data/raw/aws/pdf/dynamodb-dg.pdf"
page = 58

print(f"\n🚀 Running Docling Parser on {pdf_path} (Page {page})...\n")

document = parse_pdf_with_docling(pdf_path, page_number=page)

# Find table blocks
tables = [block for block in document.blocks if block.type == BlockType.TABLE]

print(f"✅ Found {len(tables)} table(s) on Page {page}!\n")

for i, table in enumerate(tables, 1):
    print("=" * 60)
    print(f"📋 TABLE #{i} — MARKDOWN CONTENT (for LLM & Search):")
    print("=" * 60)
    print(table.content)
    
    print("\n" + "=" * 60)
    print("🔍 METADATA ATTACHED TO THIS TABLE:")
    print("=" * 60)
    print(json.dumps(table.metadata, indent=2))
    
    print("\n" + "=" * 60)
    print("🧩 FIRST 3 CELL STRUCTURES (Retaining row/column spans):")
    print("=" * 60)
    if table.table_cells:
        for cell in table.table_cells[:3]:
            print(f" - Cell [{cell.row_index}, {cell.column_index}]: '{cell.text}' "
                  f"(header={cell.column_header}, span=({cell.row_span}x{cell.column_span}))")
    print("=" * 60 + "\n")

