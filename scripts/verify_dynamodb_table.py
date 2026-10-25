
import pymupdf
from resolveops.ingestion.pdf_parser import extract_tables

path = "data/raw/aws/pdf/dynamodb-dg.pdf"

with pymupdf.open(path) as pdf:
    page = pdf[57]
    tables = extract_tables(page)

assert len(tables) >= 1

rows = tables[0]

assert rows[1][0] == "3.3.2"
assert "2.55.15" in rows[1][2]
assert "pom.xml" in rows[2][2]

print("Real PDF table extraction passed!")
print("Rows:", len(rows))
print("Columns:", len(rows[0]))
