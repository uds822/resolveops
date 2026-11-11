
from pathlib import Path
import pymupdf
import json
import time

ROOT = Path(__file__).resolve().parents[1]
pdf_path = ROOT / "data/raw/aws/pdf/dynamodb-dg.pdf"
out = ROOT / "data/profiling/tables"


def main():
    out.mkdir(parents=True, exist_ok=True)
    num = int(input("Enter PDF page number: "))

    with pymupdf.open(str(pdf_path)) as pdf:
        if num < 1 or num > pdf.page_count:
            raise SystemExit("Invalid page number")

        page = pdf.load_page(num - 1)
        if page is None:
            raise RuntimeError(f"Could not load page {num} from {pdf_path}")

        start = time.perf_counter()

        text = page.get_text("text")
        tables = page.find_tables()

        if not tables.tables:
            print("No tables detected.")
            return

        table = tables.tables[0]

        print("Original table extraction:")
        print(table.extract()[1][0])

        print("\nDirect cell extraction:")
        cell = table.rows[1].cells[0]

        if cell:
            cell_text = page.get_text("text", clip=pymupdf.Rect(cell))
            print(repr(cell_text))

        duration = time.perf_counter() - start
        rows = [detected_table.extract() for detected_table in tables.tables]

        result = []
        for row in table.rows:
            values = []
            for cell in row.cells:
                if cell is None:
                    values.append("")
                else:
                    cell_text = page.get_text(
                        "text",
                        clip=pymupdf.Rect(cell),
                    )
                    values.append(cell_text.strip())
            result.append(values)

        (out / "normal_text.txt").write_text(text, encoding="utf-8")
        (out / "tables.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        direct_cells_path = out / "direct_cells.json"
        direct_cells_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Tables detected:", len(rows))
        print("Extraction time:", round(duration, 3), "seconds")
        if result:
            print("Header:", result[0])
        if len(result) > 1:
            print("First data row:", result[1])
        print("Saved:", direct_cells_path)
        print("Results saved to:", out)


if __name__ == "__main__":
    main()
