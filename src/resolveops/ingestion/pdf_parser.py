

from pathlib import Path
import pymupdf


def extract_pdf_page(pdf_path, page_number):
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    with pymupdf.open(path) as pdf:
        if page_number < 1 or page_number > len(pdf):
            raise ValueError("Invalid page number")

        page = pdf[page_number - 1]
        text = page.get_text("text")

    return {
        "source": str(path),
        "page_number": page_number,
        "block_type": "text",
        "content": text
    }


def extract_tables(page):
    result = []

    tables = page.find_tables(use_layout=False)

    for table in tables.tables:
        rows = []

        for row in table.rows:
            values = []

            for cell in row.cells:
                if cell is None:
                    values.append("")
                else:
                    text = page.get_text(
                        "text",
                        clip=pymupdf.Rect(cell)
                    )
                    values.append(text.strip())

            rows.append(values)

        result.append(rows)

    return result
