
import pymupdf

from resolveops.ingestion.pdf_parser import extract_tables


def test_table_extraction(tmp_path):
    path = tmp_path / "table.pdf"

    data = [
        ["Version", "Change", "SDK"],
        ["3.3.2", "Updated", "2.55.15"],
        ["3.3.1", "PartiQL", "pom.xml"]
    ]

    with pymupdf.open() as pdf:
        page = pdf.new_page()

        x, y = 72, 100
        w, h = 120, 35

        # Draw table borders
        for i in range(4):
            page.draw_line(
                (x + i * w, y),
                (x + i * w, y + 3 * h)
            )

            page.draw_line(
                (x, y + i * h),
                (x + 3 * w, y + i * h)
            )

        # Add text inside cells
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                page.insert_text(
                    (x + j * w + 5, y + i * h + 22),
                    value,
                    fontsize=10
                )

        pdf.save(path)

    with pymupdf.open(path) as pdf:
        tables = extract_tables(pdf[0])

    assert len(tables) == 1
    assert tables[0] == data


def test_page_without_tables():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Only paragraph text")

        tables = extract_tables(page)

    assert tables == []
