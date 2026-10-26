
import pymupdf

from resolveops.ingestion.models import BlockType
from resolveops.ingestion.pdf_parser import parse_pdf


def test_pdf_document_preserves_table_structure(tmp_path):
    path = tmp_path / "table.pdf"
    expected = [
        ["Version", "Change", "SDK"],
        ["3.3.2", "Updated", "2.55.15"],
        ["3.3.1", "PartiQL", "pom.xml"],
    ]

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        x, y = 72, 100
        width, height = 120, 35

        for index in range(4):
            page.draw_line(
                (x + index * width, y),
                (x + index * width, y + 3 * height),
            )
            page.draw_line(
                (x, y + index * height),
                (x + 3 * width, y + index * height),
            )

        for row_index, row in enumerate(expected):
            for column_index, value in enumerate(row):
                page.insert_text(
                    (
                        x + column_index * width + 5,
                        y + row_index * height + 22,
                    ),
                    value,
                    fontsize=10,
                )

        pdf.save(path)

    document = parse_pdf(path)
    table_blocks = [
        block
        for block in document.blocks
        if block.type == BlockType.TABLE
    ]

    assert len(table_blocks) == 1
    assert table_blocks[0].table_data == expected
    assert "3.3.2 | Updated | 2.55.15" in table_blocks[0].content
