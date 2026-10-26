
import pymupdf
import pytest

from resolveops.ingestion.models import BlockType, SourceType
from resolveops.ingestion.pdf_parser import (
    _text_fragments_outside_tables,
    extract_pdf_page,
    parse_pdf,
)


def test_pdf_extraction(tmp_path):
    path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "DynamoDB version 3.3.2")
        pdf.save(path)

    result = extract_pdf_page(path, 1)

    assert "DynamoDB version 3.3.2" in result["content"]
    assert result["page_number"] == 1
    assert result["block_type"] == "text"


def test_invalid_page(tmp_path):
    path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)

    with pytest.raises(ValueError):
        extract_pdf_page(path, 100)


def test_pdf_returns_canonical_document(tmp_path):
    path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "DynamoDB guide")
        pdf.save(path)

    result = parse_pdf(path, page_number=1)

    assert result.source_type == SourceType.PDF
    assert result.title == "sample"
    assert result.metadata["page_count"] == 1
    assert result.blocks[0].type == BlockType.PAGE_TEXT
    assert result.blocks[0].page_number == 1
    assert "DynamoDB guide" in result.blocks[0].content


def test_parse_pdf_rejects_invalid_page(tmp_path):
    path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)

    with pytest.raises(ValueError, match="Invalid page number"):
        parse_pdf(path, page_number=2)


def test_mixed_block_splits_text_around_table():
    raw_block = (
        0.0,
        0.0,
        100.0,
        60.0,
        "Before table\nCell value\nAfter table\n",
        4,
        0,
    )
    words = [
        (5.0, 5.0, 45.0, 15.0, "Before", 4, 0, 0),
        (50.0, 5.0, 85.0, 15.0, "table", 4, 0, 1),
        (5.0, 25.0, 35.0, 35.0, "Cell", 4, 1, 0),
        (40.0, 25.0, 75.0, 35.0, "value", 4, 1, 1),
        (5.0, 45.0, 35.0, 55.0, "After", 4, 2, 0),
        (40.0, 45.0, 75.0, 55.0, "table", 4, 2, 1),
    ]
    table_rects = [pymupdf.Rect(0.0, 20.0, 100.0, 40.0)]

    fragments = _text_fragments_outside_tables(
        raw_block,
        words,
        table_rects,
    )

    assert [text for _, text in fragments] == [
        "Before table",
        "After table",
    ]
    assert list(fragments[0][0]) == [5.0, 5.0, 85.0, 15.0]
    assert list(fragments[1][0]) == [5.0, 45.0, 75.0, 55.0]


def test_real_pdf_preserves_order_around_table(tmp_path):
    path = tmp_path / "mixed-content.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 70, 312, 160),
            "Before table\n"
            "HeaderA          HeaderB\n"
            "ValueA           ValueB\n"
            "After table",
            fontsize=10,
        )

        for x_position in (72, 125, 192):
            page.draw_line((x_position, 84), (x_position, 114))

        for y_position in (84, 99.2, 114):
            page.draw_line((72, y_position), (192, y_position))

        pdf.save(path)

    with pymupdf.open(path) as pdf:
        raw_blocks = pdf[0].get_text("blocks", sort=True)

    assert len(raw_blocks) == 1
    assert "Before table" in raw_blocks[0][4]
    assert "After table" in raw_blocks[0][4]

    document = parse_pdf(path)

    assert [block.type for block in document.blocks] == [
        BlockType.PAGE_TEXT,
        BlockType.TABLE,
        BlockType.PAGE_TEXT,
    ]
    assert document.blocks[0].content == "Before table"
    assert document.blocks[1].table_data == [
        ["HeaderA", "HeaderB"],
        ["ValueA", "ValueB"],
    ]
    assert document.blocks[2].content == "After table"

    before_bbox = document.blocks[0].metadata["bbox"]
    table_bbox = document.blocks[1].metadata["bbox"]
    after_bbox = document.blocks[2].metadata["bbox"]

    assert before_bbox[3] <= table_bbox[1]
    assert table_bbox[3] <= after_bbox[1]
