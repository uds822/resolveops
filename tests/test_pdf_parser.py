
import pymupdf
import pytest

from resolveops.ingestion.models import BlockType, SourceType
from resolveops.ingestion.pdf_parser import extract_pdf_page, parse_pdf


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
