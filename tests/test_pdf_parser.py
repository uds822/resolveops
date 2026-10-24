
import pymupdf
import pytest

from resolveops.ingestion.pdf_parser import extract_pdf_page


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
