
import pymupdf
import pytest

from resolveops.ingestion.document_parser import parse_document
from resolveops.ingestion.models import BlockType, SourceType


def test_markdown_dispatch(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("# AWS S3\n\nVersioning guide.")

    result = parse_document(path)

    assert result.source_type == SourceType.MARKDOWN
    assert result.title == "AWS S3"
    assert len(result.blocks) == 2


def test_html_dispatch(tmp_path):
    path = tmp_path / "sample.html"

    path.write_text(
        "<html><head><title>AWS</title></head>"
        "<body><main><p>Access denied</p></main></body></html>"
    )

    result = parse_document(path)

    assert result.source_type == SourceType.HTML
    assert result.blocks[0].content == "Access denied"


def test_pdf_dispatch(tmp_path):
    path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "DynamoDB guide")
        pdf.save(path)

    result = parse_document(path, page_number=1)

    assert result.source_type == SourceType.PDF
    assert result.blocks[0].type == BlockType.PAGE_TEXT
    assert "DynamoDB guide" in result.blocks[0].content
    assert result.blocks[0].page_number == 1


def test_pdf_dispatch_parses_all_pages_by_default(tmp_path):
    path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.insert_text((72, 72), "First page")
        second = pdf.new_page()
        second.insert_text((72, 72), "Second page")
        pdf.save(path)

    result = parse_document(path)

    assert result.metadata["parsed_pages"] == [1, 2]
    assert [block.page_number for block in result.blocks] == [1, 2]


def test_unsupported_format(tmp_path):
    path = tmp_path / "sample.xyz"

    with pytest.raises(ValueError):
        parse_document(path)
