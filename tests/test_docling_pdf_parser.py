from types import SimpleNamespace

import pymupdf
import pytest

from resolveops.ingestion.document_parser import parse_document
from resolveops.ingestion.docling_pdf_parser import parse_pdf_with_docling
from resolveops.ingestion.models import BlockType, SourceType
from resolveops.ingestion.pdf_parser import parse_pdf


class FakeConverter:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def convert(self, path, **options):
        self.calls.append((path, options))
        document = SimpleNamespace(iterate_items=lambda: iter(self.items))
        return SimpleNamespace(document=document)


def _item(label, ref, *, text="", page=1, rows=None, level=1):
    bbox = SimpleNamespace(
        l=10.0,
        t=20.0,
        r=100.0,
        b=40.0,
        coord_origin=SimpleNamespace(value="TOPLEFT"),
    )
    provenance = [SimpleNamespace(page_no=page, bbox=bbox)]
    data = (
        SimpleNamespace(
            grid=[[SimpleNamespace(text=cell) for cell in row] for row in rows]
        )
        if rows is not None
        else None
    )
    return SimpleNamespace(
        label=label,
        self_ref=ref,
        text=text,
        level=level,
        prov=provenance,
        data=data,
    )


def test_docling_adapter_preserves_reading_order_and_table_grid(tmp_path):
    path = tmp_path / "guide.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page()
        pdf.save(path)

    converter = FakeConverter(
        [
            (_item("title", "#/texts/0", text="Guide"), 0),
            (_item("paragraph", "#/texts/1", text="Before table"), 1),
            (_item("table", "#/tables/0", rows=[["Name", "Value"], ["A", "1"]]), 1),
            (_item("section_header", "#/texts/2", text="Next", page=2, level=2), 1),
        ]
    )

    document = parse_pdf_with_docling(
        path, page_number=None, source_id="guide-source", converter=converter
    )

    assert document.source_type == SourceType.PDF
    assert document.source == "guide-source"
    assert document.title == "Guide"
    assert document.metadata["parsed_pages"] == [1, 2]
    assert converter.calls == [(path, {})]
    assert [block.type for block in document.blocks] == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.TABLE,
        BlockType.HEADING,
    ]
    assert [block.order for block in document.blocks] == [0, 1, 2, 3]
    assert [block.page_number for block in document.blocks] == [1, 1, 1, 2]
    assert document.blocks[2].table_data == [["Name", "Value"], ["A", "1"]]
    assert document.blocks[2].content == "Name | Value\nA | 1"
    assert document.blocks[3].heading_level == 2
    assert document.blocks[0].metadata["bbox_origin"] == "TOPLEFT"


def test_docling_adapter_preserves_merged_cell_spans(tmp_path):
    path = tmp_path / "merged.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)

    merged_header = SimpleNamespace(
        text="Category",
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=2,
        row_span=1,
        col_span=2,
        column_header=True,
    )
    first_value = SimpleNamespace(
        text="A",
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    second_value = SimpleNamespace(
        text="B",
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=1,
        end_col_offset_idx=2,
    )
    table_item = _item("table", "#/tables/0", rows=[["Category", ""], ["A", "B"]])
    table_item.data = SimpleNamespace(
        num_rows=2,
        num_cols=2,
        table_cells=[merged_header, first_value, second_value],
        grid=[[merged_header, merged_header], [first_value, second_value]],
    )
    document = parse_pdf_with_docling(
        path,
        converter=FakeConverter([(table_item, 0)]),
    )

    table = document.blocks[0]
    assert table.table_data == [["Category", ""], ["A", "B"]]
    assert table.content == "Category | \nA | B"
    assert table.table_cells[0].column_span == 2
    assert table.table_cells[0].column_header is True


def test_docling_adapter_passes_selected_page_to_converter(tmp_path):
    path = tmp_path / "guide.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page()
        pdf.save(path)

    converter = FakeConverter([(_item("paragraph", "#/texts/0", text="Page two", page=2), 0)])
    document = parse_pdf_with_docling(path, page_number=2, converter=converter)

    assert converter.calls == [(path, {"page_range": (2, 2)})]
    assert document.metadata["parsed_pages"] == [2]
    assert document.blocks[0].page_number == 2

    with pytest.raises(ValueError, match="Invalid page number"):
        parse_pdf_with_docling(path, page_number=3, converter=converter)

    ranged = parse_pdf_with_docling(
        path, page_range=(1, 2), converter=FakeConverter([])
    )
    assert ranged.metadata["parsed_pages"] == [1, 2]

    with pytest.raises(ValueError, match="Choose page_number or page_range"):
        parse_pdf_with_docling(
            path, page_number=1, page_range=(1, 2), converter=converter
        )


def test_parse_pdf_can_select_docling_engine(tmp_path, monkeypatch):
    path = tmp_path / "guide.pdf"
    path.write_bytes(b"pdf supplied to the selected engine")
    expected = object()

    def fake_parse(pdf_path, page_number=None, *, page_range=None, source_id=None):
        assert (pdf_path, page_number, source_id) == (path, 1, "guide-source")
        return expected

    monkeypatch.setattr(
        "resolveops.ingestion.docling_pdf_parser.parse_pdf_with_docling", fake_parse
    )

    assert parse_pdf(path, page_number=1, source_id="guide-source", engine="docling") is expected
    assert (
        parse_document(
            path, page_number=1, source_id="guide-source", pdf_engine="docling"
        )
        is expected
    )
