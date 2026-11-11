"""Adapt Docling's PDF output to ResolveOps' document model."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pymupdf

from resolveops.ingestion.models import Block, BlockType, Document, SourceType, TableCell
from resolveops.ingestion.normalization import (
    build_block,
    make_document_id,
    resolve_source,
    table_to_text,
)


def _label(item: Any) -> str:
    label = item.label
    return label.value if hasattr(label, "value") else str(label)


def _cell_bbox(cell: Any) -> tuple[list[float] | None, str | None]:
    bbox = getattr(cell, "bbox", None)
    if bbox is None:
        return None, None
    return (
        [bbox.l, bbox.t, bbox.r, bbox.b],
        (
            bbox.coord_origin.value
            if hasattr(bbox.coord_origin, "value")
            else str(bbox.coord_origin)
        ),
    )


def _table_structure(item: Any) -> tuple[list[list[str]], list[TableCell]]:
    data = item.data
    grid = data.grid
    row_count = getattr(data, "num_rows", len(grid))
    column_count = getattr(data, "num_cols", max((len(row) for row in grid), default=0))
    rows = [["" for _ in range(column_count)] for _ in range(row_count)]
    table_cells: list[TableCell] = []

    # Use source cells instead of the expanded grid where available. Docling's
    # grid repeats a spanning cell in every covered slot; source offsets keep
    # the merge relationship and avoid repeating its text in retrieval output.
    seen: set[tuple[int, int, int, int, str]] = set()
    source_cells = getattr(data, "table_cells", None)
    cells = (
        ((None, None, cell) for cell in source_cells)
        if source_cells
        else (
            (row_index, column_index, cell)
            for row_index, row in enumerate(grid)
            for column_index, cell in enumerate(row)
        )
    )
    for grid_row, grid_column, cell in cells:
        row_index = getattr(cell, "start_row_offset_idx", grid_row or 0)
        column_index = getattr(cell, "start_col_offset_idx", grid_column or 0)
        row_end = getattr(cell, "end_row_offset_idx", row_index + 1)
        column_end = getattr(cell, "end_col_offset_idx", column_index + 1)
        text = str(getattr(cell, "text", "")).strip()
        key = (row_index, column_index, row_end, column_end, text)
        if key in seen:
            continue
        seen.add(key)

        if 0 <= row_index < len(rows) and 0 <= column_index < len(rows[row_index]):
            rows[row_index][column_index] = text

        bbox, bbox_origin = _cell_bbox(cell)
        table_cells.append(
            TableCell(
                row_index=row_index,
                column_index=column_index,
                row_span=max(1, row_end - row_index),
                column_span=max(1, column_end - column_index),
                text=text,
                column_header=bool(getattr(cell, "column_header", False)),
                row_header=bool(getattr(cell, "row_header", False)),
                row_section=bool(getattr(cell, "row_section", False)),
                fillable=bool(getattr(cell, "fillable", False)),
                bbox=bbox,
                bbox_origin=bbox_origin,
            )
        )

    return rows, table_cells

#Bad tables are saved for developers to inspect, but they are quarantined and never shown to the LLM
def _table_review_reasons(rows: list[list[str]], cells: list[TableCell]) -> list[str]:
    """Flag structural failures without guessing at a table's intended content."""
    reasons: list[str] = []
    column_count = len(rows[0]) if rows else 0
    if not rows or not column_count or not any(cell.strip() for row in rows for cell in row):
        reasons.append("empty_table")
    if any(len(row) != column_count for row in rows):
        reasons.append("ragged_rows")
    if any(
        cell.row_index + cell.row_span > len(rows)
        or cell.column_index + cell.column_span > column_count
        for cell in cells
    ):
        reasons.append("cell_outside_grid")
    return reasons


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def parse_pdf_with_docling(
    pdf_path: str | Path,
    page_number: int | None = None,
    *,
    page_range: tuple[int, int] | None = None,
    source_id: str | None = None,
    converter: Any = None,
) -> Document:
    """Convert a PDF with Docling and preserve its item reading order."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    with pymupdf.open(path) as pdf:
        page_count = len(pdf)

    if page_number is not None and page_range is not None:
        raise ValueError("Choose page_number or page_range")
    if page_number is not None and not 1 <= page_number <= page_count:
        raise ValueError("Invalid page number")
    if page_range is not None and not (1 <= page_range[0] <= page_range[1] <= page_count):
        raise ValueError("Invalid page range")

    if converter is None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as error:
            raise ImportError(
                "Docling is required for this PDF engine; install resolveops[docling]"
            ) from error

        pipeline_options = PdfPipelineOptions(do_ocr=False, queue_max_size=1)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    selected_range = page_range or ((page_number, page_number) if page_number else None)
    options = {"page_range": selected_range} if selected_range else {}
    conversion = converter.convert(path, **options)
    status = getattr(conversion, "status", "success")
    status = status.value if hasattr(status, "value") else str(status)
    if status != "success" or conversion.document is None:
        raise RuntimeError(f"Docling conversion did not complete: {status}")
    docling_document = conversion.document
    source = resolve_source(path, source_id)
    document_id = make_document_id(SourceType.PDF, source)
    blocks: list[Block] = []
    title = path.stem
    heading_context: list[str] = []
    docling_version = _package_version("docling")
    docling_core_version = _package_version("docling-core")

    for item, _level in docling_document.iterate_items():
        label = _label(item)
        rows: list[list[str]] | None = None
        table_cells: list[TableCell] | None = None
        if label == "table":
            rows, table_cells = _table_structure(item)

        if label == "table":
            block_type = BlockType.TABLE
            content = table_to_text(rows)
        else:
            block_type = {
                "title": BlockType.HEADING,
                "section_header": BlockType.HEADING,
                "list_item": BlockType.LIST_ITEM,
                "code": BlockType.CODE,
                "paragraph": BlockType.PARAGRAPH,
                "text": BlockType.PARAGRAPH,
                "caption": BlockType.PARAGRAPH,
                "footnote": BlockType.PARAGRAPH,
                "formula": BlockType.PARAGRAPH,
            }.get(label)
            if block_type is None:
                continue
            content = item.text.strip()
            if not content:
                continue

        if label == "title" and title == path.stem:
            title = content
        provenance = item.prov[0] if item.prov else None
        item_page = provenance.page_no if provenance else (
            selected_range[0] if selected_range and selected_range[0] == selected_range[1] else None
        )
        metadata: dict[str, Any] = {}
        if provenance is not None:
            bbox = provenance.bbox
            metadata["bbox"] = [bbox.l, bbox.t, bbox.r, bbox.b]
            metadata["bbox_origin"] = bbox.coord_origin.value
        heading_level = None
        if block_type == BlockType.HEADING:
            heading_level = 1 if label == "title" else min(item.level, 6)
            heading_context = heading_context[: heading_level - 1]
            heading_context.append(content)
        elif block_type == BlockType.TABLE:
            review_reasons = _table_review_reasons(rows, table_cells)
            if item_page is None:
                review_reasons.append("missing_page_provenance")
            metadata["table_extraction"] = {
                "engine": "docling",
                "engine_version": docling_version,
                "core_version": docling_core_version,
                "validation_state": "needs_review" if review_reasons else "passed",
                "review_reasons": review_reasons,
                "heading_context": list(heading_context),
            }

        blocks.append(
            build_block(
                document_id=document_id,
                block_type=block_type,
                content=content,
                order=len(blocks),
                identity=f"page:{item_page}:{item.self_ref}",
                page_number=item_page,
                heading_level=heading_level,
                table_data=rows,
                table_cells=table_cells,
                metadata=metadata,
            )
        )

    selected_pages = (
        list(range(selected_range[0], selected_range[1] + 1))
        if selected_range else list(range(1, page_count + 1))
    )
    return Document(
        id=document_id,
        source=source,
        source_type=SourceType.PDF,
        title=title,
        blocks=blocks,
        metadata={
            "file_path": str(path),
            "page_count": page_count,
            "parsed_pages": selected_pages,
            "pdf_engine": "docling",
            "pdf_engine_version": docling_version,
            "docling_core_version": docling_core_version,
        },
    )
