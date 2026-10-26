

from pathlib import Path

import pymupdf

from resolveops.ingestion.models import BlockType, Document, SourceType
from resolveops.ingestion.normalization import (
    build_block,
    make_document_id,
    resolve_source,
    table_to_text,
)


def _validate_page_number(page_number: int, page_count: int) -> None:
    if page_number < 1 or page_number > page_count:
        raise ValueError("Invalid page number")


def extract_pdf_page(pdf_path: str | Path, page_number: int) -> dict:
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    with pymupdf.open(path) as pdf:
        _validate_page_number(page_number, len(pdf))

        page = pdf[page_number - 1]
        text = page.get_text("text")

    return {
        "source": str(path),
        "page_number": page_number,
        "block_type": "text",
        "content": text,
    }


def _extract_table_entries(page) -> list[tuple[pymupdf.Rect, list[list[str]]]]:
    result: list[tuple[pymupdf.Rect, list[list[str]]]] = []

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

        if rows:
            result.append((pymupdf.Rect(table.bbox), rows))

    return result


def extract_tables(page) -> list[list[list[str]]]:
    return [rows for _, rows in _extract_table_entries(page)]


def _bbox_identity(page_number: int, rect: pymupdf.Rect) -> str:
    coordinates = ":".join(f"{value:.3f}" for value in rect)
    return f"page:{page_number}:bbox:{coordinates}"


def parse_pdf(
    pdf_path: str | Path,
    page_number: int | None = None,
    *,
    source_id: str | None = None,
) -> Document:
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    source = resolve_source(path, source_id)
    document_id = make_document_id(SourceType.PDF, source)
    blocks = []

    with pymupdf.open(path) as pdf:
        page_count = len(pdf)

        if page_number is not None:
            _validate_page_number(page_number, page_count)
            selected_pages = [page_number]
        else:
            selected_pages = list(range(1, page_count + 1))

        for current_page_number in selected_pages:
            page = pdf[current_page_number - 1]
            table_entries = _extract_table_entries(page)
            ordered_entries = []

            for raw_block in page.get_text("blocks", sort=True):
                rect = pymupdf.Rect(raw_block[:4])
                text = raw_block[4].strip()

                if not text:
                    continue

                if any(
                    rect.intersects(table_rect)
                    for table_rect, _ in table_entries
                ):
                    continue

                ordered_entries.append(
                    (rect.y0, rect.x0, "text", rect, text, None)
                )

            for table_rect, rows in table_entries:
                ordered_entries.append(
                    (
                        table_rect.y0,
                        table_rect.x0,
                        "table",
                        table_rect,
                        table_to_text(rows),
                        rows,
                    )
                )

            ordered_entries.sort(key=lambda entry: (entry[0], entry[1]))

            for _, _, kind, rect, text, rows in ordered_entries:
                block_type = (
                    BlockType.TABLE
                    if kind == "table"
                    else BlockType.PAGE_TEXT
                )
                blocks.append(
                    build_block(
                        document_id=document_id,
                        block_type=block_type,
                        content=text,
                        order=len(blocks),
                        identity=_bbox_identity(current_page_number, rect),
                        page_number=current_page_number,
                        table_data=rows,
                        metadata={"bbox": list(rect)},
                    )
                )

        metadata = {
            "file_path": str(path),
            "page_count": page_count,
            "parsed_pages": selected_pages,
        }

    return Document(
        id=document_id,
        source=source,
        source_type=SourceType.PDF,
        title=path.stem,
        blocks=blocks,
        metadata=metadata,
    )
