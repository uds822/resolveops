

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

def _word_center_is_in_table(
    word,
    table_rects: list[pymupdf.Rect],
) -> bool:
    center_x = (word[0] + word[2]) / 2
    center_y = (word[1] + word[3]) / 2

    return any(
        table_rect.x0 <= center_x <= table_rect.x1
        and table_rect.y0 <= center_y <= table_rect.y1
        for table_rect in table_rects
    )


def _fragment_from_words(words) -> tuple[pymupdf.Rect, str]:
    rect = pymupdf.Rect(
        min(word[0] for word in words),
        min(word[1] for word in words),
        max(word[2] for word in words),
        max(word[3] for word in words),
    )
    text = " ".join(word[4] for word in words)

    return rect, text


def _text_fragments_outside_tables(
    raw_block,
    page_words,
    table_rects: list[pymupdf.Rect],
) -> list[tuple[pymupdf.Rect, str]]:
    original_text = raw_block[4].strip()
    block_rect = pymupdf.Rect(raw_block[:4])

    if not any(block_rect.intersects(table_rect) for table_rect in table_rects):
        return [(block_rect, original_text)] if original_text else []

    if len(raw_block) <= 5:
        return [(block_rect, original_text)] if original_text else []

    block_number = raw_block[5]
    block_words = [word for word in page_words if word[5] == block_number]

    if not block_words:
        return [(block_rect, original_text)] if original_text else []

    has_table_words = any(
        _word_center_is_in_table(word, table_rects)
        for word in block_words
    )

    if not has_table_words:
        return [(block_rect, original_text)] if original_text else []

    lines = {}
    for word in block_words:
        line_key = (word[5], word[6])
        lines.setdefault(line_key, []).append(word)

    ordered_lines = sorted(
        lines.values(),
        key=lambda line: (
            min(word[1] for word in line),
            min(word[0] for word in line),
        ),
    )
    fragments: list[tuple[pymupdf.Rect, str]] = []

    for line in ordered_lines:
        current_words = []

        for word in sorted(line, key=lambda item: item[7]):
            if _word_center_is_in_table(word, table_rects):
                if current_words:
                    fragments.append(_fragment_from_words(current_words))
                    current_words = []
            else:
                current_words.append(word)

        if current_words:
            fragments.append(_fragment_from_words(current_words))

    return fragments


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
            table_rects = [table_rect for table_rect, _ in table_entries]
            page_words = page.get_text("words", sort=True)
            ordered_entries = []

            for raw_block in page.get_text("blocks", sort=True):
                text_fragments = _text_fragments_outside_tables(
                    raw_block,
                    page_words,
                    table_rects,
                )

                for fragment_rect, text in text_fragments:
                    ordered_entries.append(
                        (
                            fragment_rect.y0,
                            fragment_rect.x0,
                            "text",
                            fragment_rect,
                            text,
                            None,
                        )
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
