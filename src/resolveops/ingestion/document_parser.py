
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from resolveops.ingestion.html_parser import parse_html
from resolveops.ingestion.markdown_parser import parse_markdown
from resolveops.ingestion.models import Document
from resolveops.ingestion.pdf_parser import parse_pdf


def parse_document(
    file_path: str | Path,
    page_number: int | None = None,
    *,
    page_range: tuple[int, int] | Sequence[int] | None = None,
    source_id: str | None = None,
    pdf_engine: Literal["pymupdf", "docling"] = "pymupdf",
) -> Document:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return parse_pdf(
            path,
            page_number=page_number,
            page_range=page_range,
            source_id=source_id,
            engine=pdf_engine,
        )

    if ext in {".html", ".htm"}:
        return parse_html(path, source_id=source_id)

    if ext in {".md", ".markdown"}:
        return parse_markdown(path, source_id=source_id)

    raise ValueError(f"Unsupported format: {ext}")
