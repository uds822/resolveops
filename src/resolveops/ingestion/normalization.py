from pathlib import Path
from typing import Any

from resolveops.ingestion.identifiers import make_stable_id
from resolveops.ingestion.models import Block, BlockType, SourceType


def resolve_source(path: Path, source_id: str | None = None) -> str:
    if source_id is None:
        return path.resolve().as_uri()

    source = source_id.strip()
    if not source:
        raise ValueError("source_id cannot be empty")

    return source


def make_document_id(source_type: SourceType, source: str) -> str:
    return make_stable_id("doc", source_type.value, source)


def build_block(
    *,
    document_id: str,
    block_type: BlockType,
    content: str,
    order: int,
    identity: str | None = None,
    page_number: int | None = None,
    heading_level: int | None = None,
    language: str | None = None,
    table_data: list[list[str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Block:
    location = identity if identity is not None else str(order)
    block_id = make_stable_id(
        "block",
        document_id,
        location,
        block_type.value,
        content,
    )

    return Block(
        id=block_id,
        type=block_type,
        content=content,
        order=order,
        page_number=page_number,
        heading_level=heading_level,
        language=language,
        table_data=table_data,
        metadata=metadata or {},
    )


def table_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell for cell in row) for row in rows)
