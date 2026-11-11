from pathlib import Path
from typing import Any

from resolveops.ingestion.identifiers import make_stable_id
from resolveops.ingestion.models import Block, BlockType, SourceType, TableCell

#Converts a file path into a clean, normalized string (e.g. forward slashes, resolved absolute path).
def resolve_source(path: Path, source_id: str | None = None) -> str:
    if source_id is None:
        return path.resolve().as_uri()

    source = source_id.strip()
    if not source:
        raise ValueError("source_id cannot be empty")

    return source

#Generates the permanent ID for the document using make_stable_id.
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
    table_cells: list[TableCell] | None = None,
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

    if block_type == BlockType.TABLE and table_data is not None and table_cells is None:
        table_cells = [
            TableCell(
                row_index=row_index,
                column_index=column_index,
                text=value,
            )
            for row_index, row in enumerate(table_data)
            for column_index, value in enumerate(row)
        ]

    return Block(
        id=block_id,
        type=block_type,
        content=content,
        order=order,
        page_number=page_number,
        heading_level=heading_level,
        language=language,
        table_data=table_data,
        table_cells=table_cells,
        metadata=metadata or {},
    )

#Converts a 2D table grid [["ColA", "ColB"], ["1", "2"]] into a clean Markdown table string:
def table_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell for cell in row) for row in rows)
