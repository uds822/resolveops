"""Create citation-ready retrieval units from a canonical document."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from resolveops.ingestion.identifiers import make_stable_id
from resolveops.ingestion.models import Block, BlockType, Document, SourceType, TableCell


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    source: str
    source_type: SourceType
    document_title: str | None
    kind: Literal["text", "table"]
    content: str
    block_ids: list[str]
    context_block_ids: list[str]
    heading_context: list[str]
    page_numbers: list[int]
    indexable: bool = True
    table_data: list[list[str]] | None = None
    table_cells: list[TableCell] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def chunk_document(document: Document, *, max_chars: int = 1800) -> list[Chunk]:
    """Group text by headings and size; emit every table as one complete chunk.

    A single block can exceed max_chars. It remains intact so citations and
    source structure are never cut in the middle of a paragraph or table.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    chunks: list[Chunk] = []
    headings: list[Block] = []
    pending: list[Block] = []

    def emit(kind: Literal["text", "table"], blocks: list[Block]) -> None:
        if not blocks:
            return
        context = [heading.content for heading in headings]
        content = "\n\n".join([*context, *(block.content for block in blocks)])
        pages = list(dict.fromkeys(
            block.page_number for block in blocks if block.page_number is not None
        ))
        table = blocks[0] if kind == "table" else None
        extraction = table.metadata.get("table_extraction", {}) if table else {}
        chunks.append(Chunk(
            id=make_stable_id(
                "chunk", document.id, kind, content, *(block.id for block in blocks)
            ),
            document_id=document.id,
            source=document.source,
            source_type=document.source_type,
            document_title=document.title,
            kind=kind,
            content=content,
            block_ids=[block.id for block in blocks],
            context_block_ids=[heading.id for heading in headings],
            heading_context=context,
            page_numbers=pages,
            indexable=extraction.get("validation_state") != "needs_review",
            table_data=table.table_data if table else None,
            table_cells=table.table_cells if table else None,
            metadata={
                "bbox": table.metadata.get("bbox"),
                "bbox_origin": table.metadata.get("bbox_origin"),
                "table_extraction": extraction,
            } if table else {},
        ))

    for block in sorted(document.blocks, key=lambda item: item.order):
        if block.type == BlockType.HEADING:
            emit("text", pending)
            pending = []
            level = block.heading_level or 1
            headings = [heading for heading in headings if (heading.heading_level or 1) < level]
            headings.append(block)
        elif block.type == BlockType.TABLE:
            emit("text", pending)
            pending = []
            emit("table", [block])
        elif block.content.strip():
            candidate_length = len("\n\n".join(
                [*(heading.content for heading in headings),
                 *(item.content for item in pending), block.content]
            ))
            if pending and candidate_length > max_chars:
                emit("text", pending)
                pending = []
            pending.append(block)

    emit("text", pending)
    return chunks
