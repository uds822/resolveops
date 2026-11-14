"""Orchestrate document loading, parsing, chunking, and vector indexing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from resolveops.ingestion.chunking import Chunk, chunk_document
from resolveops.ingestion.document_parser import parse_document
from resolveops.ingestion.models import Document
from resolveops.retrieval.vector_store import VectorStore


class IngestionResult(BaseModel):
    """Structured report returned after ingesting a document."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "partial", "failed"]
    document_id: str | None = None
    source: str
    source_type: str | None = None
    pages_processed: list[int] = Field(default_factory=list)
    total_blocks: int = 0
    total_chunks: int = 0
    indexed_chunks: int = 0
    quarantined_chunks: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


def ingest_document(
    file_path: str | Path,
    vector_store: VectorStore,
    *,
    page_number: int | None = None,
    page_range: tuple[int, int] | None = None,
    source_id: str | None = None,
    pdf_engine: Literal["pymupdf", "docling"] = "pymupdf",
    max_chars: int = 1800,
) -> IngestionResult:
    """Run the complete offline ingestion pipeline on a single file.

    Steps:
      1. Parse file into canonical Document (PDF, HTML, or Markdown)
      2. Chunk document into semantic text & whole-table units
      3. Index valid chunks into the provided VectorStore (skipping quarantined tables)
      4. Return an IngestionResult metrics report
    """
    path = Path(file_path)
    started = time.monotonic()
    source_label = source_id or str(path)
#Before doing any heavy work, it checks if the file actually exists.
#If not, it exits immediately without wasting CPU.
    if not path.is_file():
        return IngestionResult(
            status="failed",
            source=source_label,
            error=f"File not found: {path}",
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    try:
        # 1. Parse document
        document: Document = parse_document(
            path,
            page_number=page_number,
            page_range=page_range,
            source_id=source_id,
            pdf_engine=pdf_engine,
        )

        # 2. Chunk document
        chunks: list[Chunk] = chunk_document(document, max_chars=max_chars)

        # 3. Add chunks to vector store
        indexed_count = vector_store.add_chunks(chunks)
        quarantined_count = sum(1 for c in chunks if not c.indexable)

        parsed_pages = document.metadata.get("parsed_pages", [])
        if not parsed_pages:
            parsed_pages = sorted(
                list({b.page_number for b in document.blocks if b.page_number is not None})
            )

        elapsed = round(time.monotonic() - started, 3)

        status: Literal["success", "partial", "failed"] = "success"
        if quarantined_count > 0:
            status = "partial"  # parsed, but some chunks quarantined for review

        return IngestionResult(
            status=status,
            document_id=document.id,
            source=document.source,
            source_type=document.source_type.value,
            pages_processed=parsed_pages,
            total_blocks=len(document.blocks),
            total_chunks=len(chunks),
            indexed_chunks=indexed_count,
            quarantined_chunks=quarantined_count,
            elapsed_seconds=elapsed,
        )

    except Exception as exc:
        return IngestionResult(
            status="failed",
            source=source_label,
            error=str(exc),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

