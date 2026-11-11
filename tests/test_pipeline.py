from pathlib import Path

import pymupdf
import pytest

from resolveops.ingestion.pipeline import IngestionResult, ingest_document
from resolveops.retrieval.embeddings import EmbeddingModel
from resolveops.retrieval.vector_store import InMemoryVectorStore


class MockEmbedding(EmbeddingModel):
    """Fast deterministic mock for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_ingest_markdown_file_end_to_end(tmp_path):
    # 1. Create a markdown doc
    md_file = tmp_path / "guide.md"
    md_file.write_text("# DynamoDB Guide\n\nDynamoDB is a managed NoSQL database.")

    store = InMemoryVectorStore(MockEmbedding())

    # 2. Run ingestion pipeline
    result = ingest_document(md_file, store)

    # 3. Verify metrics
    assert isinstance(result, IngestionResult)
    assert result.status == "success"
    assert result.source_type == "markdown"
    assert result.total_chunks > 0
    assert result.indexed_chunks == result.total_chunks
    assert result.quarantined_chunks == 0
    assert result.elapsed_seconds >= 0.0

    # 4. Verify search finds it
    search_results = store.search("NoSQL database")
    assert len(search_results) > 0
    assert "DynamoDB" in search_results[0].chunk.content


def test_ingest_pdf_file_end_to_end(tmp_path):
    # 1. Create a 2-page PDF
    pdf_file = tmp_path / "manual.pdf"
    with pymupdf.open() as pdf:
        p1 = pdf.new_page()
        p1.insert_text((72, 72), "Page one: DynamoDB partition limits.")
        p2 = pdf.new_page()
        p2.insert_text((72, 72), "Page two: Secondary index limits.")
        pdf.save(pdf_file)

    store = InMemoryVectorStore(MockEmbedding())

    # 2. Run ingestion pipeline
    result = ingest_document(pdf_file, store, pdf_engine="pymupdf")

    # 3. Verify metrics
    assert result.status == "success"
    assert result.source_type == "pdf"
    assert result.pages_processed == [1, 2]
    assert result.indexed_chunks > 0
    assert store.count() > 0


def test_ingest_nonexistent_file_returns_failed_result():
    store = InMemoryVectorStore(MockEmbedding())
    missing_file = Path("non_existent_document_12345.pdf")

    result = ingest_document(missing_file, store)

    assert result.status == "failed"
    assert "File not found" in (result.error or "")
    assert result.indexed_chunks == 0
    assert store.count() == 0

