from __future__ import annotations

from pathlib import Path
import pytest
from qdrant_client import QdrantClient

from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.embeddings import EmbeddingModel
from resolveops.retrieval.vector_store import (
    QdrantVectorStore,
    SearchResult,
    SENTINEL_POINT_ID,
    chunk_id_to_uuid,
)


class Mock3DEmbedding(EmbeddingModel):
    """Predictable 3D embedding model for testing vector store math."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)

    def _vector_for(self, text: str) -> list[float]:
        t = text.lower()
        if "dynamodb" in t or "nosql" in t:
            return [1.0, 0.0, 0.0]
        if "s3" in t or "bucket" in t:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def _make_chunk(chunk_id: str, content: str, doc_id: str = "doc_1", indexable: bool = True) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=doc_id,
        source="test.pdf",
        source_type=SourceType.PDF,
        document_title="Test Manual",
        kind="text",
        content=content,
        block_ids=["b1"],
        context_block_ids=[],
        heading_context=["Config"],
        page_numbers=[1],
        indexable=indexable,
    )


def test_qdrant_memory_mode_add_and_search():
    client = QdrantClient(":memory:")
    embedder = Mock3DEmbedding()
    store = QdrantVectorStore(embedding_model=embedder, client=client)

    c1 = _make_chunk("c1", "DynamoDB table partition limits.")
    c2 = _make_chunk("c2", "Amazon S3 bucket policies.")
    c3 = _make_chunk("c3_bad", "Corrupted row.", indexable=False)

    added = store.add_chunks([c1, c2, c3])
    assert added == 2
    assert store.count() == 2

    # Duplicate check
    added_again = store.add_chunks([c1])
    assert added_again == 0
    assert store.count() == 2

    # Query for DynamoDB
    results = store.search("DynamoDB limits", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == "c1"
    assert results[0].score == pytest.approx(1.0, abs=1e-3)


def test_qdrant_sentinel_point_generation_id_isolation():
    client = QdrantClient(":memory:")
    embedder = Mock3DEmbedding()
    store = QdrantVectorStore(embedding_model=embedder, client=client)

    assert store.get_generation_id() is None

    # Stamp generation ID
    store.set_generation_id("gen_20261114_030000")
    assert store.get_generation_id() == "gen_20261114_030000"

    # Verify sentinel point does NOT count toward active chunk count
    assert store.count() == 0

    # Add a user chunk
    c1 = _make_chunk("c1", "DynamoDB item size is 400 KB.")
    store.add_chunks([c1])
    assert store.count() == 1

    # Verify sentinel point is NEVER returned in search results
    results = store.search("anything", top_k=10)
    for res in results:
        assert res.chunk.id != SENTINEL_POINT_ID


def test_qdrant_delete_and_replace():
    client = QdrantClient(":memory:")
    embedder = Mock3DEmbedding()
    store = QdrantVectorStore(embedding_model=embedder, client=client)

    c1 = _make_chunk("c1", "DynamoDB item size is 400 KB.", doc_id="doc_dynamo")
    c2 = _make_chunk("c2", "Secondary index limits.", doc_id="doc_dynamo")
    c3 = _make_chunk("c3", "S3 bucket encryption.", doc_id="doc_s3")

    store.add_chunks([c1, c2, c3])
    assert store.count() == 3

    # Delete single chunk
    removed = store.delete_chunks(["c3"])
    assert removed == 1
    assert store.count() == 2

    # Replace document
    replacement = _make_chunk("c1_new", "Updated DynamoDB 400 KB item limit.", doc_id="doc_dynamo")
    deleted, added = store.replace_document("doc_dynamo", [replacement])
    assert deleted == 2
    assert added == 1
    assert store.count() == 1

    results = store.search("DynamoDB", top_k=1)
    assert results[0].chunk.id == "c1_new"


def test_qdrant_disk_persistence_and_restart(tmp_path: Path):
    """Simulate a full server restart: write to disk, close client, create new instance, verify state."""
    db_path = tmp_path / "qdrant_db"
    embedder = Mock3DEmbedding()

    # Step 1: Initialize store on disk and insert data
    store1 = QdrantVectorStore(
        embedding_model=embedder,
        path=db_path,
        generation_id="gen_session_1",
    )
    c1 = _make_chunk("c1", "Amazon DynamoDB managed NoSQL.")
    store1.add_chunks([c1])
    assert store1.count() == 1
    assert store1.get_generation_id() == "gen_session_1"

    # Step 2: Explicitly close client so files are flushed to disk
    store1.client.close()

    # Step 3: Open second store pointing to the same disk path
    store2 = QdrantVectorStore(
        embedding_model=embedder,
        path=db_path,
    )
    assert store2.count() == 1
    assert store2.get_generation_id() == "gen_session_1"

    # Search works seamlessly after restart
    results = store2.search("DynamoDB NoSQL", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "c1"
    store2.client.close()

