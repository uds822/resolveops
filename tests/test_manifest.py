from __future__ import annotations

from pathlib import Path
import pytest
from qdrant_client import QdrantClient

from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import EmbeddingModel
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.manifest import (
    IndexManifest,
    InconsistentIndexError,
    load_manifest,
    verify_index_consistency,
)
from resolveops.retrieval.vector_store import QdrantVectorStore


class Mock3DEmbedding(EmbeddingModel):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def _make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc_1",
        source="doc.pdf",
        source_type=SourceType.PDF,
        document_title="Doc",
        kind="text",
        content=content,
        block_ids=["b1"],
        context_block_ids=[],
        heading_context=["Sec"],
        page_numbers=[1],
        indexable=True,
    )


def test_manifest_3way_handshake_success(tmp_path: Path):
    """Verify that Qdrant, BM25 snapshot, and IndexManifest are strictly aligned."""
    manifest_path = tmp_path / "index_manifest.json"
    bm25_path = tmp_path / "bm25_index.json"
    qdrant_dir = tmp_path / "qdrant_db"

    embedder = Mock3DEmbedding()
    vector_store = QdrantVectorStore(embedding_model=embedder, path=qdrant_dir)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("c1", "First chunk about DynamoDB.")
    c2 = _make_chunk("c2", "Second chunk about S3.")
    retriever.add_chunks([c1, c2])

    # Save state atomically
    manifest = retriever.save_state(manifest_path, bm25_path)

    assert manifest_path.is_file()
    assert bm25_path.is_file()
    assert manifest.chunk_count == 2
    assert manifest.generation_id == vector_store.get_generation_id()
    assert manifest.generation_id == bm25.get_generation_id()

    # Verify consistency passes without error
    verified = retriever.verify_consistency(manifest_path)
    assert verified.generation_id == manifest.generation_id
    assert verified.chunk_count == 2
    vector_store.client.close()


def test_manifest_tampered_bm25_fails(tmp_path: Path):
    """Verify that tampering with the BM25 snapshot file fails the checksum validation."""
    manifest_path = tmp_path / "index_manifest.json"
    bm25_path = tmp_path / "bm25_index.json"
    qdrant_dir = tmp_path / "qdrant_db"

    embedder = Mock3DEmbedding()
    vector_store = QdrantVectorStore(embedding_model=embedder, path=qdrant_dir)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("c1", "Chunk content.")
    retriever.add_chunks([c1])
    retriever.save_state(manifest_path, bm25_path)

    # Tamper with the BM25 snapshot file on disk
    with open(bm25_path, "a", encoding="utf-8") as f:
        f.write(" ")

    with pytest.raises(InconsistentIndexError, match="BM25 snapshot checksum mismatch"):
        verify_index_consistency(manifest_path, vector_store, bm25)
    vector_store.client.close()


def test_manifest_desynced_qdrant_generation_fails(tmp_path: Path):
    """
    Directly tests user finding:
    If Qdrant's generation ID does not match the manifest/BM25,
    verification must fail immediately with InconsistentIndexError.
    """
    manifest_path = tmp_path / "index_manifest.json"
    bm25_path = tmp_path / "bm25_index.json"
    qdrant_dir = tmp_path / "qdrant_db"

    embedder = Mock3DEmbedding()
    vector_store = QdrantVectorStore(embedding_model=embedder, path=qdrant_dir)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("c1", "Valid chunk content.")
    retriever.add_chunks([c1])
    retriever.save_state(manifest_path, bm25_path)

    # Simulate Qdrant state drift (e.g., restored from an older backup)
    vector_store.set_generation_id("gen_stale_or_drifted_9999")

    with pytest.raises(InconsistentIndexError, match="VectorStore generation ID desynchronization"):
        verify_index_consistency(manifest_path, vector_store, bm25)
    vector_store.client.close()


def test_manifest_desynced_bm25_generation_fails(tmp_path: Path):
    """Verify that BM25 generation mismatch triggers InconsistentIndexError."""
    manifest_path = tmp_path / "index_manifest.json"
    bm25_path = tmp_path / "bm25_index.json"
    qdrant_dir = tmp_path / "qdrant_db"

    embedder = Mock3DEmbedding()
    vector_store = QdrantVectorStore(embedding_model=embedder, path=qdrant_dir)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("c1", "Valid chunk content.")
    retriever.add_chunks([c1])
    retriever.save_state(manifest_path, bm25_path)

    # Simulate BM25 in-memory generation drift
    bm25.set_generation_id("gen_different_002")

    with pytest.raises(InconsistentIndexError, match="BM25 generation ID desynchronization"):
        verify_index_consistency(manifest_path, vector_store, bm25)
    vector_store.client.close()


def test_manifest_chunk_count_mismatch_fails(tmp_path: Path):
    """Verify that asymmetric chunk counts trigger InconsistentIndexError."""
    manifest_path = tmp_path / "index_manifest.json"
    bm25_path = tmp_path / "bm25_index.json"
    qdrant_dir = tmp_path / "qdrant_db"

    embedder = Mock3DEmbedding()
    vector_store = QdrantVectorStore(embedding_model=embedder, path=qdrant_dir)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("c1", "First chunk.")
    c2 = _make_chunk("c2", "Second chunk.")
    retriever.add_chunks([c1, c2])
    manifest = retriever.save_state(manifest_path, bm25_path)

    # Directly delete one chunk from Qdrant to cause a desync
    vector_store.delete_chunks(["c2"])
    assert vector_store.count() == 1
    assert bm25.count() == 2

    with pytest.raises(InconsistentIndexError, match="Chunk count mismatch across stores"):
        verify_index_consistency(manifest_path, vector_store, bm25)
    vector_store.client.close()

