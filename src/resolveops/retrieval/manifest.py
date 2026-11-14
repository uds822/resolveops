"""Index Manifest and Coordination for Cross-Store Consistency in ResolveOps.

Guarantees that Dense Vector Store (Qdrant) and Sparse Lexical Store (BM25)
remain in lockstep with matching generation IDs, schema versions, and counts.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InconsistentIndexError(RuntimeError):
    """Raised when Qdrant and BM25 index states or generation IDs disagree."""


class IndexManifest(BaseModel):
    """The authoritative coordination record binding Qdrant and BM25 states."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, description="Manifest schema version")
    generation_id: str = Field(description="Unique timestamped generation identifier")
    updated_at: str = Field(description="ISO 8601 timestamp of last state sync")
    corpus_fingerprint: str | None = Field(default=None, description="SHA-256 fingerprint of indexed chunks")
    chunk_count: int = Field(description="Total number of active indexable chunks")
    embedding_model: str = Field(description="Model identifier used for vector embeddings")
    vector_dimension: int = Field(description="Dimensionality of dense vectors")
    distance_metric: str = Field(default="Cosine", description="Vector similarity metric")
    qdrant_collection: str = Field(description="Name of the Qdrant collection")
    bm25_snapshot_file: str = Field(description="Filename/relative path to BM25 snapshot")
    bm25_checksum: str = Field(description="SHA-256 integrity hash of BM25 snapshot file")


def generate_generation_id(prefix: str = "gen") -> str:
    """Generate a chronological, unique generation ID (e.g. gen_20261114_015400)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex}"


def save_manifest(filepath: str | Path, manifest: IndexManifest) -> None:
    """Atomically write index manifest to disk via temporary file rename."""
    dest = Path(filepath).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(".tmp")

    payload = manifest.model_dump(mode="json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    tmp_path.replace(dest)


def load_manifest(filepath: str | Path) -> IndexManifest | None:
    """Load index manifest from disk if it exists; returns None if not found."""
    path = Path(filepath).resolve()
    if not path.is_file():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return IndexManifest.model_validate(data)


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 checksum of a file on disk."""
    path = Path(filepath).resolve()
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_index_consistency(
    manifest_path: str | Path,
    vector_store: Any,
    bm25_index: Any,
) -> IndexManifest:
    """
    Authoritative 3-way consistency handshake:
    Validates that:
    1. Manifest exists on disk and is readable.
    2. BM25 snapshot file exists and matches manifest's SHA-256 checksum.
    3. BM25 in-memory generation ID matches manifest.generation_id.
    4. Dense VectorStore generation ID matches manifest.generation_id.
    5. Chunk counts across manifest, Qdrant/VectorStore, and BM25 index match exactly.

    Raises InconsistentIndexError on any discrepancy.
    Returns the verified IndexManifest.
    """
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise InconsistentIndexError(f"Index manifest not found at: {manifest_path}")

    m_path = Path(manifest_path).resolve()
    bm25_file = Path(manifest.bm25_snapshot_file)
    if not bm25_file.is_absolute():
        bm25_file = (m_path.parent / bm25_file).resolve()

    if not bm25_file.is_file():
        raise InconsistentIndexError(f"BM25 snapshot referenced in manifest missing: {bm25_file}")

    actual_hash = compute_file_sha256(bm25_file)
    if actual_hash != manifest.bm25_checksum:
        raise InconsistentIndexError(
            f"BM25 snapshot checksum mismatch: manifest expected {manifest.bm25_checksum}, "
            f"computed {actual_hash}"
        )

    bm25_gen = bm25_index.get_generation_id()
    if bm25_gen != manifest.generation_id:
        raise InconsistentIndexError(
            f"BM25 generation ID desynchronization: manifest has '{manifest.generation_id}', "
            f"BM25 index has '{bm25_gen}'"
        )

    dense_gen = vector_store.get_generation_id()
    if dense_gen != manifest.generation_id:
        raise InconsistentIndexError(
            f"VectorStore generation ID desynchronization: manifest has '{manifest.generation_id}', "
            f"VectorStore has '{dense_gen}'"
        )

    dense_count = vector_store.count()
    bm25_count = bm25_index.count()
    if not (manifest.chunk_count == dense_count == bm25_count):
        raise InconsistentIndexError(
            f"Chunk count mismatch across stores: manifest={manifest.chunk_count}, "
            f"vector_store={dense_count}, bm25={bm25_count}"
        )

    return manifest
