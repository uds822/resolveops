from __future__ import annotations

import datetime
import copy
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Sequence
from pydantic import BaseModel, ConfigDict

from resolveops.ingestion.chunking import Chunk
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.manifest import (
    IndexManifest,
    InconsistentIndexError,
    compute_file_sha256,
    generate_generation_id,
    save_manifest,
    verify_index_consistency,
)
from resolveops.retrieval.vector_store import VectorStore


def synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class HybridSearchResult(BaseModel):
    """
    A retrieved chunk with its fused RRF score and rank audit metadata.
    Provides complete transparency into why this chunk was ranked where it is.
    """
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float                         # Combined Reciprocal Rank Fusion score
    dense_rank: int | None = None        # 1-based rank in vector search (None if not in candidate pool)
    bm25_rank: int | None = None         # 1-based rank in BM25 search (None if not in candidate pool)
    dense_score: float | None = None     # Raw cosine similarity score
    bm25_score: float | None = None      # Raw BM25 lexical score


class HybridRetriever:
    """
    Enterprise Hybrid Retriever combining Dense (VectorStore) and Sparse (BM25Index)
    search using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._lock = RLock()
        if hasattr(bm25_index, "_indexed_ids") and bm25_index._indexed_ids:
            self._indexed_ids: set[str] = set(bm25_index._indexed_ids)
        else:
            self._indexed_ids = set()

    @synchronized
    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """
        Index chunks into BOTH stores simultaneously, ensuring perfect lockstep.
        Enforces batch-level deduplication and rolls back if either store fails.
        """
        seen_in_batch: set[str] = set()
        new_chunks: list[Chunk] = []

        for c in chunks:
            if not c.indexable:
                continue
            if c.id in self._indexed_ids or c.id in seen_in_batch:
                continue
            seen_in_batch.add(c.id)
            new_chunks.append(c)

        if not new_chunks:
            return 0

        # Step 1: Add to Dense VectorStore
        try:
            dense_added = self.vector_store.add_chunks(new_chunks)
        except Exception:
            self.vector_store.delete_chunks([c.id for c in new_chunks])
            raise

        # Step 2: Add to Sparse BM25Index with rollback protection
        try:
            sparse_added = self.bm25_index.add_chunks(new_chunks)
        except Exception as exc:
            new_ids = [c.id for c in new_chunks]
            self.vector_store.delete_chunks(new_ids)
            self.bm25_index.delete_chunks(new_ids)
            raise RuntimeError(f"Sparse indexing failed; rolled back both stores: {exc}") from exc

        # Step 3: Validate lockstep consistency
        if dense_added != sparse_added:
            new_ids = [c.id for c in new_chunks]
            self.vector_store.delete_chunks(new_ids)
            self.bm25_index.delete_chunks(new_ids)
            raise RuntimeError(
                f"Hybrid index desynchronization detected: dense added {dense_added}, sparse added {sparse_added}. Rolled back both stores."
            )

        for c in new_chunks:
            self._indexed_ids.add(c.id)

        return dense_added

    @synchronized
    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Remove specific chunks from both dense and sparse indices."""
        dense_removed = self.vector_store.delete_chunks(chunk_ids)
        sparse_removed = self.bm25_index.delete_chunks(chunk_ids)
        for cid in chunk_ids:
            self._indexed_ids.discard(cid)
        return min(dense_removed, sparse_removed)

    @synchronized
    def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document from both stores."""
        dense_removed = self.vector_store.delete_document(document_id)
        sparse_removed = self.bm25_index.delete_document(document_id)
        if hasattr(self.bm25_index, "_indexed_ids"):
            self._indexed_ids = set(self.bm25_index._indexed_ids)
        return min(dense_removed, sparse_removed)

    @synchronized
    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> tuple[int, int]:
        """Atomically replace document chunks across both stores with rollback protection."""
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("Replacement chunks must belong to the requested document")
        dense_snapshot = self.vector_store.snapshot_document(document_id)
        sparse_snapshot = copy.deepcopy(self.bm25_index.__dict__)
        try:
            dense_deleted, dense_added = self.vector_store.replace_document(document_id, chunks)
            sparse_deleted, sparse_added = self.bm25_index.replace_document(document_id, chunks)
            if dense_added != sparse_added or dense_deleted != sparse_deleted:
                raise RuntimeError("Hybrid replacement desynchronization detected")
        except Exception as exc:
            self.bm25_index.__dict__.clear()
            self.bm25_index.__dict__.update(sparse_snapshot)
            try:
                self.vector_store.restore_document(document_id, dense_snapshot)
            except Exception as rollback_error:
                raise RuntimeError("Replacement failed and dense rollback failed; index requires recovery") from rollback_error
            raise

        self._indexed_ids = set(self.bm25_index._indexed_ids)

        return dense_deleted, dense_added

    @synchronized
    def count(self) -> int:
        """Return total indexed chunk count, asserting dense and sparse are in lockstep."""
        dense_c = self.vector_store.count()
        sparse_c = self.bm25_index.count()
        if dense_c != sparse_c or dense_c != len(self._indexed_ids):
            raise InconsistentIndexError(
                f"Hybrid index desync: dense={dense_c}, sparse={sparse_c}, tracking={len(self._indexed_ids)}"
            )
        return len(self._indexed_ids)

    @synchronized
    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_k: int = 20,
    ) -> list[HybridSearchResult]:
        """
        Performs hybrid retrieval using the Reciprocal Rank Fusion (RRF) algorithm:
        1. Queries Dense VectorStore for top candidate_pool_k results.
        2. Queries Sparse BM25Index for top candidate_pool_k results.
        3. Fuses ranks using RRF score = sum(weight / (rrf_k + rank)).
        4. Returns top_k highest scoring chunks with full rank breakdown.
        """
        if not query or not query.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Candidate pool must be at least as large as top_k
        pool_k = max(candidate_pool_k, top_k)

        # 1. Fetch candidates from both search engines
        dense_results = self.vector_store.search(query=query, top_k=pool_k)
        bm25_results = self.bm25_index.search(query=query, top_k=pool_k)

        if not dense_results and not bm25_results:
            return []

        # 2. Track candidate chunks and their individual ranks & raw scores
        candidates: dict[str, Chunk] = {}
        dense_ranks: dict[str, int] = {}
        dense_raw_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        bm25_raw_scores: dict[str, float] = {}

        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk.id
            candidates[cid] = res.chunk
            dense_ranks[cid] = rank
            dense_raw_scores[cid] = res.score

        for rank, res in enumerate(bm25_results, start=1):
            cid = res.chunk.id
            candidates[cid] = res.chunk
            bm25_ranks[cid] = rank
            bm25_raw_scores[cid] = res.score

        # 3. Calculate Reciprocal Rank Fusion (RRF) score for each candidate
        fused_scores: dict[str, float] = {}
        for cid in candidates:
            score = 0.0
            if cid in dense_ranks:
                score += self.dense_weight * (1.0 / (self.rrf_k + dense_ranks[cid]))
            if cid in bm25_ranks:
                score += self.bm25_weight * (1.0 / (self.rrf_k + bm25_ranks[cid]))
            fused_scores[cid] = score

        # 4. Sort descending by fused RRF score
        sorted_cids = sorted(
            fused_scores.keys(),
            key=lambda cid: fused_scores[cid],
            reverse=True,
        )

        # 5. Build final explainable search results
        return [
            HybridSearchResult(
                chunk=candidates[cid],
                score=round(fused_scores[cid], 6),
                dense_rank=dense_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                dense_score=dense_raw_scores.get(cid),
                bm25_score=bm25_raw_scores.get(cid),
            )
            for cid in sorted_cids[:top_k]
        ]

    @synchronized
    def save_state(
        self,
        manifest_path: str | Path,
        bm25_path: str | Path,
        generation_id: str | None = None,
    ) -> IndexManifest:
        """
        Atomically persist hybrid retrieval state to disk:
        1. Generates new unique generation ID if none provided.
        2. Stamps dense vector store with the generation ID.
        3. Saves BM25 snapshot with the generation ID.
        4. Calculates BM25 snapshot SHA-256 checksum.
        5. Assembles authoritative IndexManifest and writes to disk atomically.
        """
        gen_id = generation_id or generate_generation_id()
        self.vector_store.set_generation_id(gen_id)
        self.bm25_index.save_snapshot(bm25_path, generation_id=gen_id)

        bm25_file = Path(bm25_path).resolve()
        bm25_hash = compute_file_sha256(bm25_file)

        embed_model = getattr(self.vector_store, "embedding_model", None)
        model_name = getattr(embed_model, "model_name", "unknown")
        dim = getattr(self.vector_store, "vector_dimension", 0)
        if dim == 0 and hasattr(embed_model, "embed_query"):
            dim = len(embed_model.embed_query("dim_probe"))

        col_name = getattr(self.vector_store, "collection_name", "memory")

        manifest = IndexManifest(
            schema_version=1,
            generation_id=gen_id,
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            chunk_count=self.count(),
            embedding_model=str(model_name),
            vector_dimension=int(dim),
            distance_metric="Cosine",
            qdrant_collection=str(col_name),
            bm25_snapshot_file=bm25_file.name,
            bm25_checksum=bm25_hash,
        )
        save_manifest(manifest_path, manifest)
        return manifest

    def verify_consistency(self, manifest_path: str | Path) -> IndexManifest:
        """Verify cross-store consistency against the manifest."""
        return verify_index_consistency(
            manifest_path=manifest_path,
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
        )
