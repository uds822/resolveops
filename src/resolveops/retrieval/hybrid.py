from __future__ import annotations

from typing import Sequence
from pydantic import BaseModel, ConfigDict

from resolveops.ingestion.chunking import Chunk
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.vector_store import VectorStore


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
        self._indexed_ids: set[str] = set()

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """
        Index chunks into BOTH stores simultaneously, ensuring perfect lockstep.
        Automatically deduplicates and skips quarantined unindexable chunks.
        """
        # Filter for valid, un-indexed chunks
        new_chunks = [
            c for c in chunks
            if c.indexable and c.id not in self._indexed_ids
        ]
        if not new_chunks:
            return 0

        # Index into Dense and Sparse stores
        self.vector_store.add_chunks(new_chunks)
        self.bm25_index.add_chunks(new_chunks)

        for c in new_chunks:
            self._indexed_ids.add(c.id)

        return len(new_chunks)

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

    def count(self) -> int:
        """Return the number of uniquely indexed chunks."""
        return len(self._indexed_ids)