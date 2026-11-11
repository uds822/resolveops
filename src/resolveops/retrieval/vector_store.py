from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from resolveops.ingestion.chunking import Chunk
from resolveops.retrieval.embeddings import EmbeddingModel


class SearchResult(BaseModel):
    """A retrieved chunk paired with its relevance score."""
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float


class VectorStore(ABC):
    """Abstract interface for all vector stores."""

    @abstractmethod
    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and index chunks. Returns count of newly added chunks."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Search for the most relevant chunks given a text query."""
        ...


class InMemoryVectorStore(VectorStore):
    """
    Fast, lightweight in-memory vector store using NumPy cosine similarity.
    Ideal for local development, testing, and bounded pilots.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """
        Embed and store chunks.
        
        Enforces two crucial rules:
        1. Skips chunks where indexable is False (quarantined bad tables).
        2. Skips duplicate chunk IDs (idempotent).
        """
        new_chunks = [
            chunk for chunk in chunks
            if chunk.indexable and chunk.id not in self._chunks
        ]

        if not new_chunks:
            return 0

        # Batch embed the texts for efficiency
        texts = [chunk.content for chunk in new_chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        for chunk, vector in zip(new_chunks, embeddings):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = np.array(vector, dtype=np.float32)

        return len(new_chunks)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Search for top_k chunks closest to query using cosine similarity.
        """
        if not self._chunks:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # 1. Embed query
        query_vec = np.array(
            self.embedding_model.embed_query(query),
            dtype=np.float32,
        )
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        # 2. Build doc matrix and calculate cosine similarity
        chunk_ids = list(self._chunks.keys())
        doc_matrix = np.stack([self._vectors[cid] for cid in chunk_ids])
        doc_norms = np.linalg.norm(doc_matrix, axis=1)

        # Avoid division by zero
        valid = doc_norms > 0
        scores = np.zeros(len(chunk_ids), dtype=np.float32)
        scores[valid] = np.dot(doc_matrix[valid], query_vec) / (doc_norms[valid] * query_norm)

        # 3. Sort by highest score first
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                chunk=self._chunks[chunk_ids[idx]],
                score=round(float(scores[idx]), 4),
            )
            for idx in top_indices
        ]

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)