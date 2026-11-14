from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from resolveops.ingestion.chunking import Chunk


class BM25SearchResult(BaseModel):
    """A chunk retrieved via BM25 paired with its lexical match score."""
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float

def technical_tokenize(text: str) -> list[str]:
    """
    Tokenize text while preserving technical identifiers like:
    - Version numbers: '3.3.2', '2.55.15'
    - Filenames: 'pom.xml', 'docker-compose.yml'
    - Identifiers: 'TransactionConflictExceptions', 'cve-2024-1234'
    """
    raw_tokens = re.findall(r"[a-z0-9_.\-]+", text.lower())
    # Strip trailing/leading punctuation (e.g., 'cve-2024-1234.' -> 'cve-2024-1234')
    # while keeping dots and dashes INSIDE the word ('pom.xml', '3.3.2')
    cleaned = [t.strip(".-_") for t in raw_tokens]
    return [t for t in cleaned if t]


class BM25Index:
    """
    Pure-Python Okapi BM25 search index (the same math used in Elasticsearch).
    Lightweight, runs in milliseconds, zero external servers or Java needed.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        generation_id: str | None = None,
    ):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Document length normalization parameter
        self.generation_id = generation_id

        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._doc_freqs: Counter[str] = Counter()  # How many docs contain token t
        self._indexed_ids: set[str] = set()
        self.avg_doc_len: float = 0.0

    def get_generation_id(self) -> str | None:
        """Return the current generation ID of the BM25 index."""
        return self.generation_id

    def set_generation_id(self, generation_id: str) -> None:
        """Set the generation ID of the BM25 index."""
        self.generation_id = generation_id

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Index chunks for BM25 lexical search with atomic batch processing."""
        seen_in_batch: set[str] = set()
        valid_chunks: list[Chunk] = []

        for chunk in chunks:
            if not chunk.indexable:
                continue
            if chunk.id in self._indexed_ids or chunk.id in seen_in_batch:
                continue
            seen_in_batch.add(chunk.id)
            valid_chunks.append(chunk)

        if not valid_chunks:
            return 0

        # Pre-tokenize all chunks in isolation; if tokenization fails, state is untouched
        batch_tokens = [technical_tokenize(chunk.content) for chunk in valid_chunks]
        batch_lens = [len(t) for t in batch_tokens]

        # Mutate state atomically
        for chunk, tokens, doc_len in zip(valid_chunks, batch_tokens, batch_lens):
            self._chunks.append(chunk)
            self._doc_tokens.append(tokens)
            self._doc_lens.append(doc_len)
            self._indexed_ids.add(chunk.id)

            for token in set(tokens):
                self._doc_freqs[token] += 1

        self.avg_doc_len = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        return len(valid_chunks)

    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Remove specific chunks by ID. Returns count of removed chunks."""
        ids_to_remove = set(chunk_ids)
        if not ids_to_remove:
            return 0

        new_chunks: list[Chunk] = []
        new_doc_tokens: list[list[str]] = []
        new_doc_lens: list[int] = []
        self._doc_freqs = Counter()

        removed = 0
        for chunk, tokens, doc_len in zip(self._chunks, self._doc_tokens, self._doc_lens):
            if chunk.id in ids_to_remove:
                self._indexed_ids.discard(chunk.id)
                removed += 1
                continue
            new_chunks.append(chunk)
            new_doc_tokens.append(tokens)
            new_doc_lens.append(doc_len)
            for token in set(tokens):
                self._doc_freqs[token] += 1

        self._chunks = new_chunks
        self._doc_tokens = new_doc_tokens
        self._doc_lens = new_doc_lens
        self.avg_doc_len = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        return removed

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document ID."""
        matching_ids = [c.id for c in self._chunks if c.document_id == document_id]
        return self.delete_chunks(matching_ids)

    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> tuple[int, int]:
        """Atomically replace chunks for document_id in BM25 index."""
        seen_in_batch: set[str] = set()
        valid_chunks: list[Chunk] = []
        for chunk in chunks:
            if not chunk.indexable:
                continue
            if chunk.id in seen_in_batch:
                continue
            seen_in_batch.add(chunk.id)
            valid_chunks.append(chunk)

        # Pre-tokenize all new chunks first
        batch_tokens = [technical_tokenize(chunk.content) for chunk in valid_chunks]
        batch_lens = [len(t) for t in batch_tokens]

        # Atomically delete old and add new
        deleted = self.delete_document(document_id)
        for chunk, tokens, doc_len in zip(valid_chunks, batch_tokens, batch_lens):
            self._chunks.append(chunk)
            self._doc_tokens.append(tokens)
            self._doc_lens.append(doc_len)
            self._indexed_ids.add(chunk.id)
            for token in set(tokens):
                self._doc_freqs[token] += 1

        self.avg_doc_len = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        return deleted, len(valid_chunks)

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def _idf(self, token: str) -> float:
        """Calculate Okapi Inverse Document Frequency with smoothing."""
        n_docs = len(self._chunks)
        df = self._doc_freqs.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[BM25SearchResult]:
        """Search chunks using the Okapi BM25 scoring formula."""
        if not self._chunks:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_tokens = technical_tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = [0.0] * len(self._chunks)

        for token in query_tokens:
            idf = self._idf(token)
            if idf <= 0.0:
                continue

            for doc_idx, tokens in enumerate(self._doc_tokens):
                tf = tokens.count(token)
                if tf == 0:
                    continue

                doc_len = self._doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf * (numerator / denominator)

        # Pair scores with chunks and filter out zero-score results
        scored_pairs = [
            (self._chunks[idx], scores[idx])
            for idx in range(len(self._chunks))
            if scores[idx] > 0.0
        ]

        # Sort descending by score
        scored_pairs.sort(key=lambda pair: pair[1], reverse=True)

        return [
            BM25SearchResult(chunk=chunk, score=round(score, 4))
            for chunk, score in scored_pairs[:top_k]
        ]

    def count(self) -> int:
        return len(self._chunks)

    def save_snapshot(self, filepath: str | Path, generation_id: str | None = None) -> None:
        """Persist BM25 index state and generation ID atomically to disk."""
        if generation_id is not None:
            self.generation_id = generation_id

        dest = Path(filepath).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_suffix(".tmp")

        payload = {
            "schema_version": 1,
            "generation_id": self.generation_id,
            "k1": self.k1,
            "b": self.b,
            "avg_doc_len": self.avg_doc_len,
            "chunks": [c.model_dump(mode="json") for c in self._chunks],
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        tmp_path.replace(dest)

    @classmethod
    def load_snapshot(cls, filepath: str | Path) -> BM25Index:
        """Load a saved BM25 index snapshot from disk with its generation ID."""
        path = Path(filepath).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"BM25 snapshot file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75), generation_id=data.get("generation_id"))
        chunks_raw = data.get("chunks", [])
        chunks = [Chunk.model_validate(c) for c in chunks_raw]
        index.add_chunks(chunks)
        index.generation_id = data.get("generation_id")
        return index
