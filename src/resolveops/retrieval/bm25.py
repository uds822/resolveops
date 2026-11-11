from __future__ import annotations

import math
import re
from collections import Counter
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

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Document length normalization parameter

        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._doc_freqs: Counter[str] = Counter()  # How many docs contain token t
        self.avg_doc_len: float = 0.0

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Index chunks for BM25 lexical search. Skips unindexable chunks."""
        valid_chunks = [c for c in chunks if c.indexable]
        if not valid_chunks:
            return 0

        start_idx = len(self._chunks)
        for chunk in valid_chunks:
            tokens = technical_tokenize(chunk.content)
            self._chunks.append(chunk)
            self._doc_tokens.append(tokens)
            self._doc_lens.append(len(tokens))

            # Update document frequency for unique tokens in this doc
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freqs[token] += 1

        self.avg_doc_len = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        return len(valid_chunks)

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