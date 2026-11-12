"""Cross-encoder reranking layer for high-precision passage scoring."""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Sequence

import requests
from pydantic import BaseModel, ConfigDict

from resolveops.ingestion.chunking import Chunk


def _sigmoid(x: float) -> float:
    """Map raw transformer logit to a bounded probability [0.0, 1.0]."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class RerankResult(BaseModel):
    """A chunk re-scored by a Cross-Encoder with full provenance and audit history."""

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float          # Normalized relevance probability (0.0 to 1.0)
    logit: float          # Raw transformer output logit
    initial_rank: int     # 1-based rank from candidate retrieval (e.g. #4 in Hybrid Search)
    reranked_rank: int    # 1-based rank after reranker scoring (e.g. #1)


class Reranker(ABC):
    """Abstract base class for all Cross-Encoder rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int = 3,
    ) -> list[RerankResult]:
        """Re-score a sequence of candidate chunks for a query.

        Args:
            query: The user's search query.
            chunks: Candidate chunks retrieved from the candidate funnel (e.g. HybridRetriever).
            top_k: Number of highest-relevance chunks to return.

        Returns:
            Sorted list of RerankResult objects in descending relevance order.
        """
        ...


class NVIDIAReranker(Reranker):
    """Production Cross-Encoder reranker powered by NVIDIA NIM (e.g. nemotron-3.5-rerank-8b).

    Unlike Bi-Encoders which compare pre-computed vectors, this model feeds (query, passage)
    jointly through an 8-billion parameter transformer with full cross-attention.
    """

    DEFAULT_ENDPOINT = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking"
    DEFAULT_MODEL = "nvidia/nemotron-3.5-rerank-8b"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        endpoint_url: str = DEFAULT_ENDPOINT,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is required for NVIDIAReranker. "
                "Set it in .env or pass api_key explicitly."
            )
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int = 3,
    ) -> list[RerankResult]:
        if not query or not query.strip():
            return []

        chunk_list = list(chunks)
        if not chunk_list:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Build payload matching NVIDIA NeMo Retriever ranking specification
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": {"text": query.strip()},
            "passages": [{"text": c.content} for c in chunk_list],
        }

        response = requests.post(
            self.endpoint_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"NVIDIA Reranking API failed ({response.status_code}): {response.text}"
            )

        data = response.json()
        rankings = data.get("rankings", [])

        # Build results with audit mapping between initial and final ranks
        results: list[RerankResult] = []
        for new_rank, item in enumerate(rankings[:top_k], start=1):
            idx = item.get("index")
            if idx is None or idx < 0 or idx >= len(chunk_list):
                continue
            raw_logit = float(item.get("logit", 0.0))
            prob = round(_sigmoid(raw_logit), 6)

            results.append(
                RerankResult(
                    chunk=chunk_list[idx],
                    score=prob,
                    logit=round(raw_logit, 4),
                    initial_rank=idx + 1,  # 1-based index in candidate pool
                    reranked_rank=new_rank,
                )
            )

        return results


class MockReranker(Reranker):
    """Fast, deterministic reranker for offline CI unit testing without API keys."""

    def __init__(self, target_boost_keyword: str | None = None) -> None:
        self.target_boost_keyword = target_boost_keyword

    def rerank(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int = 3,
    ) -> list[RerankResult]:
        if not query or not query.strip() or not chunks:
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        q_terms = set(query.lower().split())

        scored_candidates: list[tuple[int, Chunk, float, float]] = []
        for idx, chunk in enumerate(chunks):
            content_lower = chunk.content.lower()
            term_matches = sum(1 for t in q_terms if t in content_lower)
            boost = 5.0 if self.target_boost_keyword and self.target_boost_keyword.lower() in content_lower else 0.0

            logit = float(term_matches * 1.5 + boost - 2.0)
            prob = round(_sigmoid(logit), 6)
            scored_candidates.append((idx + 1, chunk, prob, logit))

        # Sort descending by logit / score
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        return [
            RerankResult(
                chunk=chunk,
                score=prob,
                logit=round(logit, 4),
                initial_rank=orig_rank,
                reranked_rank=new_rank,
            )
            for new_rank, (orig_rank, chunk, prob, logit) in enumerate(scored_candidates[:top_k], start=1)
        ]


def make_reranker(
    model_name: str = "nvidia:nvidia/nemotron-3.5-rerank-8b",
    api_key: str | None = None,
) -> Reranker:
    """Factory creating appropriate Reranker instance based on prefix."""
    if model_name.startswith("nvidia:"):
        actual_model = model_name.removeprefix("nvidia:") or NVIDIAReranker.DEFAULT_MODEL
        return NVIDIAReranker(model_name=actual_model, api_key=api_key)
    if model_name == "mock":
        return MockReranker()

    raise ValueError(f"Unsupported reranker model: '{model_name}'. Use 'nvidia:<model>' or 'mock'.")

