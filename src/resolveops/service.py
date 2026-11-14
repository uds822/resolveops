"""End-to-end RAG orchestrator service connecting retrieval, reranking, and generation."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from resolveops.generation.context_builder import CitationReference, build_grounded_prompt
from resolveops.generation.llm import GenerationResult, LLMClient
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.reranker import Reranker


class RAGTimings(BaseModel):
    """Detailed latency telemetry breakdown across each pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    retrieval_ms: float = Field(description="Stage 1 Hybrid search (Dense + BM25 + RRF) latency in ms")
    rerank_ms: float = Field(description="Stage 2 Cross-Encoder reranking latency in ms")
    generation_ms: float = Field(description="Stage 3 LLM generation latency in ms")
    total_ms: float = Field(description="End-to-end total query execution latency in ms")


class RAGResponse(BaseModel):
    """The complete, audited response returned to the caller or API layer."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The user's original query")
    answer: str = Field(description="Verified, citation-tagged answer text")
    citations: list[CitationReference] = Field(default_factory=list, description="Provenance metadata for cited references")
    retrieved_citations: list[CitationReference] = Field(default_factory=list, description="All references retrieved and passed in context")
    cited_citations: list[CitationReference] = Field(default_factory=list, description="References explicitly cited in the answer")
    hallucinated_markers: list[int] = Field(default_factory=list, description="Citation numbers in text that had no corresponding source")
    citation_validation_failed: bool = Field(default=False, description="True if answer hallucinated markers or failed to cite context")
    timings: RAGTimings = Field(description="Latency profile across pipeline stages")
    candidate_count: int = Field(description="Number of candidates retrieved in Stage 1")
    reranked_count: int = Field(description="Number of high-confidence chunks scored in Stage 2")
    model_name: str = Field(description="Generative LLM model name used")
    prompt_tokens: int = Field(default=0, description="Tokens consumed by prompt")
    completion_tokens: int = Field(default=0, description="Tokens generated in completion")
    total_tokens: int = Field(default=0, description="Total tokens utilized")


class RAGService:
    """Production facade orchestrating the two-stage retrieval and citation-grounded generation pipeline."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm_client: LLMClient,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client

    def answer_query(
        self,
        query: str,
        candidate_pool: int = 10,
        top_k: int = 3,
        max_chars_budget: int = 12000,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> RAGResponse:
        """Execute end-to-end question answering pipeline: Retrieve -> Rerank -> Prompt -> Generate.

        Args:
            query: The user's technical question.
            candidate_pool: Number of candidate chunks to retrieve in Stage 1 (default 10).
            top_k: Number of highest-relevance chunks to pass to the LLM (default 3).
            max_chars_budget: Maximum character budget for the injected context.
            max_tokens: Maximum tokens for LLM generation.
            temperature: Sampling temperature for LLM generation (default 0.0 for factual accuracy).

        Returns:
            RAGResponse with grounded answer, citation mappings, and latency metrics.
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query cannot be empty or whitespace")

        t_start = time.monotonic()

        # ── Stage 1: Hybrid Candidate Retrieval (Dense + BM25 + RRF) ──────────
        t0 = time.monotonic()
        candidates = self.retriever.search(clean_query, top_k=candidate_pool)
        retrieval_ms = (time.monotonic() - t0) * 1000

        # ── Stage 2: Cross-Encoder Reranking ──────────────────────────────────
        t1 = time.monotonic()
        candidate_chunks = [c.chunk for c in candidates]
        reranked_results = self.reranker.rerank(clean_query, candidate_chunks, top_k=top_k)
        rerank_ms = (time.monotonic() - t1) * 1000

        # ── Stage 3: Grounded Context Assembly & Citation Mapping ─────────────
        top_chunks = [r.chunk for r in reranked_results]
        assembled_prompt = build_grounded_prompt(
            query=clean_query,
            chunks=top_chunks,
            max_chars_budget=max_chars_budget,
        )

        # ── Stage 4: Citation-Grounded LLM Generation ─────────────────────────
        t2 = time.monotonic()
        gen_result = self.llm_client.generate(
            prompt=assembled_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        generation_ms = (time.monotonic() - t2) * 1000
        total_ms = (time.monotonic() - t_start) * 1000

        timings = RAGTimings(
            retrieval_ms=round(retrieval_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            generation_ms=round(generation_ms, 2),
            total_ms=round(total_ms, 2),
        )

        from resolveops.generation.llm import is_refusal_answer

        is_refusal = is_refusal_answer(gen_result.answer)
        validation_failed = bool(gen_result.hallucinated_markers) or (
            bool(assembled_prompt.citations) and not bool(gen_result.cited_citations) and not is_refusal
        )

        return RAGResponse(
            query=clean_query,
            answer=gen_result.answer,
            citations=gen_result.citations,
            retrieved_citations=gen_result.retrieved_citations,
            cited_citations=gen_result.cited_citations,
            hallucinated_markers=gen_result.hallucinated_markers,
            citation_validation_failed=validation_failed,
            timings=timings,
            candidate_count=len(candidates),
            reranked_count=len(reranked_results),
            model_name=gen_result.model_name,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            total_tokens=gen_result.total_tokens,
        )

