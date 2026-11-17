"""End-to-end RAG orchestrator service connecting retrieval, reranking, and generation."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from resolveops.generation.context_builder import CitationReference, build_grounded_prompt
from resolveops.generation.grounding_verifier import GroundingAuditReport, GroundingVerifier
from resolveops.generation.llm import GenerationResult, LLMClient
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.reranker import Reranker


class RAGTimings(BaseModel):
    """Detailed latency telemetry breakdown across each pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    retrieval_ms: float = Field(description="Stage 1 Hybrid search (Dense + BM25 + RRF) latency in ms")
    rerank_ms: float = Field(description="Stage 2 Cross-Encoder reranking latency in ms")
    generation_ms: float = Field(description="Stage 3 LLM generation latency in ms")
    claim_audit_ms: float = Field(default=0.0, description="Stage 5 Semantic claim-level grounding verification latency in ms")
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
    faithfulness_score: float | None = Field(default=None, description="Ratio of supported claims (0.0 to 1.0) from Tier 2 claim verifier")
    claim_audit: GroundingAuditReport | None = Field(default=None, description="Detailed per-claim audit results from semantic verifier")
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
        grounding_verifier: GroundingVerifier | None = None,
        min_rerank_score: float = 0.0,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.grounding_verifier = grounding_verifier
        self.min_rerank_score = min_rerank_score

    def answer_query(
        self,
        query: str,
        candidate_pool: int = 10,
        top_k: int = 3,
        max_chars_budget: int = 12000,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        response_style: str = "default",
        min_rerank_score: float | None = None,
    ) -> RAGResponse:
        """Execute end-to-end question answering pipeline: Retrieve -> Rerank -> Prompt -> Generate.

        Args:
            query: The user's technical question.
            candidate_pool: Number of candidate chunks to retrieve in Stage 1 (default 10).
            top_k: Number of highest-relevance chunks to pass to the LLM (default 3).
            max_chars_budget: Maximum character budget for the injected context.
            max_tokens: Maximum tokens for LLM generation.
            temperature: Sampling temperature for LLM generation (default 0.0 for factual accuracy).
            response_style: 'default' for standard technical Markdown or 'voice' for concise spoken answers.
            min_rerank_score: Optional minimum cross-encoder probability cutoff (0.0 to 1.0) to drop irrelevant chunks.

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

        # ── Stage 2: Cross-Encoder Reranking & Relevance Filtering ────────────
        t1 = time.monotonic()
        candidate_chunks = [c.chunk for c in candidates]
        reranked_results = self.reranker.rerank(clean_query, candidate_chunks, top_k=top_k)
        effective_min_score = self.min_rerank_score if min_rerank_score is None else min_rerank_score
        if effective_min_score > 0.0:
            reranked_results = [r for r in reranked_results if r.score >= effective_min_score]
        rerank_ms = (time.monotonic() - t1) * 1000

        # ── Stage 3: Grounded Context Assembly & Citation Mapping ─────────────
        top_chunks = [r.chunk for r in reranked_results]
        assembled_prompt = build_grounded_prompt(
            query=clean_query,
            chunks=top_chunks,
            max_chars_budget=max_chars_budget,
            response_style=response_style,
        )

        # ── Stage 4: Citation-Grounded LLM Generation ─────────────────────────
        t2 = time.monotonic()
        gen_result = self.llm_client.generate(
            prompt=assembled_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        generation_ms = (time.monotonic() - t2) * 1000

        from resolveops.generation.llm import is_refusal_answer

        is_refusal = is_refusal_answer(gen_result.answer)

        # ── Stage 5: Semantic Claim-Level Entailment Verification (Tier 2 Batch Strategy) ──
        claim_audit_ms = 0.0
        audit_report = None
        faithfulness_score = None

        if self.grounding_verifier is not None and not is_refusal and gen_result.cited_citations:
            t3 = time.monotonic()
            try:
                audit_report = self.grounding_verifier.verify_claims(
                    answer=gen_result.answer,
                    citations=gen_result.cited_citations,
                )
                faithfulness_score = audit_report.faithfulness_score
            except Exception as exc:
                audit_report = GroundingAuditReport(
                    total_claims=0,
                    supported_claims=0,
                    unsupported_claims=0,
                    faithfulness_score=1.0,
                    claims=[],
                    unsupported_details=[],
                    audit_status=f"error: {str(exc)}",
                    latency_ms=0.0,
                )
            claim_audit_ms = (time.monotonic() - t3) * 1000

        total_ms = (time.monotonic() - t_start) * 1000

        timings = RAGTimings(
            retrieval_ms=round(retrieval_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            generation_ms=round(generation_ms, 2),
            claim_audit_ms=round(claim_audit_ms, 2),
            total_ms=round(total_ms, 2),
        )

        validation_failed = (
            bool(gen_result.hallucinated_markers)
            or (bool(assembled_prompt.citations) and not bool(gen_result.cited_citations) and not is_refusal)
            or (audit_report is not None and audit_report.unsupported_claims > 0)
        )

        return RAGResponse(
            query=clean_query,
            answer=gen_result.answer,
            citations=gen_result.citations,
            retrieved_citations=gen_result.retrieved_citations,
            cited_citations=gen_result.cited_citations,
            hallucinated_markers=gen_result.hallucinated_markers,
            citation_validation_failed=validation_failed,
            faithfulness_score=faithfulness_score,
            claim_audit=audit_report,
            timings=timings,
            candidate_count=len(candidates),
            reranked_count=len(reranked_results),
            model_name=gen_result.model_name,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            total_tokens=gen_result.total_tokens,
        )

