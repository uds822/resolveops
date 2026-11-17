"""Unit tests for ResolveOps offline RAGAS evaluator and metrics engine."""

from __future__ import annotations

import pytest

from resolveops.evals.models import EvalExample, RagasMetrics
from resolveops.evals.ragas_evaluator import (
    RagasEvaluator,
    _cosine_similarity,
    _normalize_tokens,
    format_markdown_report,
)
from resolveops.generation.context_builder import CitationReference
from resolveops.generation.grounding_verifier import MockGroundingVerifier
from resolveops.generation.llm import MockLLMClient
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import MockEmbeddingModel
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.reranker import MockReranker
from resolveops.retrieval.vector_store import InMemoryVectorStore
from resolveops.service import RAGResponse, RAGService, RAGTimings


@pytest.fixture
def evaluator() -> RagasEvaluator:
    return RagasEvaluator(
        embedding_model=MockEmbeddingModel(384),
        judge_client=MockLLMClient(),
        grounding_verifier=MockGroundingVerifier(),
    )


def test_normalize_tokens() -> None:
    tokens = _normalize_tokens("AWS Lambda retry 2 times failed!")
    assert "aws" in tokens
    assert "lambda" in tokens
    assert "retry" in tokens
    assert "failed" in tokens


def test_cosine_similarity() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine_similarity([], []) == pytest.approx(0.0)


def test_context_precision_perfect_ranking(evaluator: RagasEvaluator) -> None:
    example = EvalExample(
        example_id="ex1",
        question="How many retries?",
        reference_answer="Twice",
        expected_evidence={"source": "invocation-async.md", "excerpt": "retries the invocation twice"},
        answerability="answerable",
    )
    citations = [
        CitationReference(
            index=1,
            chunk_id="c1",
            source="invocation-async.md",
            content="When an asynchronous function fails, Lambda retries the invocation twice by default.",
            snippet="Lambda retries the invocation twice",
        ),
        CitationReference(
            index=2,
            chunk_id="c2",
            source="TTL.md",
            content="Time to Live removes old items.",
            snippet="Time to Live removes old items.",
        ),
    ]

    # Rank 1 is relevant (1/1 = 1.0), Rank 2 is irrelevant.
    # Total relevant = 1. Precision@1 = 1.0. MAP = 1.0
    prec = evaluator.compute_context_precision(example, citations)
    assert prec == pytest.approx(1.0)


def test_context_precision_demoted_ranking(evaluator: RagasEvaluator) -> None:
    example = EvalExample(
        example_id="ex1",
        question="How many retries?",
        reference_answer="Twice",
        expected_evidence={"source": "invocation-async.md", "excerpt": "retries the invocation twice"},
        answerability="answerable",
    )
    citations = [
        CitationReference(
            index=1,
            chunk_id="c1",
            source="TTL.md",
            content="Time to Live removes old items.",
            snippet="Time to Live removes old items.",
        ),
        CitationReference(
            index=2,
            chunk_id="c2",
            source="invocation-async.md",
            content="When an asynchronous function fails, Lambda retries the invocation twice by default.",
            snippet="Lambda retries the invocation twice",
        ),
    ]

    # Rank 1 is irrelevant, Rank 2 is relevant (cum_rel=1, rank=2 -> 1/2 = 0.5)
    # Total relevant = 1. MAP = 0.5
    prec = evaluator.compute_context_precision(example, citations)
    assert prec == pytest.approx(0.5)


def test_context_precision_unanswerable(evaluator: RagasEvaluator) -> None:
    example = EvalExample(
        example_id="ex_unans",
        question="What is Google BigQuery pricing?",
        reference_answer="I cannot answer this.",
        answerability="unanswerable",
    )
    assert evaluator.compute_context_precision(example, []) is None


def test_context_recall(evaluator: RagasEvaluator) -> None:
    example = EvalExample(
        example_id="ex2",
        question="What does DynamoDB TTL do?",
        reference_answer="Deletes expired items automatically without consuming write throughput.",
        essential_facts=[
            "Deletes expired items automatically",
            "Without consuming write throughput",
        ],
        answerability="answerable",
    )
    citations = [
        CitationReference(
            index=1,
            chunk_id="c1",
            source="TTL.md",
            content="Time to Live deletes expired items automatically without consuming provisioned write throughput.",
            snippet="deletes expired items automatically without consuming write throughput",
        )
    ]
    recall = evaluator.compute_context_recall(example, citations)
    assert recall == pytest.approx(1.0)


def test_faithfulness_refusal_and_claims(evaluator: RagasEvaluator) -> None:
    example = EvalExample(
        example_id="ex3",
        question="What is BigQuery pricing?",
        reference_answer="Cannot answer.",
        answerability="unanswerable",
    )
    resp = RAGResponse(
        query="What is BigQuery pricing?",
        answer="I cannot provide an answer based on the given sources as BigQuery is outside the scope.",
        citations=[],
        retrieved_citations=[],
        cited_citations=[],
        hallucinated_markers=[],
        citation_validation_failed=False,
        faithfulness_score=None,
        claim_audit=None,
        timings=RAGTimings(retrieval_ms=1.0, rerank_ms=1.0, generation_ms=1.0, claim_audit_ms=0.0, total_ms=3.0),
        candidate_count=0,
        reranked_count=0,
        model_name="mock",
    )
    # Refusal should be 100% faithful
    assert evaluator.compute_faithfulness(example, resp) == 1.0


def test_abstention_correctness(evaluator: RagasEvaluator) -> None:
    example_unans = EvalExample(
        example_id="ex4",
        question="Kubernetes ingress controller configuration?",
        reference_answer="Cannot answer.",
        answerability="unanswerable",
    )

    # 1. Successful refusal with 0 citations
    good_resp = RAGResponse(
        query="Kubernetes ingress controller configuration?",
        answer="I cannot provide an answer based on the given sources as Kubernetes is outside the documentation scope.",
        citations=[],
        retrieved_citations=[],
        cited_citations=[],
        hallucinated_markers=[],
        citation_validation_failed=False,
        timings=RAGTimings(retrieval_ms=1.0, rerank_ms=1.0, generation_ms=1.0, total_ms=3.0),
        candidate_count=0,
        reranked_count=0,
        model_name="mock",
    )
    assert evaluator.compute_abstention_correctness(example_unans, good_resp) == 1.0

    # 2. Failed refusal: falsely cited a document
    bad_resp = RAGResponse(
        query="Kubernetes ingress controller configuration?",
        answer="Kubernetes ingress routes traffic [1].",
        citations=[CitationReference(index=1, chunk_id="c1", source="TTL.md", snippet="TTL")],
        retrieved_citations=[CitationReference(index=1, chunk_id="c1", source="TTL.md", snippet="TTL")],
        cited_citations=[CitationReference(index=1, chunk_id="c1", source="TTL.md", snippet="TTL")],
        hallucinated_markers=[],
        citation_validation_failed=False,
        timings=RAGTimings(retrieval_ms=1.0, rerank_ms=1.0, generation_ms=1.0, total_ms=3.0),
        candidate_count=1,
        reranked_count=1,
        model_name="mock",
    )
    assert evaluator.compute_abstention_correctness(example_unans, bad_resp) == 0.0


def test_evaluate_dataset_and_report_generation(evaluator: RagasEvaluator) -> None:
    # Build minimal in-memory service
    embedder = MockEmbeddingModel(384)
    vstore = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vstore, bm25)
    reranker = MockReranker()
    llm = MockLLMClient()
    verifier = MockGroundingVerifier()

    service = RAGService(retriever, reranker, llm, verifier)

    examples = [
        EvalExample(
            example_id="test_001",
            question="What is TTL?",
            reference_answer="Time to Live deletes expired items.",
            essential_facts=["Deletes expired items"],
            expected_evidence={"source": "TTL.md", "excerpt": "deletes expired items"},
            category="direct_factual",
            answerability="answerable",
        ),
        EvalExample(
            example_id="test_002",
            question="Kubernetes cluster setup?",
            reference_answer="Outside scope.",
            category="unanswerable",
            answerability="unanswerable",
        ),
    ]

    report = evaluator.evaluate_dataset(examples, service)
    assert report.total_examples == 2
    assert report.answerable_count == 1
    assert report.unanswerable_count == 1
    assert len(report.category_summaries) == 2

    # Verify markdown generation
    md = format_markdown_report(report)
    assert "# ResolveOps RAGAS Offline Benchmark Report" in md
    assert "test_001" in md
    assert "test_002" in md

