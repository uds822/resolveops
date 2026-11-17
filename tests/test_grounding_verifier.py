"""Unit tests for semantic claim-level grounding and citation entailment verifier."""

from __future__ import annotations

from typing import Sequence
import pytest

from resolveops.generation.context_builder import CitationReference
from resolveops.generation.grounding_verifier import (
    ClaimStatus,
    GroundingAuditReport,
    LLMGroundingVerifier,
    MockGroundingVerifier,
    extract_cited_claims,
    make_grounding_verifier,
)
from resolveops.generation.llm import GenerationResult, MockLLMClient
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import make_embedding_model
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.reranker import MockReranker
from resolveops.retrieval.vector_store import InMemoryVectorStore
from resolveops.service import RAGService


def _dummy_citations() -> list[CitationReference]:
    return [
        CitationReference(
            index=1,
            chunk_id="chunk_01",
            source="dynamodb-dg.pdf",
            heading_context=["Global Tables", "Conflict Resolution"],
            page_numbers=[45],
            snippet="DynamoDB uses last-writer-wins reconciliation between concurrent updates.",
            content="In global tables, DynamoDB uses last-writer-wins reconciliation between concurrent updates.",
        ),
        CitationReference(
            index=2,
            chunk_id="chunk_02",
            source="Programming.Errors.md",
            heading_context=["Collection Limits"],
            page_numbers=[],
            snippet="An ItemCollectionSizeLimitExceededException occurs when an item collection exceeds 10 GB.",
            content="An ItemCollectionSizeLimitExceededException occurs when an item collection exceeds 10 GB.",
        ),
    ]


def test_extract_cited_claims() -> None:
    citations = _dummy_citations()
    answer = (
        "DynamoDB uses last-writer-wins reconciliation for global tables [1].\n"
        "However, individual item sizes can reach up to 50 MB [2].\n"
        "This third sentence does not have any citations."
    )

    claims = extract_cited_claims(answer, citations)
    assert len(claims) == 2

    cid1, sent1, marker1, cit1 = claims[0]
    assert cid1 == 1
    assert marker1 == 1
    assert "last-writer-wins" in sent1
    assert cit1.source == "dynamodb-dg.pdf"

    cid2, sent2, marker2, cit2 = claims[1]
    assert cid2 == 2
    assert marker2 == 2
    assert "50 MB" in sent2
    assert cit2.source == "Programming.Errors.md"


def test_extract_cited_claims_empty_or_no_markers() -> None:
    citations = _dummy_citations()
    assert extract_cited_claims("", citations) == []
    assert extract_cited_claims("Just a normal text without brackets.", citations) == []
    assert extract_cited_claims("An out-of-bounds marker [99].", citations) == []


def test_mock_grounding_verifier_all_supported() -> None:
    citations = _dummy_citations()
    verifier = MockGroundingVerifier(default_status=ClaimStatus.SUPPORTED)

    answer = "DynamoDB uses last-writer-wins reconciliation [1]."
    report = verifier.verify_claims(answer, citations)

    assert report.total_claims == 1
    assert report.supported_claims == 1
    assert report.unsupported_claims == 0
    assert report.faithfulness_score == 1.0
    assert report.claims[0].status == ClaimStatus.SUPPORTED
    assert report.unsupported_details == []
    assert report.audit_status == "verified"


def test_mock_grounding_verifier_detects_unsupported() -> None:
    citations = _dummy_citations()
    verifier = MockGroundingVerifier(unsupported_keywords=["50 mb", "unsupported"])

    answer = (
        "DynamoDB uses last-writer-wins reconciliation [1]. "
        "The item size limit is 50 MB [2]."
    )
    report = verifier.verify_claims(answer, citations)

    assert report.total_claims == 2
    assert report.supported_claims == 1
    assert report.unsupported_claims == 1
    assert report.faithfulness_score == 0.5
    assert len(report.unsupported_details) == 1
    assert report.unsupported_details[0].cited_marker == 2
    assert report.unsupported_details[0].status == ClaimStatus.UNSUPPORTED


def test_llm_grounding_verifier_strategy_a_batch_success() -> None:
    """Test LLMGroundingVerifier with single batch judge response."""
    citations = _dummy_citations()

    judge_json_response = """[
      {
        "claim_id": 1,
        "status": "supported",
        "evidence_quote": "uses last-writer-wins reconciliation",
        "explanation": "Directly confirmed in source text."
      },
      {
        "claim_id": 2,
        "status": "unsupported",
        "evidence_quote": null,
        "explanation": "Source states collection limit is 10 GB, does not mention 50 MB item limits."
      }
    ]"""

    mock_llm = MockLLMClient(default_answer=judge_json_response)
    verifier = LLMGroundingVerifier(llm_client=mock_llm)

    answer = (
        "DynamoDB uses last-writer-wins reconciliation [1].\n"
        "Individual items can be up to 50 MB in size [2]."
    )
    report = verifier.verify_claims(answer, citations)

    assert report.total_claims == 2
    assert report.supported_claims == 1
    assert report.unsupported_claims == 1
    assert report.faithfulness_score == 0.5
    assert report.audit_status == "verified"
    assert report.claims[0].status == ClaimStatus.SUPPORTED
    assert report.claims[0].evidence_quote == "uses last-writer-wins reconciliation"
    assert report.claims[1].status == ClaimStatus.UNSUPPORTED


def test_llm_grounding_verifier_markdown_json_cleaning() -> None:
    """Test LLMGroundingVerifier cleanly strips ```json markdown fences."""
    citations = _dummy_citations()

    fenced_output = """```json
    [
      {
        "claim_id": 1,
        "status": "supported",
        "evidence_quote": "uses last-writer-wins reconciliation",
        "explanation": "Confirmed."
      }
    ]
    ```"""

    mock_llm = MockLLMClient(default_answer=fenced_output)
    verifier = LLMGroundingVerifier(llm_client=mock_llm)

    answer = "DynamoDB uses last-writer-wins reconciliation [1]."
    report = verifier.verify_claims(answer, citations)

    assert report.total_claims == 1
    assert report.supported_claims == 1
    assert report.faithfulness_score == 1.0
    assert report.audit_status == "verified"


def test_llm_grounding_verifier_malformed_json_fallback() -> None:
    """Test LLMGroundingVerifier handles malformed JSON without raising exceptions."""
    citations = _dummy_citations()

    broken_output = "I am an AI and I think Claim 1 is supported, but here is no JSON."
    mock_llm = MockLLMClient(default_answer=broken_output)
    verifier = LLMGroundingVerifier(llm_client=mock_llm)

    answer = "DynamoDB uses last-writer-wins reconciliation [1]."
    report = verifier.verify_claims(answer, citations)

    assert report.total_claims == 1
    assert report.audit_status == "parse_failed"
    assert report.claims[0].status == ClaimStatus.UNVERIFIED


class DummyEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_rag_service_end_to_end_with_grounding_audit() -> None:
    """Test full RAGService integration returning faithfulness_score and claim_audit."""
    from resolveops.ingestion.chunking import Chunk
    from resolveops.ingestion.models import SourceType

    embedder = DummyEmbeddingModel()
    vstore = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vstore, bm25)

    test_chunk = Chunk(
        id="c1",
        document_id="doc1",
        source="dynamodb.md",
        source_type=SourceType.MARKDOWN,
        document_title="DynamoDB Doc",
        kind="text",
        content="DynamoDB global tables replicate across multiple regions.",
        block_ids=["b1"],
        context_block_ids=[],
        heading_context=["Global Tables"],
        page_numbers=[],
        indexable=True,
    )
    retriever.add_chunks([test_chunk])

    reranker = MockReranker()
    llm = MockLLMClient(default_answer="DynamoDB global tables replicate across multiple regions [1].")
    verifier = MockGroundingVerifier(default_status=ClaimStatus.SUPPORTED)

    service = RAGService(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm,
        grounding_verifier=verifier,
    )

    response = service.answer_query("Explain DynamoDB replication")

    assert response.faithfulness_score == 1.0
    assert response.claim_audit is not None
    assert response.claim_audit.total_claims == 1
    assert response.claim_audit.supported_claims == 1
    assert response.citation_validation_failed is False
    assert response.timings.claim_audit_ms >= 0.0
