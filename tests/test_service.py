"""Unit tests for the end-to-end RAGService orchestrator."""

from __future__ import annotations

import os

import pytest
import requests

from resolveops.generation.llm import MockLLMClient, OpenRouterLLMClient
from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import EmbeddingModel
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.reranker import MockReranker
from resolveops.retrieval.vector_store import InMemoryVectorStore
from resolveops.service import RAGResponse, RAGService, RAGTimings


class MockEmbeddingModel(EmbeddingModel):
    """Deterministic offline embedder mapping keywords to orthogonal dimensions."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        t = text.lower()
        if "lambda" in t or "async" in t:
            return [1.0, 0.0, 0.0]
        if "dynamodb" in t or "nosql" in t:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def _make_dummy_chunk(
    chunk_id: str,
    source: str,
    content: str,
    headings: list[str] | None = None,
    pages: list[int] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source=source,
        source_type=SourceType.PDF if source.endswith(".pdf") else SourceType.MARKDOWN,
        document_title="AWS Service Manual",
        kind="text",
        content=content,
        block_ids=[f"block_{chunk_id}"],
        context_block_ids=[],
        heading_context=headings or [],
        page_numbers=pages or [],
        indexable=True,
    )


def test_rag_service_end_to_end_offline_pipeline() -> None:
    """Test full execution of RAGService with Mock components: Retrieve -> Rerank -> Prompt -> Generate."""
    embedder = MockEmbeddingModel()
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25, rrf_k=60)

    chunks = [
        _make_dummy_chunk(
            chunk_id="chunk_lambda",
            source="invocation-async.md",
            content="When an asynchronous Lambda function fails, Lambda retries the invocation twice by default.",
            headings=["Invoking a Lambda Event Function asynchronously", "Error handling"],
        ),
        _make_dummy_chunk(
            chunk_id="chunk_dynamo",
            source="dynamodb-dg.pdf",
            content="DynamoDB Local version 3.3.2 upgraded dependencies in pom.xml.",
            headings=["Release history for DynamoDB local"],
            pages=[58],
        ),
        _make_dummy_chunk(
            chunk_id="chunk_s3",
            source="s3-userguide.pdf",
            content="By default, all Amazon S3 resources (buckets and objects) are private to the resource owner.",
            headings=["Access management in Amazon S3"],
            pages=[32],
        ),
    ]

    retriever.add_chunks(chunks)
    reranker = MockReranker(target_boost_keyword="lambda")
    llm_client = MockLLMClient(model_name="mock-deepseek")

    service = RAGService(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
    )

    query = "How does Lambda handle errors during asynchronous invocations?"
    response = service.answer_query(query=query, candidate_pool=5, top_k=2)

    print("\n" + "═" * 80)
    print("🔬 [UNIT TEST] RAGService Full Pipeline Execution")
    print("═" * 80)
    print(f"Query: {response.query}")
    print(f"Answer:\n{response.answer}")
    print(f"\nModel: {response.model_name}")
    print(f"Candidates Retrieved: {response.candidate_count}  │  Reranked Passed to LLM: {response.reranked_count}")
    print(f"Citations ({len(response.citations)}):")
    for cit in response.citations:
        print(f"   [{cit.index}] {cit.source} (Pages: {cit.page_numbers}) ▸ {' ▸ '.join(cit.heading_context)}")
        print(f"       Snippet: {cit.snippet}")
    print(f"\nLatency Profile:")
    print(f"   ├─ Hybrid Retrieval : {response.timings.retrieval_ms:.2f} ms")
    print(f"   ├─ Reranking        : {response.timings.rerank_ms:.2f} ms")
    print(f"   ├─ Generation       : {response.timings.generation_ms:.2f} ms")
    print(f"   └─ Total End-to-End : {response.timings.total_ms:.2f} ms")
    print("═" * 80 + "\n")

    assert response.query == query
    assert len(response.citations) == 2
    assert "[1]" in response.answer
    assert response.citations[0].source == "invocation-async.md"
    assert response.timings.total_ms >= 0.0
    assert response.candidate_count == 3
    assert response.reranked_count == 2


def test_rag_service_empty_corpus_refusal() -> None:
    """Test that querying an empty retriever safely yields the refusal message without crashing."""
    embedder = MockEmbeddingModel()
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)
    reranker = MockReranker()
    llm_client = MockLLMClient()

    service = RAGService(retriever, reranker, llm_client)
    response = service.answer_query("How to configure multi-region replication?")

    print("\n" + "═" * 80)
    print("TEST: RAGService on empty index")
    print("═" * 80)
    print(f"Answer: {response.answer}")

    assert "cannot find enough information" in response.answer
    assert response.candidate_count == 0
    assert response.reranked_count == 0
    assert len(response.citations) == 0


def test_rag_service_empty_query_raises_value_error() -> None:
    """Test that whitespace or empty query raises ValueError."""
    embedder = MockEmbeddingModel()
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)
    service = RAGService(retriever, MockReranker(), MockLLMClient())

    with pytest.raises(ValueError, match="query cannot be empty"):
        service.answer_query("   ")


def test_rag_service_live_deepseek_generation() -> None:
    """Live test calling OpenRouter DeepSeek V3 in end-to-end pipeline (skipped in offline sandbox)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not configured in environment")

    embedder = MockEmbeddingModel()
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    chunk = _make_dummy_chunk(
        chunk_id="chunk_lambda_live",
        source="invocation-async.md",
        content="When an asynchronous function fails, Lambda attempts to run it two more times, with a delay between attempts.",
        headings=["Invoking a Lambda Event Function asynchronously", "Error handling"],
    )
    retriever.add_chunks([chunk])

    reranker = MockReranker()
    llm_client = OpenRouterLLMClient(api_key=api_key, model_name="deepseek/deepseek-chat")

    service = RAGService(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
    )

    query = "How many times does Lambda retry an asynchronous function upon error?"
    try:
        response = service.answer_query(query=query, candidate_pool=2, top_k=1)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        pytest.skip(f"Live DeepSeek test skipped (network unavailable in sandbox): {exc}")

    print("\n" + "═" * 80)
    print("TEST: Live DeepSeek V3 RAGService End-to-End Generation")
    print("═" * 80)
    print(f"Model: {response.model_name}")
    print(f"Answer:\n{response.answer}")
    print(f"Latency Profile: Total={response.timings.total_ms:.1f}ms (Retrieval={response.timings.retrieval_ms:.1f}ms, Rerank={response.timings.rerank_ms:.1f}ms, LLM={response.timings.generation_ms:.1f}ms)")
    print(f"Citations: {len(response.citations)}")

    assert len(response.answer) > 10
    assert "[1]" in response.answer
    assert "two" in response.answer.lower() or "2" in response.answer

