"""Unit and integration tests for FastAPI routes using TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from resolveops.api.main import create_app
from resolveops.generation.llm import MockLLMClient
from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import EmbeddingModel
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.reranker import MockReranker
from resolveops.retrieval.vector_store import InMemoryVectorStore
from resolveops.service import RAGService


class MockEmbeddingModel(EmbeddingModel):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        t = text.lower()
        if "iam" in t or "role" in t:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


@pytest.fixture
def client() -> TestClient:
    """Fixture creating a test client with isolated mock retrieval dependencies."""
    embedder = MockEmbeddingModel()
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)
    reranker = MockReranker(target_boost_keyword="iam")
    llm_client = MockLLMClient(model_name="mock-deepseek")

    service = RAGService(retriever, reranker, llm_client)
    app = create_app(rag_service=service, retriever=retriever)
    return TestClient(app)


def test_api_health_check(client: TestClient) -> None:
    """Test GET /health endpoint returns healthy status and chunk count."""
    response = client.get("/health")

    print("\n" + "═" * 70)
    print("TEST: GET /health")
    print("═" * 70)
    print("Response JSON:", response.json())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ResolveOps RAG API"
    assert "indexed_chunks" in data


def test_api_ingest_and_query_flow(client: TestClient) -> None:
    """Test end-to-end HTTP API lifecycle: Ingest documentation -> Query -> Verified Answer."""
    # 1. Ingest documentation text snippet
    ingest_payload = {
        "source": "id_roles.md",
        "content": "An IAM role is an IAM identity that you can create in your account with temporary credentials.",
        "document_title": "IAM User Guide",
        "heading_context": ["IAM roles", "Roles terms and concepts"],
        "page_numbers": [30],
    }

    ingest_res = client.post("/api/v1/ingest", json=ingest_payload)

    print("\n" + "═" * 70)
    print("TEST: POST /api/v1/ingest")
    print("═" * 70)
    print("Ingest Response:", ingest_res.json())

    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"
    assert ingest_res.json()["chunks_added"] == 1
    assert ingest_res.json()["total_indexed"] == 1

    # 2. Check health updated
    health_res = client.get("/health")
    assert health_res.json()["indexed_chunks"] == 1

    # 3. Query the ingested documentation
    query_payload = {
        "query": "What are IAM roles and how do credentials work?",
        "candidate_pool": 5,
        "top_k": 2,
    }

    query_res = client.post("/api/v1/query", json=query_payload)

    print("\n" + "═" * 70)
    print("TEST: POST /api/v1/query")
    print("═" * 70)
    print("Query Response JSON:")
    print(f"  Answer: {query_res.json()['answer']}")
    print(f"  Citations: {query_res.json()['citations']}")
    print(f"  Timings: {query_res.json()['timings']}")

    assert query_res.status_code == 200
    rag_data = query_res.json()
    assert "[1]" in rag_data["answer"]
    assert len(rag_data["citations"]) == 1
    assert rag_data["citations"][0]["source"] == "id_roles.md"
    assert rag_data["citations"][0]["page_numbers"] == [30]
    assert rag_data["candidate_count"] == 1
    assert rag_data["reranked_count"] == 1


def test_api_empty_query_validation(client: TestClient) -> None:
    """Test that empty query strings return 400 Bad Request."""
    res = client.post("/api/v1/query", json={"query": "   "})
    print("\n" + "═" * 70)
    print("TEST: POST /api/v1/query (Validation Error)")
    print("═" * 70)
    print("Status:", res.status_code)
    print("Detail:", res.json())

    assert res.status_code == 400
    assert "Query cannot be empty" in res.json()["detail"]


def test_api_empty_ingest_validation(client: TestClient) -> None:
    """Test that empty content strings return 400 Bad Request."""
    res = client.post("/api/v1/ingest", json={"source": "test.md", "content": "  "})
    print("\n" + "═" * 70)
    print("TEST: POST /api/v1/ingest (Validation Error)")
    print("═" * 70)
    print("Status:", res.status_code)

    assert res.status_code == 400
    assert "Content cannot be empty" in res.json()["detail"]


def test_api_deterministic_ids_and_replacement(client: TestClient) -> None:
    """Test that ingestion generates deterministic IDs and replaces stale chunks."""
    payload = {
        "source": "s3_guide.md",
        "content": "Amazon S3 provides 99.999999999% durability.",
        "heading_context": ["S3 Overview"],
        "page_numbers": [1],
    }

    # First ingestion
    res1 = client.post("/api/v1/ingest", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["chunks_added"] == 1
    assert data1["total_indexed"] == 1
    chunk_id_1 = data1["chunk_id"]
    doc_id_1 = data1["document_id"]

    # Ingest same exact payload again -> with replace=True, deletes old and adds new with SAME chunk_id
    res2 = client.post("/api/v1/ingest", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["chunk_id"] == chunk_id_1  # 100% deterministic SHA-256 ID!
    assert data2["document_id"] == doc_id_1
    assert data2["chunks_deleted"] == 1
    assert data2["chunks_added"] == 1
    assert data2["total_indexed"] == 1  # No duplicate proliferation!

    # Update document with new content -> replaces old chunk
    payload_v2 = {
        "source": "s3_guide.md",
        "content": "Amazon S3 provides 99.999999999% (11 9s) of durability and Glacier archival.",
        "heading_context": ["S3 Durability"],
        "page_numbers": [1, 2],
    }
    res3 = client.post("/api/v1/ingest", json=payload_v2)
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["chunks_deleted"] == 1
    assert data3["chunks_added"] == 1
    assert data3["total_indexed"] == 1

    # Delete document endpoint
    del_res = client.delete(f"/api/v1/documents/{doc_id_1}")
    assert del_res.status_code == 200
    assert del_res.json()["chunks_deleted"] == 1
    assert del_res.json()["total_indexed"] == 0


