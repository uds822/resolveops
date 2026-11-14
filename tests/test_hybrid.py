from pathlib import Path
import pytest

from resolveops.ingestion.chunking import Chunk, chunk_document
from resolveops.ingestion.document_parser import parse_document
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import EmbeddingModel, make_embedding_model
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.vector_store import InMemoryVectorStore


class MockEmbeddingModel(EmbeddingModel):
    """
    Predictable mock embedding returning orthogonal unit vectors.
    Lets us mathematically verify rank fusion without external dependencies.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)

    def _vector_for(self, text: str) -> list[float]:
        t = text.lower()
        if "dynamodb" in t or "nosql" in t:
            return [1.0, 0.0, 0.0]  # Vector space for DynamoDB / NoSQL concept
        if "s3" in t:
            return [0.0, 1.0, 0.0]  # Vector space for S3 concept
        return [0.0, 0.0, 1.0]      # Neutral catch-all vector


def _make_chunk(chunk_id: str, content: str, indexable: bool = True) -> Chunk:
    """Helper to construct valid Chunks for retrieval testing."""
    return Chunk(
        id=chunk_id,
        document_id="doc-aws-1",
        source="aws_manual.pdf",
        source_type=SourceType.PDF,
        document_title="AWS Architecture",
        kind="text",
        content=content,
        block_ids=["b1"],
        context_block_ids=[],
        heading_context=["Database Configuration"],
        page_numbers=[57],
        indexable=indexable,
    )


def test_hybrid_empty_retriever_and_edge_cases():
    """Verify retriever handles empty queries, empty index, and invalid arguments gracefully."""
    vector_store = InMemoryVectorStore(MockEmbeddingModel())
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    assert retriever.count() == 0
    assert retriever.search("anything") == []
    assert retriever.search("") == []
    assert retriever.search("   ") == []

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search("test", top_k=0)


def test_hybrid_dual_indexing_and_quarantine_enforcement():
    """Verify add_chunks indexes both stores in lockstep and respects quarantine."""
    vector_store = InMemoryVectorStore(MockEmbeddingModel())
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    chunks = [
        _make_chunk("c1", "Amazon DynamoDB item size limit is 400 KB."),
        _make_chunk("c2", "Amazon S3 bucket policy troubleshooting."),
        _make_chunk("c3_bad", "Malformed table without headers.", indexable=False),
    ]

    added = retriever.add_chunks(chunks)

    # 1. Quarantined chunk c3_bad must be skipped
    assert added == 2
    assert retriever.count() == 2
    assert vector_store.count() == 2
    assert bm25.count() == 2

    # 2. Idempotency: Adding same chunks again must add 0
    assert retriever.add_chunks(chunks) == 0


def test_rrf_scoring_boosts_chunks_matched_by_both_engines():
    """
    Mathematical Proof Test for Reciprocal Rank Fusion:
    - Chunk A has BOTH semantic relevance ('dynamodb') AND exact keyword ('pom.xml').
    - Chunk B has ONLY semantic relevance ('dynamodb partition keys').
    - Chunk C has ONLY exact keyword ('update your pom.xml build file').

    Because Chunk A is corroborated by BOTH engines, RRF must place it at Rank #1.
    """
    vector_store = InMemoryVectorStore(MockEmbeddingModel())
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25, rrf_k=60)

    chunk_a = _make_chunk("chunk_a", "DynamoDB client configuration in pom.xml file.")
    chunk_b = _make_chunk("chunk_b", "NoSQL table partition key throughput limits.")
    chunk_c = _make_chunk("chunk_c", "Update Java pom.xml for maven projects.")

    retriever.add_chunks([chunk_a, chunk_b, chunk_c])

    # Search for "dynamodb pom.xml"
    results = retriever.search(query="dynamodb pom.xml", top_k=3)

    assert len(results) == 3

    # Winner must be Chunk A because it scored in BOTH Dense and BM25
    winner = results[0]
    assert winner.chunk.id == "chunk_a"
    assert winner.dense_rank is not None
    assert winner.bm25_rank is not None

    # Calculate exact expected RRF math:
    # 1 / (60 + dense_rank) + 1 / (60 + bm25_rank)
    expected_score = (1.0 / (60 + winner.dense_rank)) + (1.0 / (60 + winner.bm25_rank))
    assert pytest.approx(winner.score, abs=1e-5) == expected_score


def test_hybrid_bm25_weight_tuning():
    """Verify that boosting bm25_weight gives keyword matches higher priority over semantic matches."""
    vector_store = InMemoryVectorStore(MockEmbeddingModel())
    bm25 = BM25Index()

    # Emphasize BM25 by a factor of 3.0
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25,
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=3.0,
    )

    chunk_dense_only = _make_chunk("c_dense", "DynamoDB database internals.")
    chunk_exact_keyword = _make_chunk("c_keyword", "Version 3.3.2 release notes pom.xml.")

    retriever.add_chunks([chunk_dense_only, chunk_exact_keyword])

    # Query for exact version "3.3.2 pom.xml"
    results = retriever.search("3.3.2 pom.xml", top_k=2)

    assert len(results) == 2
    # The exact keyword chunk must dominate because bm25_weight is 3.0
    assert results[0].chunk.id == "c_keyword"
    assert results[0].bm25_rank == 1


def test_hybrid_retriever_on_real_aws_documentation():
    """
    Live Engineering Benchmark on Real AWS Documentation:
    Ingests 'Programming.Errors.md' and 'TTL.md' using our real parser & chunker,
    embeds with all-MiniLM-L6-v2, and executes both technical and conceptual queries.
    Prints an engineering audit table to stdout (view with `pytest -s`).
    """
    error_doc_path = Path("data/raw/aws/markdown/Programming.Errors.md")
    ttl_doc_path = Path("data/raw/aws/markdown/TTL.md")

    assert error_doc_path.exists(), f"Missing test data file: {error_doc_path}"
    assert ttl_doc_path.exists(), f"Missing test data file: {ttl_doc_path}"

    # 1. Parse and chunk real documents
    doc_errors = parse_document(error_doc_path)
    doc_ttl = parse_document(ttl_doc_path)
    real_chunks = chunk_document(doc_errors) + chunk_document(doc_ttl)

    # 2. Initialize real embedding model and hybrid retriever
    embedder = make_embedding_model("all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25, rrf_k=60)

    # 3. Index real chunks
    added_count = retriever.add_chunks(real_chunks)
    assert added_count == len(real_chunks)

    # 4. Benchmark Query 1: Exact technical error specification
    q1 = "AccessDeniedException 400 Signature Version 4"
    r1 = retriever.search(q1, top_k=2)

    print("\n" + "=" * 78)
    print(f"🔬 [REAL DOC TEST 1] EXACT TECHNICAL QUERY: '{q1}'")
    print("=" * 78)
    for rank, res in enumerate(r1, 1):
        print(f"  🏆 Result #{rank} [Fused RRF Score: {res.score:.6f}]")
        print(f"     • Dense Rank: {res.dense_rank} (Cosine Sim: {res.dense_score})")
        print(f"     • BM25 Rank:  {res.bm25_rank} (BM25 Okapi: {res.bm25_score})")
        print(f"     • Context:    {' > '.join(res.chunk.heading_context)}")
        print(f"     • Preview:    {res.chunk.content.splitlines()[0]}")
        print(f"                   {res.chunk.content.splitlines()[-1][:90]}...")
    print("-" * 78)

    # Assert Rank #1 found the exact AccessDeniedException block
    assert len(r1) > 0
    winner_1 = r1[0]
    assert "AccessDeniedException" in winner_1.chunk.content
    assert winner_1.bm25_rank == 1

    # 5. Benchmark Query 2: Conceptual query (No direct mention of 'TTL')
    q2 = "How to automatically expire and delete old items without consuming write throughput?"
    r2 = retriever.search(q2, top_k=2)

    print("\n" + "=" * 78)
    print(f"🧠 [REAL DOC TEST 2] CONCEPTUAL QUERY: '{q2}'")
    print("=" * 78)
    for rank, res in enumerate(r2, 1):
        print(f"  🏆 Result #{rank} [Fused RRF Score: {res.score:.6f}]")
        print(f"     • Dense Rank: {res.dense_rank} (Cosine Sim: {res.dense_score})")
        print(f"     • BM25 Rank:  {res.bm25_rank} (BM25 Okapi: {res.bm25_score})")
        print(f"     • Context:    {' > '.join(res.chunk.heading_context)}")
        print(f"     • Preview:    {res.chunk.content.splitlines()[0]}")
        print(f"                   {res.chunk.content.splitlines()[-1][:90]}...")
    print("=" * 78 + "\n")

    # Assert Rank #1 found the TTL document
    assert len(r2) > 0
    winner_2 = r2[0]
    assert "Time To Live" in winner_2.chunk.content or "TTL" in winner_2.chunk.content
    assert winner_2.dense_rank == 1


def test_hybrid_batch_deduplication_preserves_lockstep():
    """Verify that adding duplicate chunks in one batch only adds 1 entry to both stores."""
    vector_store = InMemoryVectorStore(MockEmbeddingModel())
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("chunk_dup", "Same exact chunk content")
    added = retriever.add_chunks([c1, c1, c1])

    assert added == 1
    assert retriever.count() == 1
    assert vector_store.count() == 1
    assert bm25.count() == 1


class BrokenEmbeddingModel(EmbeddingModel):
    """Simulates an embedding model that returns fewer vectors than chunks."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Always returns 1 vector regardless of how many texts were sent
        return [[1.0, 0.0, 0.0]]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def test_hybrid_embedding_count_mismatch_fails_and_rolls_back():
    """Verify that embedding length mismatch raises ValueError and leaves both stores empty."""
    vector_store = InMemoryVectorStore(BrokenEmbeddingModel())
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("chunk_1", "First chunk")
    c2 = _make_chunk("chunk_2", "Second chunk")

    with pytest.raises(ValueError, match="Embedding count mismatch"):
        retriever.add_chunks([c1, c2])

    assert retriever.count() == 0
    assert vector_store.count() == 0
    assert bm25.count() == 0


class NanEmbeddingModel(EmbeddingModel):
    """Simulates an embedding model that returns NaNs."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float("nan"), 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def test_hybrid_nan_embedding_validation_failure():
    """Verify that non-finite embedding vectors raise ValueError and roll back."""
    vector_store = InMemoryVectorStore(NanEmbeddingModel())
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("chunk_nan", "Chunk with bad vector")
    with pytest.raises(ValueError, match="contains non-finite values"):
        retriever.add_chunks([c1])

    assert retriever.count() == 0
    assert vector_store.count() == 0
    assert bm25.count() == 0


def test_hybrid_delete_document_and_chunks():
    """Verify deletion removes chunks from both stores in perfect sync."""
    vector_store = InMemoryVectorStore(MockEmbeddingModel())
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("chunk_a", "DynamoDB tables")
    c2 = _make_chunk("chunk_b", "S3 buckets")
    retriever.add_chunks([c1, c2])
    assert retriever.count() == 2

    # Delete 1 chunk
    removed = retriever.delete_chunks(["chunk_a"])
    assert removed == 1
    assert retriever.count() == 1
    assert vector_store.count() == 1
    assert bm25.count() == 1

    # Delete by document
    doc_removed = retriever.delete_document("doc-aws-1")
    assert doc_removed == 1
    assert retriever.count() == 0
    assert vector_store.count() == 0
    assert bm25.count() == 0


class DynamicDimensionEmbedder(EmbeddingModel):
    """Returns vectors with dynamic dimensionality based on call."""
    def __init__(self, dims: list[int]):
        self.dims = dims
        self.call_idx = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        dim = self.dims[self.call_idx]
        self.call_idx = min(self.call_idx + 1, len(self.dims) - 1)
        return [[1.0] * dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0] * self.dims[0]


def test_multi_batch_dimension_mismatch_rejected():
    """Verify that a second batch with different vector dimension is rejected."""
    # First batch returns 3D vectors, second batch returns 2D vectors
    embedder = DynamicDimensionEmbedder(dims=[3, 2])
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c1 = _make_chunk("c1", "Batch 1 content")
    retriever.add_chunks([c1])
    assert retriever.count() == 1

    c2 = _make_chunk("c2", "Batch 2 content with different dimensions")
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        retriever.add_chunks([c2])

    # First batch must still be intact
    assert retriever.count() == 1
    assert vector_store.count() == 1
    assert bm25.count() == 1


def test_failed_replacement_preserves_existing_document():
    """Verify that if replacement fails during embedding, the existing document is preserved."""
    # First call succeeds with 3D vector, second call fails with length mismatch
    class FlakyEmbedder(EmbeddingModel):
        def __init__(self):
            self.first = True
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if self.first:
                self.first = False
                return [[1.0, 0.0, 0.0]]
            raise RuntimeError("Simulated embedding outage")
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    vector_store = InMemoryVectorStore(FlakyEmbedder())
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25)

    c_orig = _make_chunk("orig_chunk", "Original document content")
    retriever.add_chunks([c_orig])
    assert retriever.count() == 1

    # Attempt to replace document, but embedding fails
    c_new = _make_chunk("new_chunk", "New document content")
    with pytest.raises(RuntimeError, match="Simulated embedding outage"):
        retriever.replace_document("doc-aws-1", [c_new])

    # Crucial assertion: original document was NOT deleted!
    assert retriever.count() == 1
    assert vector_store.count() == 1
    assert bm25.count() == 1
    assert "orig_chunk" in vector_store._chunks


