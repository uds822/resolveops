import pytest

from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.embeddings import EmbeddingModel
from resolveops.retrieval.vector_store import InMemoryVectorStore, SearchResult


class DeterministicMockEmbedding(EmbeddingModel):
    """
    Mock model with predictable vectors so we can mathematically
    verify that cosine similarity ranking works.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)

    def _vector_for(self, text: str) -> list[float]:
        # Gives distinct orthogonal vectors based on keywords
        t = text.lower()
        if "dynamodb" in t or "size" in t:
            return [1.0, 0.0, 0.0]
        if "partition" in t:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def _make_chunk(chunk_id: str, content: str, indexable: bool = True) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        source="test.pdf",
        source_type=SourceType.PDF,
        document_title="Test Guide",
        kind="text",
        content=content,
        block_ids=["b1"],
        context_block_ids=[],
        heading_context=["DynamoDB Limits"],
        page_numbers=[58],
        indexable=indexable,
    )


def test_empty_store_returns_empty_list():
    store = InMemoryVectorStore(DeterministicMockEmbedding())
    assert store.search("anything") == []
    assert store.count() == 0


def test_add_chunks_skips_unindexable_and_duplicates():
    store = InMemoryVectorStore(DeterministicMockEmbedding())
    good_chunk = _make_chunk("c1", "DynamoDB item size is 400 KB.", indexable=True)
    bad_chunk = _make_chunk("c2", "Corrupted table", indexable=False)

    # 1. Add chunks (bad_chunk should be quarantined!)
    added = store.add_chunks([good_chunk, bad_chunk])
    assert added == 1
    assert store.count() == 1

    # 2. Add good_chunk again (should be skipped as duplicate!)
    added_again = store.add_chunks([good_chunk])
    assert added_again == 0
    assert store.count() == 1


def test_search_ranks_most_similar_chunk_first():
    store = InMemoryVectorStore(DeterministicMockEmbedding())
    c1 = _make_chunk("c1", "DynamoDB item size is 400 KB.")
    c2 = _make_chunk("c2", "Partition key limits.")
    c3 = _make_chunk("c3", "Unrelated cooking recipe.")

    store.add_chunks([c1, c2, c3])

    # Search for item size (should match c1 with score = 1.0)
    results = store.search("item size", top_k=2)

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk.id == "c1"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].chunk.id != "c1"