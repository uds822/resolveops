import pytest

from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.bm25 import BM25Index, technical_tokenize


def _make_chunk(chunk_id: str, content: str, indexable: bool = True) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        source="dynamodb-dg.pdf",
        source_type=SourceType.PDF,
        document_title="DynamoDB Guide",
        kind="text",
        content=content,
        block_ids=["b1"],
        context_block_ids=[],
        heading_context=["Release Notes"],
        page_numbers=[58],
        indexable=indexable,
    )


def test_technical_tokenize_preserves_version_and_filename():
    tokens = technical_tokenize("Updated version 3.3.2 in pom.xml and CVE-2024-1234.")
    assert "3.3.2" in tokens
    assert "pom.xml" in tokens
    assert "cve-2024-1234" in tokens


def test_bm25_exact_version_match_ranks_first():
    index = BM25Index()

    c1 = _make_chunk("c1", "Version 3.3.2 updated AWS Java SDK dependencies.")
    c2 = _make_chunk("c2", "Version 3.3.1 added PartiQL support in pom.xml.")
    c3 = _make_chunk("c3", "DynamoDB local general overview.")

    index.add_chunks([c1, c2, c3])

    # Search specifically for version 3.3.2
    results = index.search("What happened in version 3.3.2?", top_k=2)

    assert len(results) > 0
    assert results[0].chunk.id == "c1"
    assert results[0].score > 0.0


def test_bm25_skips_unindexable_quarantined_chunks():
    index = BM25Index()

    good_chunk = _make_chunk("c1", "Valid chunk content.", indexable=True)
    bad_chunk = _make_chunk("c2", "Corrupted table.", indexable=False)

    added = index.add_chunks([good_chunk, bad_chunk])
    assert added == 1
    assert index.count() == 1


def test_bm25_empty_index_returns_empty_list():
    index = BM25Index()
    assert index.search("anything") == []