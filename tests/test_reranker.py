import math
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.reranker import (
    MockReranker,
    NVIDIAReranker,
    RerankResult,
    _sigmoid,
    make_reranker,
)


def _make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-test",
        source="test_manual.pdf",
        source_type=SourceType.PDF,
        document_title="AWS Manual",
        kind="text",
        content=content,
        block_ids=["b1"],
        context_block_ids=[],
        heading_context=["Overview"],
        page_numbers=[1],
        indexable=True,
    )


def test_sigmoid_math_and_boundaries():
    """Verify that _sigmoid maps raw logits to bounded [0, 1] probabilities."""
    assert _sigmoid(0.0) == 0.5
    assert pytest.approx(_sigmoid(2.0), abs=1e-4) == 0.880797
    assert pytest.approx(_sigmoid(-2.0), abs=1e-4) == 0.119202
    # Overflow protection
    assert _sigmoid(1000.0) == 1.0
    assert _sigmoid(-1000.0) == 0.0


def test_reranker_empty_and_invalid_inputs():
    """Verify edge cases: empty query, empty chunk list, and invalid top_k."""
    reranker = MockReranker()

    assert reranker.rerank("", [_make_chunk("c1", "hello")]) == []
    assert reranker.rerank("   ", [_make_chunk("c1", "hello")]) == []
    assert reranker.rerank("query", []) == []

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        reranker.rerank("query", [_make_chunk("c1", "hello")], top_k=0)


def test_mock_reranker_rank_promotion_and_audit_metadata():
    """Verify that a relevant chunk at initial rank #3 gets promoted to #1 with accurate audit fields."""
    reranker = MockReranker(target_boost_keyword="dynamodb")

    chunks = [
        _make_chunk("c1", "Amazon S3 bucket policies and permissions."),
        _make_chunk("c2", "AWS Lambda async invocation retries."),
        _make_chunk("c3", "Amazon DynamoDB item size limits and partition throughput."),
    ]

    # Initial order: c1 (#1), c2 (#2), c3 (#3)
    results = reranker.rerank(query="what are DynamoDB item limits?", chunks=chunks, top_k=2)

    print("\n" + "=" * 70)
    print("🔬 [UNIT TEST] MOCK RERANKER PROMOTION AUDIT:")
    print("=" * 70)
    for res in results:
        print(f"  🏆 Final Rank #{res.reranked_rank} (Initial Rank #{res.initial_rank})")
        print(f"     • Chunk ID: {res.chunk.id} | Score (Prob): {res.score:.4f} | Logit: {res.logit:.4f}")
        print(f"     • Content:  {res.chunk.content[:70]}...")
    print("=" * 70 + "\n")

    assert len(results) == 2

    # c3 must be promoted from initial rank #3 straight to final rank #1!
    winner = results[0]
    assert winner.chunk.id == "c3"
    assert winner.initial_rank == 3
    assert winner.reranked_rank == 1
    assert winner.score > results[1].score
    assert winner.score == pytest.approx(1.0 / (1.0 + math.exp(-winner.logit)), abs=1e-4)


def test_nvidia_reranker_payload_and_parsing_with_mock_api():
    """Verify NVIDIAReranker builds correct payload and maps returned rankings."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "rankings": [
            {"index": 1, "logit": 3.5},
            {"index": 0, "logit": -1.2},
        ],
        "usage": {"total_tokens": 50},
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        reranker = NVIDIAReranker(api_key="nvapi-mock-key")
        chunks = [
            _make_chunk("c1", "Irrelevant passage."),
            _make_chunk("c2", "Highly relevant target passage."),
        ]

        results = reranker.rerank("target query", chunks, top_k=2)

        # 1. Verify API request structure
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == NVIDIAReranker.DEFAULT_MODEL
        assert call_kwargs["json"]["query"] == {"text": "target query"}
        assert len(call_kwargs["json"]["passages"]) == 2
        assert call_kwargs["headers"]["Authorization"] == "Bearer nvapi-mock-key"

        # 2. Verify results mapping (index 1 was ranked first)
        assert len(results) == 2
        assert results[0].chunk.id == "c2"
        assert results[0].initial_rank == 2
        assert results[0].reranked_rank == 1
        assert results[0].logit == 3.5
        assert results[0].score == pytest.approx(_sigmoid(3.5), abs=1e-4)

        assert results[1].chunk.id == "c1"
        assert results[1].initial_rank == 1
        assert results[1].reranked_rank == 2


def test_make_reranker_factory():
    """Verify factory routes prefixes properly."""
    mock = make_reranker("mock")
    assert isinstance(mock, MockReranker)

    nv = make_reranker("nvidia:nvidia/nemotron-3.5-rerank-8b", api_key="test-key")
    assert isinstance(nv, NVIDIAReranker)
    assert nv.model_name == "nvidia/nemotron-3.5-rerank-8b"

    with pytest.raises(ValueError, match="Unsupported reranker model"):
        make_reranker("unknown:model")


@pytest.mark.skipif(not os.environ.get("NVIDIA_API_KEY"), reason="Requires live NVIDIA_API_KEY")
def test_live_nvidia_reranker_cross_service_disambiguation():
    """
    Live Cloud Test on NVIDIA Nemotron 3.5 8B Reranker:
    Demonstrates solving the exact cross-service failure from our earlier benchmark:
    Query is about Lambda retries, but candidate pool includes both a Lambda doc and a DynamoDB doc.
    The Cross-Encoder MUST elevate the Lambda doc and demote the DynamoDB doc!
    """
    reranker = make_reranker("nvidia:nvidia/nemotron-3.5-rerank-8b")

    lambda_chunk = _make_chunk(
        "c_lambda",
        "Invoking a Lambda Event Function asynchronously. If the function fails, "
        "Lambda automatically retries the invocation twice by default with delays."
    )
    dynamo_chunk = _make_chunk(
        "c_dynamodb",
        "Error handling with DynamoDB. Batch operations and error retries. "
        "If DynamoDB returns unprocessed items, retry the operation with exponential backoff."
    )

    # Initial order: DynamoDB doc is #1, Lambda doc is #2 (simulating false positive from candidate retrieval)
    candidates = [dynamo_chunk, lambda_chunk]

    query = "what happens when a Lambda function fails and how are retries handled?"
    try:
        results = reranker.rerank(query=query, chunks=candidates, top_k=2)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        pytest.skip(f"Offline environment: cannot connect to NVIDIA NIM ({exc})")

    print("\n" + "=" * 78)
    print(f"🌐 [LIVE NVIDIA NEMOTRON 8B RERANKER TEST]")
    print(f"   Query: '{query}'")
    print("=" * 78)
    for res in results:
        print(f"  🏆 Final Rank #{res.reranked_rank} (Was Candidate #{res.initial_rank})")
        print(f"     • Chunk:       {res.chunk.id}")
        print(f"     • Logit:       {res.logit:+.4f}")
        print(f"     • Relevance:   {res.score * 100:.2f}%")
        print(f"     • Snippet:     {res.chunk.content[:85]}...")
    print("=" * 78 + "\n")

    assert len(results) == 2
    # The Lambda doc MUST win at Rank #1 over the DynamoDB doc!
    assert results[0].chunk.id == "c_lambda"
    assert results[0].reranked_rank == 1
    assert results[0].logit > results[1].logit

