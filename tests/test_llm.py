"""Unit tests for the LLM client adapter layer."""

from __future__ import annotations

import os

import pytest
import requests

from resolveops.generation.context_builder import CitationReference, build_grounded_prompt
from resolveops.generation.llm import (
    GenerationResult,
    GroqLLMClient,
    LLMClient,
    MockLLMClient,
    NVIDIALLMClient,
    OpenRouterLLMClient,
    make_llm_client,
)
from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType


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
        document_title="AWS Technical Guide",
        kind="text",
        content=content,
        block_ids=[f"block_{chunk_id}"],
        context_block_ids=[],
        heading_context=headings or [],
        page_numbers=pages or [],
        indexable=True,
    )


def test_mock_llm_client_generates_grounded_answer_with_citations() -> None:
    """Test that MockLLMClient synthesizes an answer referencing all citation markers."""
    chunks = [
        _make_dummy_chunk(
            chunk_id="c1",
            source="invocation-async.md",
            content="Lambda retries failed asynchronous invocations twice by default.",
            headings=["Error retries"],
        ),
        _make_dummy_chunk(
            chunk_id="c2",
            source="dynamodb-dg.pdf",
            content="DynamoDB Local version 3.3.2 upgraded dependencies in pom.xml.",
            headings=["Release history"],
            pages=[58],
        ),
    ]

    query = "How does Lambda retry failures and what is DynamoDB version 3.3.2?"
    prompt = build_grounded_prompt(query=query, chunks=chunks)

    client = MockLLMClient()
    result = client.generate(prompt)

    print("\n" + "═" * 70)
    print("TEST: MockLLMClient grounded generation")
    print("═" * 70)
    print(f"Model Name: {result.model_name}")
    print(f"Generated Answer:\n{result.answer}")
    print(f"Citations Returned: {len(result.citations)}")
    print(f"Tokens Used: prompt={result.prompt_tokens}, completion={result.completion_tokens}, total={result.total_tokens}")
    print(f"Latency: {result.latency_ms} ms")

    assert "[1]" in result.answer
    assert "[2]" in result.answer
    assert len(result.citations) == 2
    assert result.citations[0].source == "invocation-async.md"
    assert result.citations[1].source == "dynamodb-dg.pdf"
    assert result.total_tokens > 0


def test_mock_llm_client_refusal_when_no_sources() -> None:
    """Test that MockLLMClient safely returns standard refusal when no sources are provided."""
    query = "What is the secret formula for dark matter?"
    prompt = build_grounded_prompt(query=query, chunks=[])

    client = MockLLMClient()
    result = client.generate(prompt)

    print("\n" + "═" * 70)
    print("TEST: MockLLMClient refusal on empty sources")
    print("═" * 70)
    print(f"Generated Answer: {result.answer}")

    assert "cannot find enough information" in result.answer
    assert len(result.citations) == 0


def test_mock_llm_client_custom_answer_override() -> None:
    """Test custom answer override on MockLLMClient."""
    prompt = build_grounded_prompt("test", chunks=[_make_dummy_chunk("c", "doc.md", "some text")])
    client = MockLLMClient(default_answer="Custom verified answer [1].")
    result = client.generate(prompt)

    print("\n" + "═" * 70)
    print("TEST: MockLLMClient custom answer")
    print("═" * 70)
    print(f"Generated Answer: {result.answer}")

    assert result.answer == "Custom verified answer [1]."


def test_llm_citation_verification_and_hallucination_detection() -> None:
    """Verify that cited citations are distinguished from retrieved citations and hallucinations flagged."""
    chunks = [
        _make_dummy_chunk("c1", "guide1.md", "Content 1"),
        _make_dummy_chunk("c2", "guide2.md", "Content 2"),
        _make_dummy_chunk("c3", "guide3.md", "Content 3"),
    ]
    prompt = build_grounded_prompt("test question", chunks=chunks)

    # LLM only cites [1] and invents a hallucinated marker [9]
    client = MockLLMClient(default_answer="Based on architecture [1], dark matter [9] is unknown.")
    result = client.generate(prompt)

    assert len(result.retrieved_citations) == 3
    assert len(result.cited_citations) == 1
    assert result.cited_citations[0].source == "guide1.md"
    assert result.hallucinated_markers == [9]


def test_refusal_sanitizes_citation_markers() -> None:
    """Test that out-of-scope refusals strip citation markers and return zero citations."""
    chunks = [
        _make_dummy_chunk("c1", "s3-userguide.pdf", "S3 details"),
        _make_dummy_chunk("c2", "iam-ug.pdf", "IAM details"),
    ]
    prompt = build_grounded_prompt("who is rajan", chunks=chunks)

    refusal_answer = (
        "The provided documentation sources [1], [2], and [3] do not contain any information "
        "about an individual named 'Rajan'. Since this question falls outside the scope of the "
        "referenced AWS IAM and S3 documentation, I cannot provide an answer based on the given sources."
    )
    client = MockLLMClient(default_answer=refusal_answer)
    result = client.generate(prompt)

    # Citation markers must be sanitized out on pure refusals
    assert "[1]" not in result.answer
    assert "[2]" not in result.answer
    assert "[3]" not in result.answer
    assert result.citations == []
    assert result.cited_citations == []
    assert result.hallucinated_markers == []


def test_partial_refusal_with_factual_paragraphs_preserves_citations() -> None:
    """Mixed answers with a missing-link disclaimer plus cited factual details must NOT strip citations."""
    chunks = [
        _make_dummy_chunk("c1", "Concepts.MultiAZ.md", "Multi-AZ DB instance deployments have one standby instance."),
        _make_dummy_chunk("c2", "object-lock.md", "S3 Object Lock uses a WORM model to prevent deletion."),
    ]
    prompt = build_grounded_prompt("RDS Multi-AZ and Object Lock", chunks=chunks)

    mixed_answer = (
        "The provided technical documentation does not contain information linking RDS Multi-AZ with Object Lock.\n\n"
        "Multi-AZ deployments for Amazon RDS can be either a Multi-AZ DB instance deployment with one standby "
        "instance or a Multi-AZ DB cluster deployment with two standby instances [1].\n\n"
        "S3 Object Lock uses a write-once-read-many (WORM) model to prevent Amazon S3 objects from being deleted [2]."
    )
    client = MockLLMClient(default_answer=mixed_answer)
    result = client.generate(prompt)

    assert "[1]" in result.answer
    assert "[2]" in result.answer
    assert len(result.citations) == 2
    assert result.cited_citations[0].source == "Concepts.MultiAZ.md"
    assert result.cited_citations[1].source == "object-lock.md"



def test_make_llm_client_factory() -> None:
    """Test factory creation for mock and nvidia providers."""
    mock_client = make_llm_client("mock")
    assert isinstance(mock_client, MockLLMClient)

    nvidia_client = make_llm_client("nvidia:meta/llama-3.2-90b-vision-instruct", api_key="nvapi-test-key")
    assert isinstance(nvidia_client, NVIDIALLMClient)
    assert nvidia_client.model_name == "meta/llama-3.2-90b-vision-instruct"

    openrouter_client = make_llm_client("openrouter:openai/gpt-4o-mini", api_key="sk-or-test-key")
    assert isinstance(openrouter_client, OpenRouterLLMClient)
    assert openrouter_client.model_name == "openai/gpt-4o-mini"

    groq_client = make_llm_client("groq:openai/gpt-oss-120b", api_key="gsk_test_key")
    assert isinstance(groq_client, GroqLLMClient)
    assert groq_client.model_name == "openai/gpt-oss-120b"

    groq_default = make_llm_client("groq", api_key="gsk_test_key")
    assert isinstance(groq_default, GroqLLMClient)
    assert groq_default.model_name == GroqLLMClient.DEFAULT_MODEL

    with pytest.raises(ValueError, match="Unsupported model specification"):
        make_llm_client("anthropic:claude-3-5-sonnet")


def test_groq_llm_client_missing_key_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that GroqLLMClient raises ValueError if neither GROQ_API_KEY nor Groq_Api_Key is set."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("Groq_Api_Key", raising=False)
    try:
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    except ImportError:
        pass

    with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
        GroqLLMClient(api_key=None)


def test_groq_llm_client_payload_and_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GroqLLMClient sends reasoning_effort & max_completion_tokens for gpt-oss and extracts citations."""
    from unittest.mock import MagicMock

    captured: dict = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [
                {"message": {"content": "AWS Lambda retries failed asynchronous invocations twice by default [1]."}}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
        }
        return resp

    monkeypatch.setattr(requests, "post", fake_post)

    chunk = _make_dummy_chunk("c1", "invocation-async.md", "Lambda retries failed async invocations twice.")
    prompt = build_grounded_prompt("How many times does Lambda retry?", [chunk])

    client = GroqLLMClient(api_key="gsk_test", model_name="openai/gpt-oss-120b")
    res = client.generate(prompt, max_tokens=380, temperature=0.0)

    assert captured["url"] == GroqLLMClient.DEFAULT_ENDPOINT
    assert captured["payload"]["model"] == "openai/gpt-oss-120b"
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["max_completion_tokens"] == 2048
    assert "max_tokens" not in captured["payload"]
    assert len(res.citations) == 1
    assert res.citations[0].source == "invocation-async.md"
    assert res.total_tokens == 125


def test_fullwidth_citation_brackets_normalized_to_ascii() -> None:
    """Models emitting fullwidth 【1】 / 【1, 2】 brackets must normalize cleanly to clickable [1][2] citations."""
    chunks = [
        _make_dummy_chunk("c1", "TTL.md", "TTL deletes expired items without consuming write throughput."),
        _make_dummy_chunk("c2", "TTL.md", "Global Tables replicated deletes consume write capacity in replicas."),
    ]
    prompt = build_grounded_prompt("TTL expiry write throughput", chunks=chunks)

    raw_answer = (
        "The initial TTL delete does not consume write throughput\u202f【1】. "
        "Replicated deletes in Global Tables consume write capacity in replica regions【1, 2】."
    )
    client = MockLLMClient(default_answer=raw_answer)
    result = client.generate(prompt)

    assert "【" not in result.answer
    assert "[1]" in result.answer
    assert "[2]" in result.answer
    assert len(result.citations) == 2
    assert result.hallucinated_markers == []


def test_openrouter_llm_client_missing_key_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OpenRouterLLMClient raises ValueError if no API key is available."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    except ImportError:
        pass

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        OpenRouterLLMClient(api_key=None)


def test_live_openrouter_llm_generation() -> None:
    """Live integration test against OpenRouter API (skipped in offline sandbox mode)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not configured in environment")

    chunks = [
        _make_dummy_chunk(
            chunk_id="c_lambda",
            source="invocation-async.md",
            content=(
                "When you invoke a function asynchronously, Lambda sends the event to a queue. "
                "A separate process reads events from the queue and executes the function. "
                "If the function returns an error, Lambda attempts to run it two more times, with a delay between attempts."
            ),
            headings=["Invoking a Lambda Event Function asynchronously", "Error handling"],
        )
    ]

    query = "How many times does Lambda retry an asynchronous function upon error?"
    prompt = build_grounded_prompt(query=query, chunks=chunks)

    client = OpenRouterLLMClient(api_key=api_key, model_name="openai/gpt-4o-mini")

    try:
        result = client.generate(prompt, max_tokens=150)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        pytest.skip(f"Live OpenRouter API call skipped (network unavailable in sandbox): {exc}")

    print("\n" + "═" * 70)
    print("TEST: Live OpenRouter GPT-4o-Mini Grounded Generation")
    print("═" * 70)
    print(f"Model: {result.model_name}")
    print(f"Answer:\n{result.answer}")
    print(f"Citations: {len(result.citations)} references")
    print(f"Tokens: prompt={result.prompt_tokens}, completion={result.completion_tokens}, total={result.total_tokens}")
    print(f"Latency: {result.latency_ms} ms")

    assert len(result.answer) > 10
    assert "[1]" in result.answer
    assert "two" in result.answer.lower() or "2" in result.answer


def test_nvidia_llm_client_missing_key_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that NVIDIALLMClient raises ValueError if no API key is available."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    try:
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    except ImportError:
        pass

    with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
        NVIDIALLMClient(api_key=None)


def test_live_nvidia_llm_generation() -> None:
    """Live integration test against NVIDIA NIM API (skipped in offline sandbox mode)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        pytest.skip("NVIDIA_API_KEY not configured in environment")

    chunks = [
        _make_dummy_chunk(
            chunk_id="c_lambda",
            source="invocation-async.md",
            content=(
                "When you invoke a function asynchronously, Lambda sends the event to a queue. "
                "A separate process reads events from the queue and executes the function. "
                "If the function returns an error, Lambda attempts to run it two more times, with a delay between attempts."
            ),
            headings=["Invoking a Lambda Event Function asynchronously", "Error handling"],
        )
    ]

    query = "How many times does Lambda retry an asynchronous function upon error?"
    prompt = build_grounded_prompt(query=query, chunks=chunks)

    client = NVIDIALLMClient(api_key=api_key, model_name="meta/llama-3.2-90b-vision-instruct")

    try:
        result = client.generate(prompt, max_tokens=150)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        pytest.skip(f"Live NVIDIA API call skipped (network unavailable in sandbox): {exc}")

    print("\n" + "═" * 70)
    print("TEST: Live NVIDIA Llama-3.2-90B Grounded Generation")
    print("═" * 70)
    print(f"Model: {result.model_name}")
    print(f"Answer:\n{result.answer}")
    print(f"Tokens: prompt={result.prompt_tokens}, completion={result.completion_tokens}, total={result.total_tokens}")
    print(f"Latency: {result.latency_ms} ms")

    assert len(result.answer) > 10
    assert "[1]" in result.answer
    assert "two" in result.answer.lower() or "2" in result.answer
