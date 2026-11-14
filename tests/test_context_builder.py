"""Unit tests for context builder and citation grounding."""

from __future__ import annotations

import pytest

from resolveops.generation.context_builder import (
    DEFAULT_SYSTEM_PROMPT,
    AssembledPrompt,
    CitationReference,
    build_grounded_prompt,
    format_single_chunk,
)
from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType


def _make_dummy_chunk(
    chunk_id: str,
    source: str,
    content: str,
    headings: list[str] | None = None,
    pages: list[int] | None = None,
    kind: str = "text",
) -> Chunk:
    source_type = SourceType.PDF if source.endswith(".pdf") else SourceType.MARKDOWN
    return Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source=source,
        source_type=source_type,
        document_title="Test AWS Guide",
        kind=kind,
        content=content,
        block_ids=[f"block_{chunk_id}"],
        context_block_ids=[],
        heading_context=headings or [],
        page_numbers=pages or [],
        indexable=True,
    )


def test_format_single_chunk_markdown_and_pdf() -> None:
    """Test chunk formatting with and without page numbers."""
    c_md = _make_dummy_chunk(
        chunk_id="chunk_md_001",
        source="data/raw/aws/markdown/TTL.md",
        content="Time To Live (TTL) for DynamoDB is a cost-effective method for deleting items...",
        headings=["Using time to live (TTL) in DynamoDB", "Overview"],
        pages=[],
    )
    formatted_md, citation_md = format_single_chunk(c_md, index=1)

    print("\n" + "═" * 70)
    print("TEST: format_single_chunk (Markdown)")
    print("═" * 70)
    print(formatted_md)
    print(f"Citation: {citation_md}")

    assert "[1] Source: TTL.md" in formatted_md
    assert "Section: Using time to live (TTL) in DynamoDB ▸ Overview" in formatted_md
    assert "Time To Live (TTL)" in formatted_md
    assert citation_md.index == 1
    assert citation_md.source == "TTL.md"
    assert citation_md.page_numbers == []
    assert citation_md.chunk_id == "chunk_md_001"

    c_pdf = _make_dummy_chunk(
        chunk_id="chunk_pdf_002",
        source="data/raw/aws/pdf/dynamodb-dg.pdf",
        content="Version 3.3.2 | Upgrading dependencies | Updated AWS SDK for Java to version 2.55.15",
        headings=["Release history for DynamoDB local"],
        pages=[58],
        kind="table",
    )
    formatted_pdf, citation_pdf = format_single_chunk(c_pdf, index=2)

    print("\n" + "═" * 70)
    print("TEST: format_single_chunk (PDF with Page Number)")
    print("═" * 70)
    print(formatted_pdf)
    print(f"Citation: {citation_pdf}")

    assert "[2] Source: dynamodb-dg.pdf (Pages: 58)" in formatted_pdf
    assert "Section: Release history for DynamoDB local" in formatted_pdf
    assert citation_pdf.index == 2
    assert citation_pdf.source == "dynamodb-dg.pdf"
    assert citation_pdf.page_numbers == [58]


def test_build_grounded_prompt_assembled_structure() -> None:
    """Test full prompt assembly with multiple sources and citation registry."""
    chunks = [
        _make_dummy_chunk(
            chunk_id="c_lambda",
            source="invocation-async.md",
            content="When an asynchronous function fails, Lambda retries the invocation twice by default.",
            headings=["Invoking a Lambda Event Function asynchronously", "Error retries"],
        ),
        _make_dummy_chunk(
            chunk_id="c_dynamo",
            source="Programming.Errors.md",
            content="DynamoDB returns HTTP status code 400 for client-side errors like AccessDeniedException.",
            headings=["Error handling with DynamoDB", "HTTP status code 400"],
        ),
    ]

    query = "How does Lambda handle errors during asynchronous invocations?"
    assembled = build_grounded_prompt(query=query, chunks=chunks)

    print("\n" + "═" * 70)
    print("TEST: build_grounded_prompt (Assembled)")
    print("═" * 70)
    print("--- SYSTEM PROMPT ---")
    print(assembled.system_prompt[:200] + "...")
    print("\n--- USER PROMPT ---")
    print(assembled.user_prompt)
    print("\n--- CITATION REGISTRY ---")
    for cit in assembled.citations:
        print(f"[{cit.index}] {cit.source} (Pages: {cit.page_numbers}) ▸ {cit.heading_context} -> {cit.snippet}")

    assert len(assembled.citations) == 2
    assert assembled.citations[0].source == "invocation-async.md"
    assert assembled.citations[1].source == "Programming.Errors.md"

    # Verify query and markers are in user prompt
    assert query in assembled.user_prompt
    assert "[1] Source: invocation-async.md" in assembled.user_prompt
    assert "[2] Source: Programming.Errors.md" in assembled.user_prompt

    # Test citation lookup helper
    c1 = assembled.get_citation(1)
    assert c1 is not None
    assert c1.chunk_id == "c_lambda"

    c_none = assembled.get_citation(99)
    assert c_none is None

    # Test chat messages formatting
    messages = assembled.to_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert query in messages[1]["content"]


def test_build_grounded_prompt_empty_chunks_graceful_fallback() -> None:
    """Test behavior when no chunks are found (retriever returned 0 results)."""
    query = "What is the capital of Mars?"
    assembled = build_grounded_prompt(query=query, chunks=[])

    print("\n" + "═" * 70)
    print("TEST: build_grounded_prompt (0 Chunks Refusal)")
    print("═" * 70)
    print(assembled.user_prompt)

    assert len(assembled.citations) == 0
    assert "No relevant documentation sources were found" in assembled.user_prompt
    assert query in assembled.user_prompt


def test_build_grounded_prompt_budget_truncation() -> None:
    """Test that context does not exceed character budget limit."""
    long_content = "This is a detailed technical sentence about architecture. " * 50
    chunks = [
        _make_dummy_chunk(f"chunk_{i}", f"doc_{i}.md", long_content)
        for i in range(10)
    ]

    # Set very tight budget allowing only ~1-2 chunks
    budget = 1500
    assembled = build_grounded_prompt("Summarize", chunks, max_chars_budget=budget)

    print("\n" + "═" * 70)
    print("TEST: build_grounded_prompt (Budget Truncation)")
    print("═" * 70)
    print(f"Requested 10 chunks. Assembled citations count: {len(assembled.citations)}")
    print(f"Total user prompt character count: {len(assembled.user_prompt)}")

    assert len(assembled.citations) < 10
    assert len(assembled.citations) >= 1


def test_build_grounded_prompt_validation_error() -> None:
    """Test that empty or whitespace query raises ValueError."""
    with pytest.raises(ValueError, match="query cannot be empty"):
        build_grounded_prompt(query="   ", chunks=[])


def test_build_grounded_prompt_enforces_budget_on_oversized_first_chunk() -> None:
    """Verify that a 5,000-character first chunk is bounded when context budget is small."""
    huge_table = "| Col 1 | Col 2 |\n|---|---|\n" + ("| Data A | Data B |\n" * 200)
    chunk = _make_dummy_chunk("c_huge", "table.md", huge_table)

    budget = 250
    assembled = build_grounded_prompt("Query", [chunk], max_chars_budget=budget)

    # Context section inside user prompt must be bounded
    assert len(assembled.citations) == 1
    # Check that the oversized content was truncated
    assert "[... content truncated to fit context budget ...]" in assembled.user_prompt or len(assembled.user_prompt) < 1000


def test_oversized_table_is_skipped_without_losing_later_sources() -> None:
    table = _make_dummy_chunk("table", "table.md", "header\n" + "row\n" * 1000).model_copy(update={"kind": "table"})
    text = _make_dummy_chunk("text", "doc.md", "A short relevant passage.")
    result = build_grounded_prompt("query", [table, text], max_chars_budget=200)
    assert [citation.chunk_id for citation in result.citations] == ["text"]
    assert "[1] Source: doc.md" in result.user_prompt

