"""Context assembly and citation grounding for LLM prompt construction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from resolveops.ingestion.chunking import Chunk

DEFAULT_SYSTEM_PROMPT = """You are ResolveOps Assistant, an expert cloud infrastructure and documentation specialist.
Your mission is to provide clear, human-readable, technically accurate answers grounded EXCLUSIVELY in the numbered reference sources provided below.

Strict Grounding & Citation Guidelines:
1. Positive Factual Citations ONLY:
   - Every factual claim, code snippet, API name, or configuration parameter MUST be immediately followed by its source citation marker (e.g., [1] or [2]).
   - NEVER use citation markers (like [1], [2], [3]) when stating that the documentation does not contain an answer or when stating that a topic falls outside the documentation. Only cite a source if it genuinely contains and supports a positive factual statement!
2. Strict Truthfulness:
   - Rely ONLY on the provided sources. Do not invent details, extrapolate unmentioned features, or use external knowledge not present in the text.
3. Natural & Structured Communication:
   - Write like a senior cloud engineer explaining the architecture clearly to a teammate.
   - Use clean Markdown: bold key concepts, use bullet points or numbered lists for sequential steps, and wrap code/parameters in backticks (`code`).
   - Open with a concise summary sentence answering the core query directly before elaborating.
4. Handling Out-of-Scope or Missing Information:
   - If the sources provide partial information, clearly synthesize what the documentation confirms (with citations), and gracefully explain what specific details are missing or not covered in the excerpts.
   - If the sources do not contain information to answer the question, clearly state that the provided technical documentation does not cover this topic. In this case, DO NOT include any citation markers at all. An out-of-scope refusal MUST contain ZERO citation markers.
5. Conflicting Information:
   - If different sources present conflicting or differing details, highlight the contrast clearly with citations to both sources."""


class CitationReference(BaseModel):
    """Provenance and audit metadata for a numbered citation reference (e.g. [1])."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="1-based citation index (e.g. 1 for [1])")
    chunk_id: str = Field(description="Stable SHA-256 chunk identifier")
    source: str = Field(description="Base filename or origin of the document (e.g. dynamodb-dg.pdf)")
    heading_context: list[str] = Field(default_factory=list, description="Section heading hierarchy")
    page_numbers: list[int] = Field(default_factory=list, description="Original PDF page numbers if applicable")
    snippet: str = Field(description="Brief text preview for UI tooltip/hover cards")


class AssembledPrompt(BaseModel):
    """The fully assembled prompt payload ready for LLM generation with citation mappings."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = Field(description="Strict anti-hallucination system prompt")
    user_prompt: str = Field(description="Formatted context sources combined with user query")
    citations: list[CitationReference] = Field(default_factory=list, description="Metadata registry for citations")

    def get_citation(self, index: int) -> CitationReference | None:
        """Retrieve citation metadata by 1-based index."""
        for c in self.citations:
            if c.index == index:
                return c
        return None

    def to_messages(self) -> list[dict[str, str]]:
        """Format into standard chat messages format for OpenAI, NVIDIA NIM, and Anthropic APIs."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


def format_single_chunk(chunk: Chunk, index: int) -> tuple[str, CitationReference]:
    """Format an individual chunk into a numbered reference block and extract its citation metadata."""
    source_name = Path(chunk.source).name
    pages_str = f" (Pages: {', '.join(str(p) for p in chunk.page_numbers)})" if chunk.page_numbers else ""
    section_str = f"\nSection: {' ▸ '.join(chunk.heading_context)}" if chunk.heading_context else ""

    formatted_text = (
        f"[{index}] Source: {source_name}{pages_str}{section_str}\n"
        f"Content:\n{chunk.content.strip()}"
    )

    clean_content = " ".join(chunk.content.strip().split())
    snippet = clean_content[:140] + "..." if len(clean_content) > 140 else clean_content

    citation = CitationReference(
        index=index,
        chunk_id=chunk.id,
        source=source_name,
        heading_context=list(chunk.heading_context),
        page_numbers=list(chunk.page_numbers),
        snippet=snippet,
    )

    return formatted_text, citation


def build_grounded_prompt(
    query: str,
    chunks: Sequence[Chunk],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_chars_budget: int = 12000,
) -> AssembledPrompt:
    """Assemble retrieved chunks into a numbered, citation-grounded prompt for the LLM.

    Args:
        query: The user's technical question.
        chunks: Sequence of candidate chunks (typically top-3 to top-5 from the Reranker).
        system_prompt: The grounding system instructions.
        max_chars_budget: Maximum character limit for the context block to protect token budget.

    Returns:
        AssembledPrompt containing system_prompt, user_prompt, and citations list.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query cannot be empty or whitespace")
    if max_chars_budget < 1:
        raise ValueError("max_chars_budget must be positive")

    if not chunks:
        # Graceful fallback when 0 chunks are retrieved
        user_prompt = (
            "CONTEXT SOURCES:\n"
            "(No relevant documentation sources were found in the knowledge base.)\n\n"
            f"USER QUESTION:\n{clean_query}\n\n"
            "Explain clearly and professionally that the current indexed documentation does not contain sources to answer this question."
        )
        return AssembledPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            citations=[],
        )

    formatted_blocks: list[str] = []
    citations: list[CitationReference] = []
    current_chars = 0

    for chunk in chunks:
        idx = len(citations) + 1
        block_text, citation = format_single_chunk(chunk, index=idx)

        # Ensure we stay strictly within our safe character budget
        if current_chars + len(block_text) > max_chars_budget:
            # A partial table can change the meaning of its rows and columns.
            if chunk.kind == "table":
                continue
            if formatted_blocks:
                # Already have higher-ranked chunks; stop adding
                break

            # First chunk exceeds budget: construct bounded representation
            trunc_suffix = "\n[... content truncated to fit context budget ...]"
            content_stripped = chunk.content.strip()
            header_len = len(block_text) - len(content_stripped)
            avail_chars = max_chars_budget - header_len - len(trunc_suffix)

            if avail_chars > 20:
                truncated_text = content_stripped[:avail_chars]
                if "\n" in truncated_text:
                    truncated_text = truncated_text.rsplit("\n", 1)[0]
                bounded_chunk = chunk.model_copy(update={"content": truncated_text + trunc_suffix})
                block_text, citation = format_single_chunk(bounded_chunk, index=idx)
            else:
                continue

            formatted_blocks.append(block_text)
            citations.append(citation)
            current_chars += len(block_text) + 2
            break

        formatted_blocks.append(block_text)
        citations.append(citation)
        current_chars += len(block_text) + 2

    separator = "─" * 80
    context_section = "\n\n".join(formatted_blocks) or "(No sources fit the context budget.)"

    user_prompt = (
        f"CONTEXT SOURCES:\n"
        f"{separator}\n"
        f"{context_section}\n"
        f"{separator}\n\n"
        f"USER QUESTION:\n{clean_query}\n\n"
        f"Please provide a complete, direct, and factual answer based exclusively on the numbered sources above, "
        f"including citation markers (e.g. [1], [2]) immediately following each claim."
    )

    return AssembledPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        citations=citations,
    )
