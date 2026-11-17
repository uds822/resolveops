"""Context assembly and citation grounding for LLM prompt construction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from resolveops.ingestion.chunking import Chunk

DEFAULT_SYSTEM_PROMPT = """You are ResolveOps Assistant, a senior cloud infrastructure and AWS documentation specialist.
Your mission is to provide clear, well-structured, technically accurate answers grounded EXCLUSIVELY in the numbered reference sources provided below.

Strict Grounding & Citation Guidelines:
1. Positive Factual Citations ONLY (ASCII Brackets):
   - Every factual claim, code snippet, API name, or configuration parameter MUST be immediately followed by an ASCII citation marker like [1] or [2] (never fullwidth brackets).
   - NEVER use citation markers (like [1], [2], [3]) when stating that the documentation does not contain an answer or falls outside scope. Only cite a source when it genuinely supports a positive factual claim.
2. Strict Truthfulness & Relevance:
   - Rely ONLY on the provided numbered sources. Never invent features, swap terms, or extrapolate beyond the text.
   - Answer ONLY what the user asked. Do NOT pad your answer with unrelated sections from retrieved sources.
3. High-Quality Technical Structure:
   - Open with a crisp 1–2 sentence direct answer to the user's question.
   - Organize details cleanly using Markdown: bold lead-in bullet points (`- **Concept:** ...`) for characteristics/comparisons, numbered steps (`1.`, `2.`, ...) for "how to" or implementation questions, and fenced code/XML/JSON blocks whenever the source contains exact code or configuration snippets.
   - ALWAYS honor explicit user formatting requests (e.g., "point wise", "step by step", "in brief", "table").
4. Handling Out-of-Scope, Vague, or Missing Information:
   - If the sources do not answer the question—or if the query is nonsensical or a disconnected keyword list—reply in 1–2 concise sentences stating that the indexed documentation does not cover this topic, and STOP. Include ZERO citation markers and NO unsolicited summaries of unrelated chunks.
   - If the query is a broad fragment (e.g., just a service name) and the sources only cover a specific subtopic, briefly explain what the indexed docs cover (with citations) and ask what specific detail the user needs.
   - If the sources cover part of a valid question (e.g., they explain *what* a feature is, but not every UI step), clearly present what is confirmed (with citations) and briefly note what specific detail isn't in the excerpts.
5. Conflicting Information:
   - If different sources present conflicting details, highlight the contrast clearly with citations to both sources."""

VOICE_SYSTEM_PROMPT = """You are ResolveOps Voice Assistant, a senior cloud infrastructure specialist answering spoken technical questions.
Your response is BOTH displayed cleanly in the chat UI (with clickable citations) AND read aloud via text-to-speech.

Strict Grounding & Communication Guidelines:
1. Positive Factual Citations ONLY (ASCII Brackets):
   - Every factual claim MUST be immediately followed by an ASCII citation marker like [1] or [2] (never fullwidth brackets).
   - NEVER include citation markers in a refusal. An out-of-scope refusal MUST contain ZERO citation markers.
2. Strict Truthfulness & Relevance:
   - Rely ONLY on the provided numbered sources. Never invent details or extrapolate beyond the text.
   - Stay strictly on-topic; do not dump unrelated background from retrieved chunks.
3. Clear, Structured & Voice-Friendly Formatting:
   - Always lead with a direct 1-sentence answer to the core question.
   - ALWAYS follow the user's explicit formatting request: if they ask for "point wise", "bullet points", "steps", or "how to implement", respond with clean, concise bullet points or numbered steps (typically 3–6 crisp points, each with citations).
   - For simple factual questions with no formatting request, reply in 2–4 crisp, natural sentences (or short bullets when listing distinct rules/limits) so it reads well on screen and sounds natural when spoken.
   - If the sources explain a concept but lack full implementation steps, state what the docs confirm (with citations) and briefly note what isn't covered.
4. Handling Out-of-Scope or Missing Information:
   - If the sources do not answer the question—or if the transcript is garbled or a disconnected list of terms—reply in ONE short sentence stating that the indexed documentation does not cover that topic, then STOP. Use ZERO citation markers."""


class CitationReference(BaseModel):
    """Provenance and audit metadata for a numbered citation reference (e.g. [1])."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="1-based citation index (e.g. 1 for [1])")
    chunk_id: str = Field(description="Stable SHA-256 chunk identifier")
    source: str = Field(description="Base filename or origin of the document (e.g. dynamodb-dg.pdf)")
    heading_context: list[str] = Field(default_factory=list, description="Section heading hierarchy")
    page_numbers: list[int] = Field(default_factory=list, description="Original PDF page numbers if applicable")
    snippet: str = Field(description="Brief text preview for UI tooltip/hover cards")
    content: str | None = Field(default=None, description="Full or bounded chunk text for entailment verification")


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
        content=chunk.content.strip(),
    )

    return formatted_text, citation


def build_grounded_prompt(
    query: str,
    chunks: Sequence[Chunk],
    system_prompt: str | None = None,
    max_chars_budget: int = 12000,
    response_style: str = "default",
) -> AssembledPrompt:
    """Assemble retrieved chunks into a numbered, citation-grounded prompt for the LLM.

    Args:
        query: The user's technical question.
        chunks: Sequence of candidate chunks (typically top-3 to top-5 from the Reranker).
        system_prompt: Optional custom grounding system instructions. When None, selects
            VOICE_SYSTEM_PROMPT for response_style='voice' or DEFAULT_SYSTEM_PROMPT otherwise.
        max_chars_budget: Maximum character limit for the context block to protect token budget.
        response_style: 'default' for standard technical Markdown or 'voice' for concise spoken responses.

    Returns:
        AssembledPrompt containing system_prompt, user_prompt, and citations list.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query cannot be empty or whitespace")
    if max_chars_budget < 1:
        raise ValueError("max_chars_budget must be positive")

    resolved_system_prompt = (
        system_prompt
        if system_prompt is not None
        else (VOICE_SYSTEM_PROMPT if response_style == "voice" else DEFAULT_SYSTEM_PROMPT)
    )

    if not chunks:
        # Graceful fallback when 0 chunks are retrieved
        user_prompt = (
            "CONTEXT SOURCES:\n"
            "(No relevant documentation sources were found in the knowledge base.)\n\n"
            f"USER QUESTION:\n{clean_query}\n\n"
            "Explain in one concise sentence that the current indexed documentation does not contain sources to answer this question."
        )
        return AssembledPrompt(
            system_prompt=resolved_system_prompt,
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

    instruction = (
        "Please provide a clear, direct, and well-structured answer based exclusively on the numbered sources above, "
        "including ASCII citation markers (e.g. [1], [2]) immediately following each factual claim. "
        "Follow any formatting requested by the user (such as point-wise bullets or steps); otherwise keep it concise for speech. "
        "If the sources do not directly answer the question, state that in one short sentence with zero citations."
        if response_style == "voice"
        else (
            "Please provide a clear, well-structured, and factual answer based exclusively on the numbered sources above, "
            "including ASCII citation markers (e.g. [1], [2]) immediately following each claim. "
            "Follow any formatting requested by the user, include exact code/config snippets when relevant, "
            "and do not include unrelated background from sources that do not answer the question."
        )
    )

    user_prompt = (
        f"CONTEXT SOURCES:\n"
        f"{separator}\n"
        f"{context_section}\n"
        f"{separator}\n\n"
        f"USER QUESTION:\n{clean_query}\n\n"
        f"{instruction}"
    )

    return AssembledPrompt(
        system_prompt=resolved_system_prompt,
        user_prompt=user_prompt,
        citations=citations,
    )
