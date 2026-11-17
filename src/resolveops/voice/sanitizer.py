"""Speech sanitizer converting technical Markdown + citation text into natural spoken prose."""

from __future__ import annotations

import re


_UNIT_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b1\s*TB\b"), "1 terabyte"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*TB\b"), r"\1 terabytes"),
    (re.compile(r"\b1\s*GB\b"), "1 gigabyte"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*GB\b"), r"\1 gigabytes"),
    (re.compile(r"\b1\s*MB\b"), "1 megabyte"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*MB\b"), r"\1 megabytes"),
    (re.compile(r"\b1\s*KB\b"), "1 kilobyte"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*KB\b"), r"\1 kilobytes"),
    (re.compile(r"\b1\s*ms\b"), "1 millisecond"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*ms\b"), r"\1 milliseconds"),
]


def _replace_markdown_tables(text: str) -> str:
    """Replace multi-line Markdown tables with a natural spoken pointer to the UI."""
    lines = text.splitlines()
    out: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_row = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2
        if is_table_row:
            if not in_table:
                out.append("See the screen for the detailed table.")
                in_table = True
        else:
            in_table = False
            out.append(line)

    return "\n".join(out)


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Trim overly long answers cleanly at a sentence boundary for natural speech."""
    if len(text) <= max_chars:
        return text

    suffix = " You can check the screen for the remaining details."
    budget = max(80, max_chars - len(suffix))
    candidate = text[:budget]

    # Prefer ending at the last complete sentence boundary
    last_end = max(candidate.rfind(". "), candidate.rfind("! "), candidate.rfind("? "))
    if last_end >= int(budget * 0.4):
        return candidate[: last_end + 1].strip() + suffix

    # Fallback: trim cleanly at the last word boundary
    last_space = candidate.rfind(" ")
    trimmed = candidate[:last_space].strip() if last_space > 0 else candidate.strip()
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed + suffix


def sanitize_for_speech(markdown_text: str, max_chars: int = 1200) -> str:
    """Convert citation-grounded technical Markdown into natural, human-sounding spoken text.

    Preserves the full technical meaning while stripping visual-only artifacts such as
    citation brackets ([1], [p.42]), bold/code syntax, fenced code blocks, and pipe tables,
    and expanding common cloud storage/latency units for smooth neural TTS prosody.

    Args:
        markdown_text: Raw Markdown answer returned by RAGService.
        max_chars: Soft maximum character cap for spoken playback (default 1200).

    Returns:
        Clean, punctuation-balanced natural prose ready for TTS synthesis.
    """
    if not markdown_text or not markdown_text.strip():
        return ""

    text = markdown_text

    # 1. Replace fenced code blocks with a natural spoken cue
    text = re.sub(r"```[\s\S]*?```", " See the screen for the code example. ", text)

    # 2. Replace Markdown pipe tables with a natural spoken cue
    text = _replace_markdown_tables(text)

    # 3. Convert Markdown links [label](url) -> label, then drop bare URLs
    text = re.sub(r"\[([^\]]+)\]\((?:https?://[^\)]+)\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)

    # 4. Strip numeric and page citation markers: [1], [1, 2], [1][2], [p.42], [p. 42]
    text = re.sub(r"\[\s*(?:p\.\s*)?\d+(?:\s*,\s*\d+)*\s*\]", "", text)

    # 5. Strip Markdown headings and ensure a natural spoken pause after each heading
    def _clean_heading(match: re.Match[str]) -> str:
        heading = match.group(1).strip().rstrip(":")
        if heading and heading[-1] not in ".!?":
            heading += "."
        return f"\n{heading}\n"

    text = re.sub(r"^\s*#{1,6}\s+(.+)$", _clean_heading, text, flags=re.MULTILINE)

    # 6. Strip bullet and numbered list prefixes, ensuring each item ends with a pause
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_list_item = bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line))
        if is_list_item:
            line = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line).strip()
            if line and line[-1] not in ".!?,;:":
                line += "."
        cleaned_lines.append(line)
    text = " ".join(cleaned_lines)

    # 7. Strip inline Markdown emphasis and backticks (**bold**, *italic*, __underline__, `code`)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)

    # 8. Expand common Latin abbreviations & technical symbols for smooth speech
    text = re.sub(r"\be\.g\.,?", "for example,", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\.e\.,?", "that is,", text, flags=re.IGNORECASE)
    text = text.replace(">=", " greater than or equal to ")
    text = text.replace("<=", " less than or equal to ")
    text = text.replace("→", " to ")
    text = text.replace("->", " to ")
    text = text.replace("▸", ", ")

    # 9. Expand technical storage and latency units (10 GB -> 10 gigabytes)
    for pattern, replacement in _UNIT_EXPANSIONS:
        text = pattern.sub(replacement, text)

    # 10. Normalize whitespace and clean punctuation artifacts left by removed citations
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    # 11. Keep spoken response comfortably paced on very long answers
    return _truncate_at_sentence(text, max_chars=max_chars)

