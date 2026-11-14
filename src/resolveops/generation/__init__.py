"""ResolveOps generation module: Grounded context assembly, prompt construction, citations, and LLM clients."""

from resolveops.generation.context_builder import (
    DEFAULT_SYSTEM_PROMPT,
    AssembledPrompt,
    CitationReference,
    build_grounded_prompt,
    format_single_chunk,
)
from resolveops.generation.llm import (
    GenerationResult,
    LLMClient,
    MockLLMClient,
    NVIDIALLMClient,
    OpenRouterLLMClient,
    make_llm_client,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "CitationReference",
    "AssembledPrompt",
    "format_single_chunk",
    "build_grounded_prompt",
    "GenerationResult",
    "LLMClient",
    "NVIDIALLMClient",
    "OpenRouterLLMClient",
    "MockLLMClient",
    "make_llm_client",
]
