"""LLM client adapter layer for generating grounded, citation-verified answers."""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Sequence

import requests
from pydantic import BaseModel, ConfigDict, Field

from resolveops.generation.context_builder import AssembledPrompt, CitationReference


def is_refusal_answer(answer: str) -> bool:
    """Check if the answer is purely an out-of-scope refusal or missing-information notification."""
    refusal_patterns = [
        r"(?:do|does)\s+not\s+contain\s+(?:any\s+)?information",
        r"cannot\s+provide\s+an\s+answer\s+based\s+on\s+the\s+(?:given|provided)\s+sources",
        r"falls?\s+outside\s+the\s+scope",
        r"provided\s+documentation\s+(?:excerpts\s+)?do(?:es)?\s+not\s+cover",
        r"cannot\s+find\s+enough\s+information",
        r"no\s+relevant\s+information\s+(?:found|available)",
    ]
    lower = answer.lower()
    return any(re.search(pat, lower) for pat in refusal_patterns)


def verify_and_split_citations(
    answer: str,
    prompt_citations: Sequence[CitationReference],
) -> tuple[str, list[CitationReference], list[CitationReference], list[int]]:
    """Parse citation markers from answer text and partition into cited, retrieved, and hallucinated.

    If the response is a refusal, citation markers are sanitized and zero citations are extracted.

    Returns:
        (sanitized_answer, cited_citations, retrieved_citations, hallucinated_markers)
    """
    clean_answer = answer

    # If the response is an out-of-scope refusal, strip any inappropriate citation markers
    if is_refusal_answer(answer):
        # E.g. "sources [1], [2], and [3] do not contain" -> "sources do not contain"
        clean_answer = re.sub(
            r"(?:sources|documents|references)\s+(?:\[\d+\]\s*(?:,\s*|and\s*)?)+\s+do(?:es)?\s+not\s+contain",
            "sources do not contain",
            clean_answer,
            flags=re.IGNORECASE,
        )
        clean_answer = re.sub(r"\s*\[\d+\]", "", clean_answer)
        raw_markers: list[int] = []
    else:
        raw_markers = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]

    seen_markers: set[int] = set()
    unique_markers: list[int] = []
    for m in raw_markers:
        if m not in seen_markers:
            seen_markers.add(m)
            unique_markers.append(m)

    cit_map = {c.index: c for c in prompt_citations}
    cited = [cit_map[idx] for idx in unique_markers if idx in cit_map]
    hallucinated = [idx for idx in unique_markers if idx not in cit_map]
    retrieved = list(prompt_citations)

    return clean_answer, cited, retrieved, hallucinated


class GenerationResult(BaseModel):
    """The generated answer with citation metadata and operational metrics."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="The generated response text with citation markers (e.g. [1])")
    citations: list[CitationReference] = Field(default_factory=list, description="Provenance metadata for citations")
    retrieved_citations: list[CitationReference] = Field(default_factory=list, description="All references provided in prompt context")
    cited_citations: list[CitationReference] = Field(default_factory=list, description="References explicitly cited in the answer text")
    hallucinated_markers: list[int] = Field(default_factory=list, description="Citation markers in answer that had no source in prompt")
    model_name: str = Field(description="Name/identifier of the generative model used")
    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the generated output")
    total_tokens: int = Field(default=0, description="Total tokens used for this query")
    latency_ms: float = Field(default=0.0, description="End-to-end generation latency in milliseconds")


class LLMClient(ABC):
    """Abstract base class for generative language model adapters."""

    @abstractmethod
    def generate(
        self,
        prompt: AssembledPrompt,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate a grounded answer for an assembled prompt.

        Args:
            prompt: The assembled prompt with system guardrails, numbered sources, and query.
            max_tokens: Maximum tokens to generate in the completion.
            temperature: Sampling temperature (0.0 recommended for factual RAG).

        Returns:
            GenerationResult containing the answer, citation mappings, and latency metrics.
        """
        ...


class NVIDIALLMClient(LLMClient):
    """Production LLM client calling NVIDIA NIM chat completions (OpenAI-compatible protocol)."""

    DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
    DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        endpoint_url: str = DEFAULT_ENDPOINT,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass

        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is required for NVIDIALLMClient. "
                "Set it in .env or pass api_key explicitly."
            )

        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.timeout = timeout

    def generate(
        self,
        prompt: AssembledPrompt,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            citations_list = []
        else:
            messages = prompt.to_messages()
            citations_list = prompt.citations

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.monotonic()
        response = requests.post(
            self.endpoint_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        if response.status_code != 200:
            raise RuntimeError(
                f"NVIDIA Chat Completion API failed ({response.status_code}): {response.text}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"NVIDIA API returned no choices in response: {data}")

        answer_text = choices[0].get("message", {}).get("content", "").strip()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        answer_text, cited, retrieved, hallucinated = verify_and_split_citations(answer_text, citations_list)

        return GenerationResult(
            answer=answer_text,
            citations=cited,
            retrieved_citations=retrieved,
            cited_citations=cited,
            hallucinated_markers=hallucinated,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
        )


class OpenRouterLLMClient(LLMClient):
    """Production LLM client calling OpenRouter chat completions API."""

    DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "deepseek/deepseek-chat"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        endpoint_url: str = DEFAULT_ENDPOINT,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouterLLMClient. "
                "Set it in .env or pass api_key explicitly."
            )

        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.timeout = timeout

    def generate(
        self,
        prompt: AssembledPrompt,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            citations_list = []
        else:
            messages = prompt.to_messages()
            citations_list = prompt.citations

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.monotonic()
        response = requests.post(
            self.endpoint_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter Chat Completion API failed ({response.status_code}): {response.text}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter API returned no choices in response: {data}")

        answer_text = choices[0].get("message", {}).get("content", "").strip()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        answer_text, cited, retrieved, hallucinated = verify_and_split_citations(answer_text, citations_list)

        return GenerationResult(
            answer=answer_text,
            citations=cited,
            retrieved_citations=retrieved,
            cited_citations=cited,
            hallucinated_markers=hallucinated,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
        )


class MockLLMClient(LLMClient):
    """Fast, deterministic offline mock client for testing and CI pipelines."""

    def __init__(
        self,
        model_name: str = "mock-llm-v1",
        default_answer: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.default_answer = default_answer

    def generate(
        self,
        prompt: AssembledPrompt,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        t0 = time.monotonic()

        if self.default_answer is not None:
            answer = self.default_answer
        elif not prompt.citations:
            answer = "Based on the provided documentation, I cannot find enough information to answer this question."
        else:
            # Construct a realistic synthetic answer using citation markers
            parts = []
            for cit in prompt.citations:
                parts.append(f"According to {cit.source}, {cit.snippet} [{cit.index}].")
            answer = " ".join(parts)

        latency_ms = (time.monotonic() - t0) * 1000
        answer, cited, retrieved, hallucinated = verify_and_split_citations(answer, prompt.citations)

        return GenerationResult(
            answer=answer,
            citations=cited,
            retrieved_citations=retrieved,
            cited_citations=cited,
            hallucinated_markers=hallucinated,
            model_name=self.model_name,
            prompt_tokens=150,
            completion_tokens=40,
            total_tokens=190,
            latency_ms=round(latency_ms, 3),
        )


def make_llm_client(
    model_spec: str = "openrouter:deepseek/deepseek-chat",
    api_key: str | None = None,
) -> LLMClient:
    """Factory function for creating an LLM client by provider prefix.

    Supported patterns:
        - 'mock' -> MockLLMClient (offline, 0 latency, free)
        - 'deepseek' -> OpenRouterLLMClient(model_name='deepseek/deepseek-chat')
        - 'deepseek:<model_name>' -> OpenRouterLLMClient(model_name='deepseek/<model_name>')
        - 'openrouter:<model_name>' -> OpenRouterLLMClient (e.g. openrouter:deepseek/deepseek-chat)
        - 'nvidia:<model_name>' -> NVIDIALLMClient (e.g. nvidia:meta/llama-3.2-90b-vision-instruct)
    """
    if model_spec == "mock":
        return MockLLMClient()

    if model_spec == "deepseek":
        return OpenRouterLLMClient(model_name="deepseek/deepseek-chat", api_key=api_key)

    if model_spec.startswith("deepseek:"):
        model_name = f"deepseek/{model_spec.split('deepseek:', 1)[1]}"
        return OpenRouterLLMClient(model_name=model_name, api_key=api_key)

    if model_spec.startswith("openrouter:"):
        model_name = model_spec.split("openrouter:", 1)[1]
        return OpenRouterLLMClient(model_name=model_name, api_key=api_key)

    if model_spec.startswith("nvidia:"):
        model_name = model_spec.split("nvidia:", 1)[1]
        return NVIDIALLMClient(model_name=model_name, api_key=api_key)

    raise ValueError(
        f"Unsupported model specification '{model_spec}'. "
        f"Expected 'mock', 'deepseek', 'openrouter:<model_name>', or 'nvidia:<model_name>'."
    )

