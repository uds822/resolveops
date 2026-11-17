"""Pydantic data models for the ResolveOps Voice Agent pipeline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from resolveops.generation.context_builder import CitationReference
from resolveops.generation.grounding_verifier import GroundingAuditReport


class STTResult(BaseModel):
    """Output produced by a Speech-to-Text (STT) adapter."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Clean transcribed user speech text")
    language: str = Field(default="en", description="Detected ISO language code (e.g. 'en')")
    language_probability: float = Field(default=1.0, ge=0.0, le=1.0, description="Language detection confidence")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Duration of the input audio in seconds")
    latency_ms: float = Field(default=0.0, ge=0.0, description="STT transcription latency in milliseconds")
    model_name: str = Field(default="unknown", description="STT engine/model identifier")


class TTSResult(BaseModel):
    """Output produced by a Text-to-Speech (TTS) adapter."""

    model_config = ConfigDict(extra="forbid")

    audio_bytes: bytes = Field(description="Synthesized audio payload (WAV or MP3 bytes)")
    media_type: str = Field(default="audio/wav", description="MIME type of synthesized audio (e.g. 'audio/wav', 'audio/mpeg')")
    sample_rate: int = Field(default=24000, ge=8000, description="Audio sample rate in Hz")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Estimated spoken audio duration in seconds")
    latency_ms: float = Field(default=0.0, ge=0.0, description="TTS synthesis latency in milliseconds")
    voice: str = Field(default="default", description="Voice preset identifier used for synthesis")
    model_name: str = Field(default="unknown", description="TTS engine/model identifier")


class VoiceTimings(BaseModel):
    """End-to-end latency breakdown across speech and RAG pipeline stages."""

    model_config = ConfigDict(extra="forbid")

    stt_ms: float = Field(default=0.0, description="Stage 0: Speech-to-Text transcription latency (ms)")
    retrieval_ms: float = Field(default=0.0, description="Stage 1: Hybrid search latency (ms)")
    rerank_ms: float = Field(default=0.0, description="Stage 2: Cross-Encoder reranking latency (ms)")
    generation_ms: float = Field(default=0.0, description="Stage 3: LLM generation latency (ms)")
    claim_audit_ms: float = Field(default=0.0, description="Stage 4: Semantic claim grounding audit latency (ms)")
    tts_ms: float = Field(default=0.0, description="Stage 5: Text-to-Speech synthesis latency (ms)")
    total_ms: float = Field(default=0.0, description="Total end-to-end voice query latency (ms)")


class VoiceQueryResponse(BaseModel):
    """Complete JSON response returned by POST /api/v1/voice/query."""

    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(description="Transcribed text from the user's spoken audio")
    language: str = Field(default="en", description="Detected spoken language")
    answer: str = Field(description="Full Markdown citation-tagged answer for visual display")
    spoken_text: str = Field(description="Sanitized natural-prose text sent to the TTS engine")

    citations: list[CitationReference] = Field(default_factory=list, description="Provenance metadata for cited references")
    retrieved_citations: list[CitationReference] = Field(default_factory=list, description="All retrieved context references")
    cited_citations: list[CitationReference] = Field(default_factory=list, description="References explicitly cited in answer")
    hallucinated_markers: list[int] = Field(default_factory=list, description="Unmatched citation markers if any")
    citation_validation_failed: bool = Field(default=False, description="True if citation/grounding audit flagged issues")
    faithfulness_score: float | None = Field(default=None, description="Tier 2 claim entailment faithfulness score (0.0 - 1.0)")
    claim_audit: GroundingAuditReport | None = Field(default=None, description="Detailed per-claim grounding audit report")

    audio_id: str | None = Field(default=None, description="Audio cache key for GET /api/v1/voice/audio/{audio_id}")
    audio_url: str | None = Field(default=None, description="Relative URL to fetch synthesized spoken audio")
    audio_media_type: str | None = Field(default=None, description="MIME type of synthesized audio")
    audio_duration_seconds: float | None = Field(default=None, description="Duration of synthesized speech in seconds")
    tts_error: str | None = Field(default=None, description="Populated if TTS degraded gracefully to text-only")

    timings: VoiceTimings = Field(description="Full voice + RAG latency breakdown")
    model_name: str = Field(description="Generative LLM model used")
    stt_model: str = Field(description="STT model used")
    tts_model: str | None = Field(default=None, description="TTS model used")

