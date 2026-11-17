"""Pluggable Speech-to-Text (STT) adapters for ResolveOps Voice Agent."""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from resolveops.voice.models import STTResult


DEFAULT_AWS_HOTWORDS = (
    "AWS Lambda retries asynchronous DynamoDB Amazon S3 IAM EC2 "
    "RDS Multi-AZ Object Lock TTL DLQ SQS VPC API HTTP CLI JSON"
)

# Retained as an alias for backward compatibility; hotwords are used instead of
# initial_prompt to avoid Whisper echoing prompt text during music or silence.
DEFAULT_AWS_VOCAB_PROMPT = DEFAULT_AWS_HOTWORDS

_WHISPER_HALLUCINATION_PHRASES = {
    "thank you",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe",
    "like and subscribe",
    "music",
    "applause",
    "laughter",
    "silence",
    "bye",
    "bye bye",
    "goodbye",
    "you",
    "the end",
    "subtitles by",
}

_QUERY_INTENT_WORDS = {
    "what", "how", "why", "when", "where", "which", "who", "can", "could",
    "does", "do", "is", "are", "was", "were", "will", "would", "should",
    "explain", "describe", "tell", "compare", "list", "show", "define",
    "difference", "between", "work", "works", "happen", "happens", "configure",
}


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _is_hallucinated_transcript(text: str, reference_vocab: str | None = None) -> bool:
    """Detect common Whisper silence/music hallucinations and prompt-echo artifacts."""
    cleaned = text.strip()
    if not cleaned:
        return True

    # Reject pure music-note or bracketed non-speech markers like "[Music]", "(Singing)", "♪"
    stripped_markers = re.sub(r"[\[\(][^\]\)]*[\]\)]", "", cleaned)
    stripped_markers = re.sub(r"[♪♫🎵🎶\.\,\!\?\-\s]+", "", stripped_markers)
    if not stripped_markers:
        return True

    norm_phrase = " ".join(_normalize_tokens(cleaned))
    if not norm_phrase or norm_phrase in _WHISPER_HALLUCINATION_PHRASES:
        return True

    if reference_vocab:
        ref_norm = " ".join(_normalize_tokens(reference_vocab))
        # Exact substring echo of a long prompt phrase
        if len(norm_phrase) >= 16 and norm_phrase in ref_norm:
            return True

        # Comma-separated keyword dump echoing hotwords/prompt with no question/verb intent
        tokens = _normalize_tokens(cleaned)
        ref_tokens = set(_normalize_tokens(reference_vocab))
        if cleaned.count(",") >= 2 and len(tokens) >= 4:
            overlap_ratio = sum(1 for t in tokens if t in ref_tokens) / len(tokens)
            has_intent = any(t in _QUERY_INTENT_WORDS for t in tokens)
            if overlap_ratio >= 0.8 and not has_intent:
                return True

    return False


class STTAdapter(ABC):
    """Abstract base contract for all Speech-to-Text engines."""

    model_name: str

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> STTResult:
        """Transcribe raw uploaded audio bytes (WAV, WebM, MP3, MP4, OGG) into text."""
        ...


class FasterWhisperSTT(STTAdapter):
    """Fully local, zero-cost CPU-optimized Whisper STT powered by CTranslate2.

    Features:
      - Lazy singleton model loading (avoids reloading weights on every request).
      - Built-in Silero VAD silence filtering (`vad_filter=True`) to avoid silence hallucinations.
      - Safe `hotwords` domain boosting (avoids `initial_prompt` leakage on music/noise).
      - Language, log-probability, no-speech, and prompt-echo quality gates.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        hotwords: str | None = DEFAULT_AWS_HOTWORDS,
        initial_prompt: str | None = None,
        allowed_languages: tuple[str, ...] | None = ("en",),
        min_language_probability: float = 0.35,
        max_no_speech_prob: float = 0.75,
        min_avg_logprob: float = -1.8,
        max_compression_ratio: float = 2.4,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.hotwords = hotwords
        self.initial_prompt = initial_prompt
        self.allowed_languages = tuple(l.lower() for l in allowed_languages) if allowed_languages else None
        self.min_language_probability = min_language_probability
        self.max_no_speech_prob = max_no_speech_prob
        self.min_avg_logprob = min_avg_logprob
        self.max_compression_ratio = max_compression_ratio
        self.model_name = f"faster-whisper:{model_size}"
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError(
                    "faster-whisper is required for local STT. "
                    "Install it with: pip install faster-whisper"
                ) from exc
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            return self._model

    def warmup(self) -> None:
        """Preload Whisper weights into memory at server startup."""
        self._get_model()

    def _filter_segments(self, segments) -> str:
        """Keep speech segments while dropping music/silence hallucinations and repetition loops."""
        kept: list[str] = []
        for seg in segments:
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
            compression_ratio = float(getattr(seg, "compression_ratio", 1.0) or 1.0)

            # Follow Whisper's combined silence rule: only drop for no_speech when confidence is also low
            if no_speech_prob > self.max_no_speech_prob and avg_logprob < -1.0:
                continue
            if avg_logprob < self.min_avg_logprob:
                continue
            if compression_ratio > self.max_compression_ratio:
                continue
            kept.append(text)
        return " ".join(kept).strip()

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> STTResult:
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")

        t0 = time.monotonic()
        model = self._get_model()

        suffix = Path(filename).suffix.lower() or ".webm"
        if suffix not in {".wav", ".webm", ".mp3", ".mp4", ".m4a", ".ogg", ".flac"}:
            suffix = ".webm"

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                transcribe_kwargs = {
                    "beam_size": self.beam_size,
                    "vad_filter": True,
                    "condition_on_previous_text": False,
                    "no_speech_threshold": self.max_no_speech_prob,
                    "log_prob_threshold": -1.0,
                    "compression_ratio_threshold": self.max_compression_ratio,
                }
                if self.hotwords:
                    transcribe_kwargs["hotwords"] = self.hotwords
                if self.initial_prompt:
                    transcribe_kwargs["initial_prompt"] = self.initial_prompt

                segments, info = model.transcribe(tmp_path, **transcribe_kwargs)
                detected_lang = (getattr(info, "language", "en") or "en").lower()
                lang_prob = float(getattr(info, "language_probability", 1.0))

                # Reject non-English audio (e.g. Hindi songs) or low-confidence noise/music
                if self.allowed_languages and (
                    detected_lang not in self.allowed_languages
                    or lang_prob < self.min_language_probability
                ):
                    # Exhaust generator cleanly if needed without accepting hallucinated text
                    transcript = ""
                else:
                    transcript = self._filter_segments(segments)
                    ref_vocab = " ".join(p for p in (self.hotwords, self.initial_prompt) if p)
                    if _is_hallucinated_transcript(transcript, reference_vocab=ref_vocab or None):
                        transcript = ""
            except Exception as exc:
                err_name = type(exc).__name__
                err_msg = str(exc).lower()
                if "eof" in err_name.lower() or "invaliddata" in err_name.lower() or "end of file" in err_msg:
                    # Truncated or header-only browser MediaRecorder blob (user stopped mic too fast)
                    return STTResult(
                        text="",
                        language="en",
                        language_probability=0.0,
                        duration_seconds=0.0,
                        latency_ms=round((time.monotonic() - t0) * 1000, 2),
                        model_name=self.model_name,
                    )
                raise

            latency_ms = (time.monotonic() - t0) * 1000

            return STTResult(
                text=transcript,
                language=detected_lang,
                language_probability=round(lang_prob, 4),
                duration_seconds=round(float(getattr(info, "duration", 0.0)), 3),
                latency_ms=round(latency_ms, 2),
                model_name=self.model_name,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


class MockSTT(STTAdapter):
    """Fast, deterministic offline STT adapter for unit tests and CI."""

    def __init__(
        self,
        default_text: str = "How many times does AWS Lambda retry a failed asynchronous invocation?",
        language: str = "en",
    ) -> None:
        self.default_text = default_text
        self.language = language
        self.model_name = "mock-stt-v1"

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> STTResult:
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")
        t0 = time.monotonic()

        # Special test convention: b"SILENCE" simulates silent microphone recording
        detected_language = self.language
        if audio_bytes.strip() == b"SILENCE":
            text = ""
        elif audio_bytes.startswith(b"LANG:"):
            detected_language = audio_bytes.removeprefix(b"LANG:").decode("utf-8", errors="ignore").strip() or "hi"
            text = ""
        elif audio_bytes.startswith(b"TEXT:"):
            text = audio_bytes.removeprefix(b"TEXT:").decode("utf-8", errors="ignore").strip()
        else:
            text = self.default_text

        latency_ms = (time.monotonic() - t0) * 1000
        return STTResult(
            text=text,
            language=detected_language,
            language_probability=0.99,
            duration_seconds=1.5,
            latency_ms=round(latency_ms, 2),
            model_name=self.model_name,
        )


_KNOWN_WHISPER_MODELS = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
    "large-v3-turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v3",
}


def make_stt_adapter(spec: str | None = None) -> STTAdapter:
    """Factory creating an STT adapter from a configuration string.

    Supported formats:
      - "mock"              -> MockSTT (offline CI / tests)
      - "whisper:tiny"      -> FasterWhisperSTT(model_size="tiny")
      - "whisper:base"      -> FasterWhisperSTT(model_size="base")
      - "whisper:small"     -> FasterWhisperSTT(model_size="small")  [default]
      - "whisper:small.en"  -> FasterWhisperSTT(model_size="small.en")
      - "whisper:medium"    -> FasterWhisperSTT(model_size="medium")
    """
    raw = (spec or os.environ.get("RESOLVEOPS_STT_MODEL") or "whisper:small").strip()
    lower = raw.lower()

    if lower in ("mock", "dummy"):
        return MockSTT()

    if lower.startswith("whisper:"):
        size = raw.split(":", 1)[1].strip() or "small"
        return FasterWhisperSTT(model_size=size)

    if lower in _KNOWN_WHISPER_MODELS:
        return FasterWhisperSTT(model_size=lower)

    raise ValueError(
        f"Unsupported STT spec '{raw}'. Expected 'mock' or 'whisper:<tiny|base|small|medium>'."
    )

