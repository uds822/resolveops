"""Pluggable natural neural Text-to-Speech (TTS) adapters for ResolveOps Voice Agent."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import os
import struct
import threading
import time
import wave
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from resolveops.voice.models import TTSResult


def _encode_pcm16_wav(samples: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Encode float32 [-1.0, 1.0] or int16 audio array directly into in-memory WAV bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class _TTSCache:
    """Thread-safe bounded LRU memoization cache for repeated spoken answers."""

    def __init__(self, max_entries: int = 50) -> None:
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._data: OrderedDict[tuple[str, str], TTSResult] = OrderedDict()

    def get(self, key: tuple[str, str]) -> TTSResult | None:
        with self._lock:
            val = self._data.get(key)
            if val is not None:
                self._data.move_to_end(key)
            return val

    def put(self, key: tuple[str, str], value: TTSResult) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)


class TTSAdapter(ABC):
    """Abstract base contract for all Text-to-Speech engines."""

    model_name: str
    voice: str

    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """Convert clean spoken text into natural-sounding audio bytes."""
        ...


class KokoroTTS(TTSAdapter):
    """Fully local, offline, near-ElevenLabs neural TTS powered by Kokoro-82M.

    Recommended natural voices:
      - 'af_heart'   (warm, expressive American English female — default)
      - 'af_bella'   (smooth conversational female)
      - 'am_michael' (clear, calm American English male)
      - 'bf_emma'    (natural British English female)
    """

    SAMPLE_RATE = 24000

    def __init__(
        self,
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 1.0,
        cache_size: int = 50,
    ) -> None:
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.model_name = "kokoro-82m"
        self._pipeline = None
        self._lock = threading.Lock()
        self._cache = _TTSCache(max_entries=cache_size)

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                from kokoro import KPipeline
            except ImportError as exc:
                raise ImportError(
                    "kokoro is required for local KokoroTTS. "
                    "Install it with: pip install kokoro soundfile"
                ) from exc
            self._pipeline = KPipeline(lang_code=self.lang_code)
            return self._pipeline

    def warmup(self) -> None:
        """Preload Kokoro-82M neural weights into memory at server startup."""
        self._get_pipeline()

    def synthesize(self, text: str) -> TTSResult:
        clean = text.strip()
        if not clean:
            raise ValueError("text cannot be empty for TTS synthesis")

        cache_key = (f"{self.voice}:{self.speed}", clean)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"latency_ms": 0.1})

        t0 = time.monotonic()
        pipeline = self._get_pipeline()

        chunks: list[np.ndarray] = []
        for _gs, _ps, audio in pipeline(clean, voice=self.voice, speed=self.speed):
            if audio is not None and len(audio) > 0:
                chunks.append(np.asarray(audio, dtype=np.float32))

        if not chunks:
            raise RuntimeError("KokoroTTS generated zero audio samples")

        full_audio = np.concatenate(chunks)
        wav_bytes = _encode_pcm16_wav(full_audio, sample_rate=self.SAMPLE_RATE)
        duration = len(full_audio) / float(self.SAMPLE_RATE)
        latency_ms = (time.monotonic() - t0) * 1000

        result = TTSResult(
            audio_bytes=wav_bytes,
            media_type="audio/wav",
            sample_rate=self.SAMPLE_RATE,
            duration_seconds=round(duration, 2),
            latency_ms=round(latency_ms, 2),
            voice=self.voice,
            model_name=self.model_name,
        )
        self._cache.put(cache_key, result)
        return result


class EdgeTTS(TTSAdapter):
    """Ultra-natural expressive neural TTS via Microsoft Edge Neural Voices.

    Requires zero native system binaries on Windows (pure Python `pip install edge-tts`).
    Recommended voices:
      - 'en-US-AvaMultilingualNeural'    (warm, natural, human-like — default)
      - 'en-US-AndrewMultilingualNeural' (natural, confident male)
      - 'en-US-EmmaMultilingualNeural'   (clear, friendly female)
      - 'en-IN-NeerjaExpressiveNeural'   (natural Indian English female)
    """

    def __init__(
        self,
        voice: str = "en-US-AvaMultilingualNeural",
        rate: str = "+0%",
        cache_size: int = 50,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.model_name = "edge-tts-neural"
        self._cache = _TTSCache(max_entries=cache_size)

    async def _synthesize_async(self, text: str) -> bytes:
        try:
            import edge_tts
        except ImportError as exc:
            raise ImportError(
                "edge-tts is required for EdgeTTS. Install it with: pip install edge-tts"
            ) from exc

        communicate = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    def synthesize(self, text: str) -> TTSResult:
        clean = text.strip()
        if not clean:
            raise ValueError("text cannot be empty for TTS synthesis")

        cache_key = (f"{self.voice}:{self.rate}", clean)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"latency_ms": 0.1})

        t0 = time.monotonic()

        # Safely execute async synthesis whether inside an active event loop or sync thread
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                mp3_bytes = pool.submit(lambda: asyncio.run(self._synthesize_async(clean))).result()
        except RuntimeError:
            mp3_bytes = asyncio.run(self._synthesize_async(clean))

        if not mp3_bytes:
            raise RuntimeError("EdgeTTS returned empty audio stream")

        # Estimate spoken duration (~150 words per minute = 2.5 words/sec)
        word_count = max(1, len(clean.split()))
        est_duration = max(0.5, word_count / 2.5)
        latency_ms = (time.monotonic() - t0) * 1000

        result = TTSResult(
            audio_bytes=mp3_bytes,
            media_type="audio/mpeg",
            sample_rate=24000,
            duration_seconds=round(est_duration, 2),
            latency_ms=round(latency_ms, 2),
            voice=self.voice,
            model_name=self.model_name,
        )
        self._cache.put(cache_key, result)
        return result


class MockTTS(TTSAdapter):
    """Fast, deterministic offline TTS producing a valid minimal WAV stream for CI/tests."""

    SAMPLE_RATE = 24000

    def __init__(self, voice: str = "mock_natural", fail_on_keyword: str | None = None) -> None:
        self.voice = voice
        self.fail_on_keyword = fail_on_keyword
        self.model_name = "mock-tts-v1"
        self.call_count = 0
        self._cache = _TTSCache(max_entries=50)

    def synthesize(self, text: str) -> TTSResult:
        clean = text.strip()
        if not clean:
            raise ValueError("text cannot be empty for TTS synthesis")

        if self.fail_on_keyword and self.fail_on_keyword.lower() in clean.lower():
            raise RuntimeError(f"Simulated TTS failure triggered by '{self.fail_on_keyword}'")

        cache_key = (self.voice, clean)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"latency_ms": 0.1})

        self.call_count += 1
        t0 = time.monotonic()

        # Generate a structurally valid 0.15-second 16-bit PCM WAV header + silence frames
        num_frames = int(self.SAMPLE_RATE * 0.15)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(struct.pack(f"<{num_frames}h", *([0] * num_frames)))

        wav_bytes = buf.getvalue()
        latency_ms = (time.monotonic() - t0) * 1000
        result = TTSResult(
            audio_bytes=wav_bytes,
            media_type="audio/wav",
            sample_rate=self.SAMPLE_RATE,
            duration_seconds=0.15,
            latency_ms=round(latency_ms, 2),
            voice=self.voice,
            model_name=self.model_name,
        )
        self._cache.put(cache_key, result)
        return result


def make_tts_adapter(spec: str | None = None) -> TTSAdapter:
    """Factory creating a natural TTS adapter from a configuration string.

    Supported formats:
      - "mock"                                -> MockTTS (offline CI / tests)
      - "kokoro" or "kokoro:af_heart"         -> KokoroTTS (local offline Kokoro-82M)
      - "edge" or "edge:en-US-AvaMultilingualNeural" -> EdgeTTS (expressive neural voice)
    """
    raw = (spec or os.environ.get("RESOLVEOPS_TTS_MODEL") or "").strip()

    if not raw:
        # Auto-select whichever natural TTS library is installed in this environment
        try:
            __import__("kokoro")
            return KokoroTTS(voice="af_heart")
        except ImportError:
            pass
        try:
            __import__("edge_tts")
            return EdgeTTS(voice="en-US-AvaMultilingualNeural")
        except ImportError:
            return KokoroTTS(voice="af_heart")

    lower = raw.lower()
    if lower in ("mock", "dummy"):
        return MockTTS()

    if lower == "kokoro":
        return KokoroTTS(voice="af_heart")
    if lower.startswith("kokoro:"):
        voice = raw.split(":", 1)[1].strip() or "af_heart"
        return KokoroTTS(voice=voice)

    if lower == "edge":
        return EdgeTTS(voice="en-US-AvaMultilingualNeural")
    if lower.startswith("edge:"):
        voice = raw.split(":", 1)[1].strip() or "en-US-AvaMultilingualNeural"
        return EdgeTTS(voice=voice)

    raise ValueError(
        f"Unsupported TTS spec '{raw}'. Expected 'mock', 'kokoro[:<voice>]', or 'edge[:<voice>]'."
    )

