"""ResolveOps Voice Agent package (STT, natural speech sanitization, TTS, and orchestration)."""

from resolveops.voice.audio_store import InMemoryAudioStore, StoredAudio
from resolveops.voice.models import (
    STTResult,
    TTSResult,
    VoiceQueryResponse,
    VoiceTimings,
)
from resolveops.voice.pipeline import EmptySpeechError, VoicePipeline
from resolveops.voice.sanitizer import sanitize_for_speech
from resolveops.voice.stt import FasterWhisperSTT, MockSTT, STTAdapter, make_stt_adapter
from resolveops.voice.tts import EdgeTTS, KokoroTTS, MockTTS, TTSAdapter, make_tts_adapter

__all__ = [
    "EdgeTTS",
    "EmptySpeechError",
    "FasterWhisperSTT",
    "InMemoryAudioStore",
    "KokoroTTS",
    "MockSTT",
    "MockTTS",
    "STTAdapter",
    "STTResult",
    "StoredAudio",
    "TTSAdapter",
    "TTSResult",
    "VoicePipeline",
    "VoiceQueryResponse",
    "VoiceTimings",
    "make_stt_adapter",
    "make_tts_adapter",
    "sanitize_for_speech",
]
