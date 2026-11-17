"""Unit tests for ResolveOps Voice Agent models, speech sanitizer, adapters, and pipeline."""

from __future__ import annotations

import pytest

from resolveops.voice.models import STTResult, TTSResult, VoiceQueryResponse, VoiceTimings
from resolveops.voice.sanitizer import sanitize_for_speech


def test_whisper_decodes_real_audio_without_model_download() -> None:
    """Exercise the real Whisper/PyAV boundary that mock STT tests cannot cover."""
    import io
    import wave

    audio_module = pytest.importorskip("faster_whisper.audio")
    recording = io.BytesIO()
    with wave.open(recording, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    recording.seek(0)

    samples = audio_module.decode_audio(recording, sampling_rate=16000)

    assert samples.shape == (1600,)
    assert samples.dtype.name == "float32"
    assert (samples == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Models & Natural Speech Sanitizer Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_sanitize_strips_citations_and_markdown() -> None:
    raw = (
        "**AWS Lambda** retries failed asynchronous invocations `twice` by default [1]. "
        "It adds delays between retry attempts [1, 2] and sends failures to a DLQ [p.42]."
    )
    cleaned = sanitize_for_speech(raw)
    assert "[1]" not in cleaned
    assert "[1, 2]" not in cleaned
    assert "[p.42]" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert cleaned == (
        "AWS Lambda retries failed asynchronous invocations twice by default. "
        "It adds delays between retry attempts and sends failures to a DLQ."
    )


def test_sanitize_expands_units_and_symbols() -> None:
    raw = "The item collection limit is 10 GB [1] with latency <= 300ms (e.g. 1 KB -> 400 KB)."
    cleaned = sanitize_for_speech(raw)
    assert "10 gigabytes" in cleaned
    assert "less than or equal to 300 milliseconds" in cleaned
    assert "for example," in cleaned
    assert "1 kilobyte to 400 kilobytes" in cleaned


def test_sanitize_replaces_code_blocks_and_tables() -> None:
    raw = (
        "### Configuration Steps\n"
        "Set the IAM role as shown below:\n"
        "```json\n{\"Version\": \"2012-10-17\"}\n```\n"
        "| Mode | Deletable |\n"
        "| --- | --- |\n"
        "| Governance | With permission |\n"
        "| Compliance | Never |\n"
        "- Enable Versioning\n"
        "- Configure Object Lock\n"
    )
    cleaned = sanitize_for_speech(raw)
    assert "Configuration Steps." in cleaned
    assert "See the screen for the code example." in cleaned
    assert "See the screen for the detailed table." in cleaned
    assert "Enable Versioning." in cleaned
    assert "Configure Object Lock." in cleaned
    assert "```" not in cleaned
    assert "|" not in cleaned


def test_sanitize_truncates_cleanly_at_sentence_boundary() -> None:
    raw = (
        "First sentence explains DynamoDB TTL clearly. "
        "Second sentence explains automatic deletion without write throughput. "
        "Third sentence goes into deep edge cases that exceed our spoken cap."
    )
    cleaned = sanitize_for_speech(raw, max_chars=130)
    assert cleaned.endswith("You can check the screen for the remaining details.")
    assert "First sentence explains DynamoDB TTL clearly." in cleaned
    assert len(cleaned) <= 130


def test_voice_models_validate_strictly() -> None:
    stt = STTResult(text="What is DynamoDB TTL?", language="en", language_probability=0.98, duration_seconds=2.1, latency_ms=340.5, model_name="faster-whisper:base")
    tts = TTSResult(audio_bytes=b"RIFFdemo", media_type="audio/wav", sample_rate=24000, duration_seconds=3.5, latency_ms=420.0, voice="af_heart", model_name="kokoro-82m")
    timings = VoiceTimings(stt_ms=340.5, retrieval_ms=25.0, rerank_ms=120.0, generation_ms=600.0, claim_audit_ms=200.0, tts_ms=420.0, total_ms=1705.5)

    resp = VoiceQueryResponse(
        transcript=stt.text,
        language=stt.language,
        answer="TTL deletes expired items [1].",
        spoken_text="TTL deletes expired items.",
        audio_id="aud_123",
        audio_url="/api/v1/voice/audio/aud_123",
        audio_media_type=tts.media_type,
        audio_duration_seconds=tts.duration_seconds,
        timings=timings,
        model_name="deepseek-chat",
        stt_model=stt.model_name,
        tts_model=tts.model_name,
    )
    assert resp.audio_url == "/api/v1/voice/audio/aud_123"
    assert resp.timings.total_ms == 1705.5



# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Audio Store, STT Adapters, and TTS Adapters Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_audio_store_lru_eviction() -> None:
    from resolveops.voice.audio_store import InMemoryAudioStore

    store = InMemoryAudioStore(max_entries=2)
    id1 = store.put(b"audio_1", media_type="audio/wav")
    id2 = store.put(b"audio_2", media_type="audio/mpeg")
    assert store.count() == 2

    # Access id1 so id2 becomes the least-recently-used clip
    clip1 = store.get(id1)
    assert clip1 is not None
    assert clip1.audio_bytes == b"audio_1"
    assert clip1.media_type == "audio/wav"

    # Inserting id3 must evict id2 (LRU), keeping id1 and id3
    id3 = store.put(b"audio_3", media_type="audio/wav")
    assert store.count() == 2
    assert store.get(id2) is None
    assert store.get(id1) is not None
    assert store.get(id3) is not None


def test_mock_stt_and_factory() -> None:
    from resolveops.voice.stt import FasterWhisperSTT, MockSTT, make_stt_adapter

    stt = make_stt_adapter("mock")
    assert isinstance(stt, MockSTT)

    # Normal mock transcription
    res = stt.transcribe(b"fake_webm_bytes", filename="mic.webm")
    assert "Lambda" in res.text
    assert res.language == "en"

    # Custom text payload for deterministic pipeline testing
    custom = stt.transcribe(b"TEXT:What is DynamoDB TTL?", filename="clip.wav")
    assert custom.text == "What is DynamoDB TTL?"

    # Silence detection produces empty string
    silent = stt.transcribe(b"SILENCE")
    assert silent.text == ""

    # Empty upload rejected
    with pytest.raises(ValueError, match="empty"):
        stt.transcribe(b"")

    # Factory parses Whisper variants without loading weights eagerly
    whisper_stt = make_stt_adapter("whisper:tiny")
    assert isinstance(whisper_stt, FasterWhisperSTT)
    assert whisper_stt.model_size == "tiny"

    with pytest.raises(ValueError, match="Unsupported STT spec"):
        make_stt_adapter("invalid:engine")


def test_mock_tts_valid_wav_and_lru_cache() -> None:
    from resolveops.voice.tts import EdgeTTS, KokoroTTS, MockTTS, make_tts_adapter

    tts = MockTTS()
    res1 = tts.synthesize("AWS Lambda retries twice by default.")
    assert res1.media_type == "audio/wav"
    assert res1.audio_bytes[:4] == b"RIFF"
    assert res1.audio_bytes[8:12] == b"WAVE"
    assert tts.call_count == 1

    # Synthesizing identical text hits LRU cache without recomputing
    res2 = tts.synthesize("AWS Lambda retries twice by default.")
    assert res2.audio_bytes == res1.audio_bytes
    assert tts.call_count == 1
    assert res2.latency_ms == 0.1

    # Factory creates natural neural adapters lazily
    kokoro = make_tts_adapter("kokoro:af_bella")
    assert isinstance(kokoro, KokoroTTS)
    assert kokoro.voice == "af_bella"

    edge = make_tts_adapter("edge:en-US-AndrewMultilingualNeural")
    assert isinstance(edge, EdgeTTS)
    assert edge.voice == "en-US-AndrewMultilingualNeural"

    with pytest.raises(ValueError, match="Unsupported TTS spec"):
        make_tts_adapter("unknown_tts")


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: VoicePipeline Orchestrator & FastAPI Voice Endpoints Tests
# ─────────────────────────────────────────────────────────────────────────────


def _build_mock_voice_pipeline(tts_fail_keyword: str | None = None):
    from resolveops.generation.grounding_verifier import MockGroundingVerifier
    from resolveops.generation.llm import MockLLMClient
    from resolveops.ingestion.chunking import Chunk
    from resolveops.ingestion.models import SourceType
    from resolveops.retrieval.bm25 import BM25Index
    from resolveops.retrieval.embeddings import MockEmbeddingModel
    from resolveops.retrieval.hybrid import HybridRetriever
    from resolveops.retrieval.reranker import MockReranker
    from resolveops.retrieval.vector_store import InMemoryVectorStore
    from resolveops.service import RAGService
    from resolveops.voice.pipeline import VoicePipeline
    from resolveops.voice.stt import MockSTT
    from resolveops.voice.tts import MockTTS

    embedder = MockEmbeddingModel(dimension=64)
    vstore = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vstore, bm25)

    retriever.add_chunks([
        Chunk(
            id="chk_lambda_retry",
            document_id="doc_lambda",
            source="invocation-async.md",
            source_type=SourceType.MARKDOWN,
            document_title="AWS Lambda Async Invocations",
            kind="text",
            content="When an asynchronous function fails, AWS Lambda retries the invocation twice by default, with delays between attempts.",
            block_ids=["b1"],
            context_block_ids=[],
            heading_context=["Asynchronous invocation", "Error handling"],
            page_numbers=[],
        )
    ])

    service = RAGService(
        retriever=retriever,
        reranker=MockReranker(),
        llm_client=MockLLMClient(),
        grounding_verifier=MockGroundingVerifier(),
    )
    stt = MockSTT()
    tts = MockTTS(fail_on_keyword=tts_fail_keyword)
    return VoicePipeline(rag_service=service, stt=stt, tts=tts), service, retriever


def test_voice_pipeline_end_to_end() -> None:
    pipeline, _service, _retriever = _build_mock_voice_pipeline()

    resp = pipeline.process_audio(
        audio_bytes=b"TEXT:How many times does AWS Lambda retry a failed asynchronous invocation?",
        filename="mic.webm",
    )

    assert resp.transcript == "How many times does AWS Lambda retry a failed asynchronous invocation?"
    assert "[1]" in resp.answer
    assert "[1]" not in resp.spoken_text
    assert resp.audio_id is not None
    assert resp.audio_url == f"/api/v1/voice/audio/{resp.audio_id}"
    assert resp.audio_media_type == "audio/wav"
    assert resp.tts_error is None
    assert resp.faithfulness_score == 1.0
    assert resp.timings.total_ms >= 0.0

    # Audio is retrievable from bounded cache
    cached = pipeline.get_audio(resp.audio_id)
    assert cached is not None
    assert cached.audio_bytes[:4] == b"RIFF"


def test_voice_pipeline_rejects_silence() -> None:
    from resolveops.voice.pipeline import EmptySpeechError

    pipeline, _service, _retriever = _build_mock_voice_pipeline()
    with pytest.raises(EmptySpeechError, match="No intelligible speech"):
        pipeline.process_audio(audio_bytes=b"SILENCE")


def test_voice_pipeline_degrades_gracefully_when_tts_fails() -> None:
    pipeline, _service, _retriever = _build_mock_voice_pipeline(tts_fail_keyword="Lambda")

    resp = pipeline.process_audio(
        audio_bytes=b"TEXT:How many times does AWS Lambda retry a failed asynchronous invocation?",
    )

    # Full grounded answer is still preserved even when TTS fails
    assert "[1]" in resp.answer
    assert resp.audio_id is None
    assert resp.audio_url is None
    assert resp.tts_error is not None
    assert "Simulated TTS failure" in resp.tts_error


def test_fastapi_voice_endpoints() -> None:
    from fastapi import HTTPException
    from resolveops.api.main import create_app

    pipeline, service, retriever = _build_mock_voice_pipeline()
    app = create_app(rag_service=service, retriever=retriever, voice_pipeline=pipeline)

    # 1. Verify OpenAPI schema & route registration for all voice endpoints
    route_map = {getattr(r, "path", ""): r for r in app.routes}
    assert "/health" in route_map
    assert "/api/v1/voice/query" in route_map
    assert "/api/v1/voice/audio/{audio_id}" in route_map

    # 2. Health endpoint reports voice readiness and active STT/TTS models
    health_data = route_map["/health"].endpoint()
    assert health_data["status"] == "healthy"
    assert health_data["dependencies"]["voice_ready"] is True
    assert health_data["dependencies"]["stt_model"] == "mock-stt-v1"
    assert health_data["dependencies"]["tts_model"] == "mock-tts-v1"

    # 3. Run pipeline and verify GET /api/v1/voice/audio/{audio_id} serves valid WAV
    voice_resp = pipeline.process_audio(
        audio_bytes=b"TEXT:How many times does AWS Lambda retry a failed asynchronous invocation?",
        filename="query.webm",
    )
    assert voice_resp.audio_id is not None

    audio_endpoint = route_map["/api/v1/voice/audio/{audio_id}"].endpoint
    audio_http_resp = audio_endpoint(voice_resp.audio_id)
    assert audio_http_resp.media_type == "audio/wav"
    assert audio_http_resp.body[:4] == b"RIFF"
    assert audio_http_resp.body[8:12] == b"WAVE"

    # 4. Unknown audio ID returns HTTP 404
    with pytest.raises(HTTPException) as exc_info:
        audio_endpoint("aud_doesnotexist")
    assert exc_info.value.status_code == 404

    # 5. Transcribe endpoint is registered for optional review-before-send mode
    assert "/api/v1/voice/transcribe" in route_map


def test_whisper_hallucination_and_prompt_echo_filtering() -> None:
    """Verify Whisper prompt echoes, music tags, and silence hallucinations are rejected without blocking real queries."""
    from resolveops.voice.stt import DEFAULT_AWS_HOTWORDS, _is_hallucinated_transcript

    # Exact regression case: Hindi song echoing the vocabulary list
    assert _is_hallucinated_transcript(
        "RDS Multi-AZ, Object Lock, TTL, JSON, API, CLI.",
        reference_vocab=DEFAULT_AWS_HOTWORDS,
    )
    assert _is_hallucinated_transcript("[Music]", reference_vocab=DEFAULT_AWS_HOTWORDS)
    assert _is_hallucinated_transcript("♪ ♫", reference_vocab=DEFAULT_AWS_HOTWORDS)
    assert _is_hallucinated_transcript("Thank you for watching.", reference_vocab=DEFAULT_AWS_HOTWORDS)

    # Genuine technical queries must pass cleanly
    assert not _is_hallucinated_transcript("What is Lambda retries?", reference_vocab=DEFAULT_AWS_HOTWORDS)
    assert not _is_hallucinated_transcript("DynamoDB TTL", reference_vocab=DEFAULT_AWS_HOTWORDS)
    assert not _is_hallucinated_transcript("How does RDS Multi-AZ failover work?", reference_vocab=DEFAULT_AWS_HOTWORDS)


def test_whisper_rejects_non_english_and_low_confidence_segments() -> None:
    """Exercise FasterWhisperSTT quality gates (language, logprob, no_speech_prob, hotwords vs initial_prompt)."""
    from types import SimpleNamespace
    from resolveops.voice.stt import FasterWhisperSTT

    captured_kwargs: dict = {}

    class FakeModel:
        def __init__(self, segments, info):
            self._segments = segments
            self._info = info

        def transcribe(self, path, **kwargs):
            captured_kwargs.clear()
            captured_kwargs.update(kwargs)
            return self._segments, self._info

    # 1. Hindi song (detected language = 'hi') must be rejected even if Whisper emits English words
    stt = FasterWhisperSTT(model_size="small")
    stt._model = FakeModel(
        segments=[SimpleNamespace(text="RDS Multi-AZ, Object Lock", no_speech_prob=0.1, avg_logprob=-0.4, compression_ratio=1.1)],
        info=SimpleNamespace(language="hi", language_probability=0.92, duration=4.0),
    )
    hindi_res = stt.transcribe(b"fake_audio_bytes", filename="song.webm")
    assert hindi_res.text == ""
    assert hindi_res.language == "hi"
    assert "initial_prompt" not in captured_kwargs
    assert "hotwords" in captured_kwargs
    assert captured_kwargs["condition_on_previous_text"] is False

    # 2. Very low log-probability / combined no-speech + low-confidence segments (music/noise) must be dropped
    stt._model = FakeModel(
        segments=[
            SimpleNamespace(text="garbled singing", no_speech_prob=0.1, avg_logprob=-2.1, compression_ratio=1.2),
            SimpleNamespace(text="quiet noise", no_speech_prob=0.85, avg_logprob=-1.2, compression_ratio=1.1),
        ],
        info=SimpleNamespace(language="en", language_probability=0.88, duration=3.0),
    )
    noisy_res = stt.transcribe(b"fake_audio_bytes", filename="noise.webm")
    assert noisy_res.text == ""

    # 2b. Normal accented speech with moderate avg_logprob (-1.1) must NOT be falsely dropped
    stt._model = FakeModel(
        segments=[
            SimpleNamespace(text="What is TTL expiry?", no_speech_prob=0.1, avg_logprob=-1.1, compression_ratio=1.1),
        ],
        info=SimpleNamespace(language="en", language_probability=0.85, duration=2.0),
    )
    accented_res = stt.transcribe(b"fake_audio_bytes", filename="accent.webm")
    assert accented_res.text == "What is TTL expiry?"

    # 3. Clean English technical question passes through intact
    stt._model = FakeModel(
        segments=[
            SimpleNamespace(text="What is Lambda retries?", no_speech_prob=0.05, avg_logprob=-0.25, compression_ratio=1.1),
        ],
        info=SimpleNamespace(language="en", language_probability=0.97, duration=2.2),
    )
    ok_res = stt.transcribe(b"fake_audio_bytes", filename="question.webm")
    assert ok_res.text == "What is Lambda retries?"
    assert ok_res.language == "en"


def test_voice_pipeline_rejects_non_english_audio_with_clear_message() -> None:
    """Non-English audio (e.g. Hindi song) triggers a clear EmptySpeechError instead of querying RAG."""
    from resolveops.voice.pipeline import EmptySpeechError

    pipeline, _service, _retriever = _build_mock_voice_pipeline()
    with pytest.raises(EmptySpeechError, match="English"):
        pipeline.process_audio(audio_bytes=b"LANG:hi")


def test_voice_pipeline_transcribe_audio_helper() -> None:
    """transcribe_audio returns STTResult directly for review-before-send mode."""
    pipeline, _service, _retriever = _build_mock_voice_pipeline()
    stt_res = pipeline.transcribe_audio(b"TEXT:What is Lambda retries?", filename="mic.webm")
    assert stt_res.text == "What is Lambda retries?"
    assert stt_res.language == "en"
