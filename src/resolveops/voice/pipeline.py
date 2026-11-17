"""Voice Pipeline Orchestrator wiring STT -> RAGService -> Speech Sanitizer -> TTS."""

from __future__ import annotations

import time

from resolveops.service import RAGService
from resolveops.voice.audio_store import InMemoryAudioStore, StoredAudio
from resolveops.voice.models import STTResult, VoiceQueryResponse, VoiceTimings
from resolveops.voice.sanitizer import sanitize_for_speech
from resolveops.voice.stt import STTAdapter
from resolveops.voice.tts import TTSAdapter


class EmptySpeechError(ValueError):
    """Raised when STT detects silence or no intelligible speech in the uploaded audio."""


class VoicePipeline:
    """End-to-end multimodal voice orchestrator wrapping ResolveOps RAGService.

    Pipeline flow:
      1. Stage 0 (STT): Transcribe uploaded audio bytes -> text query.
      2. Guard: Reject silent, non-English, music, or hallucinated transcripts early with EmptySpeechError (HTTP 422).
      3. Stages 1-4 (RAG): Run Hybrid Search -> Rerank -> Grounded Concise LLM -> Tier 2 Claim Audit.
      4. Sanitizer: Convert Markdown + [1] citations into natural spoken prose.
      5. Stage 5 (TTS): Synthesize spoken audio and store in bounded LRU cache.
         If TTS fails, degrade gracefully and still return the verified text + citations.
    """

    def __init__(
        self,
        rag_service: RAGService,
        stt: STTAdapter,
        tts: TTSAdapter | None = None,
        audio_store: InMemoryAudioStore | None = None,
        max_spoken_chars: int = 900,
        voice_max_tokens: int = 650,
    ) -> None:
        self.rag_service = rag_service
        self.stt = stt
        self.tts = tts
        self.audio_store = audio_store or InMemoryAudioStore(max_entries=50)
        self.max_spoken_chars = max_spoken_chars
        self.voice_max_tokens = voice_max_tokens

    def get_audio(self, audio_id: str) -> StoredAudio | None:
        """Retrieve cached synthesized audio bytes by ID."""
        return self.audio_store.get(audio_id)

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
    ) -> STTResult:
        """Run Stage 0 STT with silence, music, and non-English speech guards."""
        if not audio_bytes:
            raise ValueError("Uploaded audio payload is empty")

        stt_res = self.stt.transcribe(audio_bytes, filename=filename)
        transcript = stt_res.text.strip()
        if len(transcript) < 2:
            detected = (stt_res.language or "en").lower()
            if detected and detected != "en":
                raise EmptySpeechError(
                    f"No clear English speech detected (audio sounded like '{detected}' or background music). "
                    "Please speak your technical question clearly in English and try again."
                )
            raise EmptySpeechError(
                "No intelligible speech detected in the recording. Please speak clearly and try again."
            )
        return stt_res

    def process_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        candidate_pool: int = 10,
        top_k: int = 3,
        temperature: float = 0.0,
    ) -> VoiceQueryResponse:
        """Execute the full voice-to-voice RAG pipeline on raw uploaded audio bytes."""
        if not audio_bytes:
            raise ValueError("Uploaded audio payload is empty")

        t_start = time.monotonic()

        # ── Stage 0: Speech-to-Text (Ears) ────────────────────────────────────
        stt_res = self.transcribe_audio(audio_bytes, filename=filename)
        transcript = stt_res.text.strip()

        # ── Stages 1–4: Existing Grounded RAG Pipeline (Brain) ────────────────
        rag_res = self.rag_service.answer_query(
            query=transcript,
            candidate_pool=candidate_pool,
            top_k=top_k,
            max_tokens=self.voice_max_tokens,
            temperature=temperature,
            response_style="voice",
        )

        # ── Speech Sanitization (Markdown + [1] -> Natural Spoken Prose) ──────
        spoken_text = sanitize_for_speech(rag_res.answer, max_chars=self.max_spoken_chars)

        # ── Stage 5: Text-to-Speech (Mouth) with Graceful Degradation ─────────
        audio_id: str | None = None
        audio_url: str | None = None
        audio_media_type: str | None = None
        audio_duration: float | None = None
        tts_ms = 0.0
        tts_error: str | None = None
        tts_model_name = getattr(self.tts, "model_name", None) if self.tts else None

        if self.tts is not None and spoken_text:
            try:
                tts_res = self.tts.synthesize(spoken_text)
                audio_id = self.audio_store.put(tts_res.audio_bytes, media_type=tts_res.media_type)
                audio_url = f"/api/v1/voice/audio/{audio_id}"
                audio_media_type = tts_res.media_type
                audio_duration = tts_res.duration_seconds
                tts_ms = tts_res.latency_ms
                tts_model_name = tts_res.model_name
            except Exception as exc:
                # Never drop a verified RAG answer just because TTS failed
                tts_error = str(exc)

        total_ms = (time.monotonic() - t_start) * 1000

        timings = VoiceTimings(
            stt_ms=round(stt_res.latency_ms, 2),
            retrieval_ms=rag_res.timings.retrieval_ms,
            rerank_ms=rag_res.timings.rerank_ms,
            generation_ms=rag_res.timings.generation_ms,
            claim_audit_ms=rag_res.timings.claim_audit_ms,
            tts_ms=round(tts_ms, 2),
            total_ms=round(total_ms, 2),
        )

        return VoiceQueryResponse(
            transcript=transcript,
            language=stt_res.language,
            answer=rag_res.answer,
            spoken_text=spoken_text,
            citations=rag_res.citations,
            retrieved_citations=rag_res.retrieved_citations,
            cited_citations=rag_res.cited_citations,
            hallucinated_markers=rag_res.hallucinated_markers,
            citation_validation_failed=rag_res.citation_validation_failed,
            faithfulness_score=rag_res.faithfulness_score,
            claim_audit=rag_res.claim_audit,
            audio_id=audio_id,
            audio_url=audio_url,
            audio_media_type=audio_media_type,
            audio_duration_seconds=audio_duration,
            tts_error=tts_error,
            timings=timings,
            model_name=rag_res.model_name,
            stt_model=stt_res.model_name,
            tts_model=tts_model_name,
        )

