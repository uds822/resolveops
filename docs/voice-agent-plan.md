# ResolveOps — Voice Agent Integration Plan

## 1. Goal & Vision

Add a fully local, zero-cost voice interface to the ResolveOps RAG platform.  
Users will be able to **speak a question** and receive a **natural-sounding spoken answer** grounded in the indexed technical documentation — with the same page-level citation quality as the text pipeline.

No external paid APIs. Everything runs offline on CPU.

---

## 2. Technology Stack

### 2.1 Speech-to-Text (STT) — faster-whisper

| Property | Detail |
|---|---|
| Library | `faster-whisper` (CTranslate2-optimized Whisper) |
| Model Size | `tiny` (CPU demo) → `base` or `small` (production) |
| Language Support | Multilingual (English primary) |
| Latency | ~0.3–1.5s per utterance on CPU |
| Cost | 100% free, fully offline |
| Install | `pip install faster-whisper` |

**Why not alternatives:**
- Google Speech API: 60 min/month then paid. Fragile for a portfolio project.
- Browser WebSpeech API: Chrome-only, no server-side control, unreliable.
- Original `openai-whisper`: 4× slower than `faster-whisper` on the same hardware.

---

### 2.2 Text-to-Speech (TTS) — Kokoro TTS

| Property | Detail |
|---|---|
| Library | `kokoro` (via HuggingFace `hexgrad/Kokoro-82M`) |
| Model Size | ~350MB (82M parameter neural TTS) |
| Voice Quality | Near-ElevenLabs quality, natural prosody |
| Output Format | WAV / PCM |
| Latency | ~0.5–2s for a typical answer paragraph |
| Cost | 100% free, fully offline |
| Install | `pip install kokoro soundfile` |

**Why not alternatives:**
- Edge-TTS (Microsoft): Requires internet, Microsoft servers. Not fully offline.
- gTTS (Google): Robotic, low quality. Not suitable for a production showcase.
- Coqui TTS: Good but significantly slower than Kokoro on CPU. Heavier setup.
- ElevenLabs: Only 10,000 chars/month free. Runs out quickly during demos.
- Piper TTS: Excellent backup option — slightly lower voice quality than Kokoro but faster.

---

## 3. Architecture

### 3.1 Full End-to-End Voice Flow

```
[Browser / Client]
       │
       │  (1) Record audio (WAV/WebM blob)
       ▼
[FastAPI: POST /voice/query]
       │
       │  (2) faster-whisper STT → transcribed text
       ▼
[Existing RAG Pipeline]
       │
       ├── Hybrid Search (Dense + BM25 + RRF)
       ├── Cross-Encoder Reranking
       └── LLM Grounded Generation + Citations
       │
       │  (3) Answer text with page citations
       ▼
[Kokoro TTS → WAV bytes]
       │
       │  (4) Return: { "transcript": "...", "answer": "...", "citations": [...], "audio_url": "..." }
       ▼
[Browser plays audio + displays citations]
```

### 3.2 API Endpoints (New)

| Method | Endpoint | Input | Output |
|---|---|---|---|
| `POST` | `/voice/query` | Audio file (WAV/WebM) | JSON + audio URL |
| `GET` | `/voice/audio/{id}` | Audio ID | WAV audio stream |
| `WebSocket` | `/voice/stream` | Audio chunks | Streaming text + audio (Phase 2) |

### 3.3 Module Structure

```
src/resolveops/
  voice/
    __init__.py
    stt.py           ← faster-whisper STT adapter
    tts.py           ← Kokoro TTS adapter
    voice_pipeline.py ← Orchestrates STT → RAG → TTS
    models.py        ← VoiceQueryRequest, VoiceQueryResponse Pydantic models

api/
  routers/
    voice.py         ← FastAPI router: /voice/query, /voice/audio/{id}
```

---

## 4. Implementation Phases

### Phase V1 — Core STT Adapter (`voice/stt.py`)

**Goal:** Accept a raw audio file, return clean transcribed text.

**Deliverables:**
- `WhisperSTT` class wrapping `faster-whisper`
- Support WAV and WebM input formats
- Configurable model size: `tiny` / `base` / `small`
- Returns `STTResult(text: str, language: str, confidence: float)`
- Unit test with a pre-recorded audio fixture

**Key Design Decisions:**
- Load model once at startup (not per request) to avoid cold start penalty
- Use `beam_size=5` for accuracy, `beam_size=1` for low-latency mode
- Language auto-detection enabled by default

---

### Phase V2 — Core TTS Adapter (`voice/tts.py`)

**Goal:** Accept answer text, return a natural-sounding WAV audio stream.

**Deliverables:**
- `KokoroTTS` class wrapping the Kokoro model
- Configurable voice (default: `af_heart` — natural female English)
- Output: WAV bytes in memory (no temp file I/O on hot path)
- Returns `TTSResult(audio_bytes: bytes, duration_seconds: float)`
- Unit test validating non-empty WAV output for a sample sentence

**Key Design Decisions:**
- Strip citation references from spoken text (e.g. `[p.42]` → not spoken)
- Split long answers into sentence-level chunks for lower first-token latency
- Cache TTS output for identical repeated answers (LRU cache, 50-entry limit)

---

### Phase V3 — Voice Pipeline Orchestrator (`voice/voice_pipeline.py`)

**Goal:** Wire STT → RAG → TTS into a single callable function.

**Deliverables:**
- `run_voice_query(audio_bytes: bytes) -> VoiceQueryResponse`
- `VoiceQueryResponse`: transcript, answer text, citations list, audio bytes, latency breakdown
- Structured timing: `stt_ms`, `retrieval_ms`, `generation_ms`, `tts_ms`, `total_ms`
- Graceful error handling: STT failure returns 422, TTS failure returns text-only answer

---

### Phase V4 — FastAPI Voice Router (`api/routers/voice.py`)

**Goal:** Expose the voice pipeline as production HTTP endpoints.

**Deliverables:**
- `POST /voice/query` accepting `multipart/form-data` audio upload
- `GET /voice/audio/{session_id}` serving the WAV response with correct MIME type
- Request size limit: 10MB audio cap
- Response includes: `transcript`, `answer`, `citations`, `audio_url`, `latency`
- Integration test using `httpx.AsyncClient` with a real audio fixture

---

### Phase V5 — Browser Voice UI (Frontend)

**Goal:** Simple, clean browser UI for voice interaction.

**Deliverables:**
- Record button using `MediaRecorder API` (WebM/Opus captured in browser)
- Visual waveform animation while recording
- Auto-submit on silence detection (500ms threshold) or manual stop
- Display: spoken answer plays automatically, citation cards shown below
- Works in Chrome, Firefox, Safari (no extensions needed)

**Stack:** Plain HTML + Vanilla JS + Web Audio API (no React/Angular dependency)

---

### Phase V6 — Streaming Voice (WebSocket, Optional)

**Goal:** Real-time streaming for lower perceived latency on long answers.

**Deliverables:**
- WebSocket `/voice/stream` endpoint
- Server streams: `{type: "transcript", text: "..."}` → then `{type: "audio_chunk", bytes: "..."}` as TTS renders sentences progressively
- Client plays audio chunks as they arrive using AudioContext

> [!NOTE]
> Phase V6 is optional / stretch goal. The core experience is fully functional from V1–V5.

---

## 5. Data Flow & Latency Budget

For a typical 5-second voice query and a 2–3 sentence answer:

| Stage | Component | Target Latency |
|---|---|---|
| Audio upload | HTTP multipart | ~100ms |
| Transcription | faster-whisper `base` | ~400ms |
| Embedding query | MiniLM local | ~50ms |
| Hybrid search | BM25 + Vector + RRF | ~30ms |
| Cross-encoder rerank | ms-marco-MiniLM | ~200ms |
| LLM generation | Gemini / Ollama | ~800ms |
| TTS synthesis | Kokoro | ~600ms |
| **Total** | | **~2.2s** |

> [!TIP]
> Run STT transcription and model warmup at server startup. The first request after a cold start will take ~3–5s while Whisper and Kokoro load into memory. All subsequent requests hit the budget above.

---

## 6. Dependencies to Add

```toml
# pyproject.toml — new optional group
[project.optional-dependencies]
voice = [
    "faster-whisper>=1.0",
    "kokoro>=0.9",
    "soundfile>=0.12",
    "numpy>=1.26",
]
```

Install with:
```bash
pip install -e ".[voice]"
```

---

## 7. Resume Update (After Voice Is Built)

Replace the current 4th bullet with:

> **\textbf{Multimodal Voice Agent \& Cloud Deployment:}** Integrated a fully local, zero-cost voice interface using \textbf{faster-whisper} (STT) and \textbf{Kokoro TTS} for natural-sounding spoken answers; grounded responses with exact page and section citations served via FastAPI, containerized on serverless \textbf{GCP Cloud Run} with SHA-256 idempotent deduplication across \textbf{20,000+ pages} of AWS documentation.

---

## 8. Build Order (1 Week Sprint)

| Day | Task |
|---|---|
| **Day 1** | Phase V1: STT adapter + unit tests |
| **Day 2** | Phase V2: TTS adapter + unit tests |
| **Day 3** | Phase V3: Voice pipeline orchestrator |
| **Day 4** | Phase V4: FastAPI voice router + integration test |
| **Day 5** | Phase V5: Browser voice UI |
| **Day 6** | End-to-end demo test on real AWS documentation queries |
| **Day 7** | Polish, latency tuning, update resume + README |

---

## 9. Open Questions / Decisions

| Question | Decision |
|---|---|
| Which Kokoro voice to use? | `af_heart` (natural female) as default; make it configurable |
| Handle non-English queries? | Auto-detect language in Whisper, respond in English always |
| Audio format from browser? | Accept WebM (Chrome/Firefox default) + WAV; convert with `soundfile` if needed |
| Session storage for audio files? | In-memory LRU (dev) → GCS bucket (production) |
| Streaming (Phase V6)? | Implement after V1–V5 are stable |

