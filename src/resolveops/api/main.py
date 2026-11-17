"""Production FastAPI application exposing the ResolveOps RAG query and ingestion API."""

from __future__ import annotations

import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

from resolveops.generation.llm import MockLLMClient, make_llm_client
from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.identifiers import make_stable_id
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import make_embedding_model
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.manifest import InconsistentIndexError, generate_generation_id, load_manifest, verify_index_consistency
from resolveops.retrieval.reranker import make_reranker
from resolveops.retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore
from resolveops.service import RAGResponse, RAGService
from resolveops.voice import (
    EmptySpeechError,
    STTResult,
    VoicePipeline,
    VoiceQueryResponse,
    make_stt_adapter,
    make_tts_adapter,
)

MAX_VOICE_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB upload cap for voice recordings


# ─────────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """User search and question answering request payload."""

    query: str = Field(min_length=1, max_length=2000, description="The technical question to answer")
    candidate_pool: int = Field(default=10, ge=1, le=50, description="Number of candidate chunks in Stage 1")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of reranked chunks passed to the LLM")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM sampling temperature")


class IngestTextRequest(BaseModel):
    """Payload for directly indexing documentation text snippets into the knowledge base."""

    source: str = Field(min_length=1, max_length=500, description="Source filename or origin path (e.g. guide.md)")
    content: str = Field(min_length=1, max_length=1_000_000, description="Body text content to index (max 1MB)")
    document_title: str | None = Field(default=None, max_length=500, description="Human-readable title of document")
    heading_context: list[Annotated[str, StringConstraints(max_length=500)]] = Field(default_factory=list, max_length=20, description="Section heading hierarchy")
    page_numbers: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list, max_length=1000, description="PDF page numbers if applicable")
    replace: bool = Field(default=True, description="Replace existing document chunks for this source if present")


class IngestionBodyLimit:
    """Bound request bytes before FastAPI parses the JSON body."""

    def __init__(self, app, limit: int = 4_200_000):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/v1/ingest" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        chunks = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.limit:
                await JSONResponse({"detail": "Ingestion request is too large"}, status_code=413)(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


# ─────────────────────────────────────────────────────────────────────────────
# Application Factory
# ─────────────────────────────────────────────────────────────────────────────

def require_ingestion_key(x_api_key: str | None = Header(default=None)) -> None:
    """Fail closed for index mutations until an administrator configures a key."""
    configured = os.environ.get("RESOLVEOPS_INGEST_API_KEY")
    if not configured:
        raise HTTPException(status_code=503, detail="Ingestion is not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode("utf-8"), configured.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid ingestion credentials")


def preload_sample_corpus(retriever: HybridRetriever) -> int:
    """Preload sample AWS documentation on startup so the search index is ready immediately."""
    from resolveops.ingestion.chunking import chunk_document
    from resolveops.ingestion.document_parser import parse_document

    corpus_dir = Path(__file__).resolve().parents[3] / "data" / "raw" / "aws" / "markdown"
    if not corpus_dir.exists():
        return 0

    sample_files = [
        "invocation-async.md",
        "TTL.md",
        "Programming.Errors.md",
        "id_roles.md",
        "Versioning.md",
        "Concepts.MultiAZ.md",
        "object-lock.md",
        "id_credentials_mfa.md",
    ]

    total_chunks = 0
    for filename in sample_files:
        filepath = corpus_dir / filename
        if filepath.exists():
            doc = parse_document(filepath)
            chunks = chunk_document(doc)
            added = retriever.add_chunks(chunks)
            total_chunks += added

    return total_chunks


def build_default_rag_service() -> tuple[RAGService, HybridRetriever]:
    """Assemble default production RAG service dependencies with strict configuration validation."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass

    embed_spec = os.environ.get("RESOLVEOPS_EMBED_MODEL")
    if not embed_spec:
        embed_spec = "nvidia:nvidia/nemotron-3-embed-1b" if os.environ.get("NVIDIA_API_KEY") else "all-MiniLM-L6-v2"
    embedder = make_embedding_model(embed_spec)

    store_backend = os.environ.get("RESOLVEOPS_VECTOR_STORE", "inmemory").lower()
    manifest_path = Path(os.environ.get("RESOLVEOPS_MANIFEST_PATH", "./data/index_manifest.json")).resolve()
    bm25_path = Path(os.environ.get("RESOLVEOPS_BM25_PATH", "./data/bm25_index.json")).resolve()

    if store_backend == "qdrant":
        qdrant_path = os.environ.get("RESOLVEOPS_QDRANT_PATH", "./data/qdrant_db")
        qdrant_url = os.environ.get("RESOLVEOPS_QDRANT_URL")
        vector_store = QdrantVectorStore(
            embedding_model=embedder,
            path=qdrant_path if not qdrant_url else None,
            url=qdrant_url,
        )
        if bm25_path.is_file() and manifest_path.is_file():
            bm25 = BM25Index.load_snapshot(bm25_path)
            verify_index_consistency(manifest_path, vector_store, bm25)
        else:
            if vector_store.count() or bm25_path.exists() or manifest_path.exists():
                raise InconsistentIndexError("Incomplete persistent index: manifest and BM25 snapshot are required")
            bm25 = BM25Index()
    else:
        vector_store = InMemoryVectorStore(embedder)
        bm25 = BM25Index()

    retriever = HybridRetriever(vector_store, bm25, rrf_k=60)

    rerank_spec = os.environ.get("RESOLVEOPS_RERANK_MODEL")
    if not rerank_spec:
        rerank_spec = "nvidia:nvidia/nemotron-3.5-rerank-8b" if os.environ.get("NVIDIA_API_KEY") else "mock"
    reranker = make_reranker(rerank_spec)

    llm_spec = os.environ.get("RESOLVEOPS_LLM_MODEL")
    if not llm_spec:
        if os.environ.get("GROQ_API_KEY") or os.environ.get("Groq_Api_Key"):
            llm_spec = "groq:openai/gpt-oss-120b"
        elif os.environ.get("OPENROUTER_API_KEY"):
            llm_spec = "openrouter:deepseek/deepseek-chat"
        elif os.environ.get("NVIDIA_API_KEY"):
            llm_spec = "nvidia:meta/llama-3.2-90b-vision-instruct"
        else:
            raise RuntimeError(
                "No generative LLM configuration or provider API keys found. "
                "Set GROQ_API_KEY, OPENROUTER_API_KEY, NVIDIA_API_KEY, or explicitly set "
                "RESOLVEOPS_LLM_MODEL=mock to run in offline testing mode."
            )

    try:
        llm_client = make_llm_client(llm_spec)
    except Exception as exc:
        if llm_spec == "mock":
            llm_client = MockLLMClient()
        else:
            raise RuntimeError(
                f"Production configuration failure: cannot initialize LLM client '{llm_spec}': {exc}. "
                f"Set RESOLVEOPS_LLM_MODEL=mock to explicitly run in mock mode."
            ) from exc

    enable_audit = os.environ.get("RESOLVEOPS_ENABLE_CLAIM_AUDIT", "true").lower() in ("true", "1", "yes")
    verifier = None
    if enable_audit:
        if isinstance(llm_client, MockLLMClient):
            from resolveops.generation.grounding_verifier import MockGroundingVerifier

            verifier = MockGroundingVerifier()
        else:
            from resolveops.generation.grounding_verifier import LLMGroundingVerifier

            verifier = LLMGroundingVerifier(llm_client=llm_client)

    min_rerank_env = os.environ.get("RESOLVEOPS_MIN_RERANK_SCORE")
    min_rerank_score = float(min_rerank_env) if (min_rerank_env is not None and min_rerank_env.strip()) else 0.0

    service = RAGService(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
        grounding_verifier=verifier,
        min_rerank_score=min_rerank_score,
    )
    return service, retriever


def build_default_voice_pipeline(service: RAGService) -> VoicePipeline:
    """Assemble default STT + natural TTS VoicePipeline wrapping the active RAGService."""
    is_mock_llm = isinstance(service.llm_client, MockLLMClient)

    stt_spec = os.environ.get("RESOLVEOPS_STT_MODEL")
    if not stt_spec:
        stt_spec = "mock" if is_mock_llm else "whisper:small"
    stt = make_stt_adapter(stt_spec)

    tts_spec = os.environ.get("RESOLVEOPS_TTS_MODEL")
    if not tts_spec and is_mock_llm:
        tts_spec = "mock"
    tts = make_tts_adapter(tts_spec)

    return VoicePipeline(rag_service=service, stt=stt, tts=tts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads default models and preloads corpus on server startup."""
    if not hasattr(app.state, "rag_service") or app.state.rag_service is None:
        default_service, default_retriever = build_default_rag_service()
        app.state.rag_service = default_service
        app.state.retriever = default_retriever
        app.state.voice_pipeline = build_default_voice_pipeline(default_service)
        print(
            f"✅ [ResolveOps Startup] LLM={getattr(default_service.llm_client, 'model_name', 'unknown')} "
            f"| STT={app.state.voice_pipeline.stt.model_name} "
            f"| TTS={getattr(app.state.voice_pipeline.tts, 'model_name', 'none')}"
        )

        # Automatically index sample docs on server startup if currently empty
        if default_retriever.count() == 0:
            try:
                count = preload_sample_corpus(default_retriever)
                print(f"✅ [ResolveOps Startup] Successfully indexed {count} AWS documentation chunks.")
                if os.environ.get("RESOLVEOPS_VECTOR_STORE", "inmemory").lower() == "qdrant":
                    m_path = Path(os.environ.get("RESOLVEOPS_MANIFEST_PATH", "./data/index_manifest.json")).resolve()
                    b_path = Path(os.environ.get("RESOLVEOPS_BM25_PATH", "./data/bm25_index.json")).resolve()
                    default_retriever.save_state(m_path, b_path)
            except Exception as exc:
                print(f"⚠️ [ResolveOps Startup] Preload failed: {exc}")

    yield


def create_app(
    rag_service: RAGService | None = None,
    retriever: HybridRetriever | None = None,
    voice_pipeline: VoicePipeline | None = None,
) -> FastAPI:
    """FastAPI Application Factory for production and test environments."""

    # Only attach default lifespan loader if dependencies are NOT already provided
    app_lifespan = None if (rag_service is not None and retriever is not None) else lifespan

    app = FastAPI(
        title="ResolveOps RAG API",
        description="Production API for citation-grounded retrieval-augmented generation on technical documentation.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=app_lifespan,
    )

    app.add_middleware(IngestionBodyLimit)

    # Restrict CORS to configured origins
    cors_origins = [o.strip() for o in os.environ.get("RESOLVEOPS_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins else ["http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Attach injected dependencies immediately if provided
    app.state.rag_service = rag_service
    app.state.retriever = retriever
    if voice_pipeline is not None:
        app.state.voice_pipeline = voice_pipeline
    elif rag_service is not None:
        app.state.voice_pipeline = build_default_voice_pipeline(rag_service)
    else:
        app.state.voice_pipeline = None

    def persist_mutation(retriever_inst: HybridRetriever) -> None:
        if isinstance(retriever_inst.vector_store, QdrantVectorStore):
            retriever_inst.save_state(
                os.environ.get("RESOLVEOPS_MANIFEST_PATH", "./data/index_manifest.json"),
                os.environ.get("RESOLVEOPS_BM25_PATH", "./data/bm25_index.json"),
            )

    def mark_mutation_pending(retriever_inst: HybridRetriever) -> None:
        if isinstance(retriever_inst.vector_store, QdrantVectorStore):
            retriever_inst.vector_store.set_generation_id(generate_generation_id("pending"))

    # ── Health Check ─────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    def health_check() -> dict[str, Any]:
        retriever_inst: HybridRetriever = app.state.retriever
        service_inst: RAGService | None = getattr(app.state, "rag_service", None)
        voice_inst: VoicePipeline | None = getattr(app.state, "voice_pipeline", None)
        is_mock_llm = isinstance(service_inst.llm_client, MockLLMClient) if service_inst else True
        llm_model = getattr(service_inst.llm_client, "model_name", "unknown") if service_inst else "unknown"
        stt_model = getattr(voice_inst.stt, "model_name", None) if voice_inst else None
        tts_model = getattr(voice_inst.tts, "model_name", None) if (voice_inst and voice_inst.tts) else None

        return {
            "status": "healthy",
            "service": "ResolveOps RAG API",
            "version": "0.1.0",
            "indexed_chunks": retriever_inst.count() if retriever_inst else 0,
            "dependencies": {
                "retriever": "ready" if retriever_inst else "uninitialized",
                "llm_model": llm_model,
                "is_mock_llm": is_mock_llm,
                "voice_ready": voice_inst is not None,
                "stt_model": stt_model,
                "tts_model": tts_model,
            },
        }

    # ── Query Endpoint ───────────────────────────────────────────────────────
    @app.post("/api/v1/query", response_model=RAGResponse, tags=["RAG"])
    def query_endpoint(req: QueryRequest) -> RAGResponse:
        clean = req.query.strip()
        if not clean:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty or whitespace.",
            )

        service_inst: RAGService = app.state.rag_service
        try:
            return service_inst.answer_query(
                query=clean,
                candidate_pool=req.candidate_pool,
                top_k=req.top_k,
                temperature=req.temperature,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"RAG query pipeline execution failed: {str(exc)}",
            ) from exc

    async def _read_voice_upload(request: Request, x_audio_filename: str | None) -> tuple[bytes, str]:
        """Read uploaded voice audio with a strict 10 MB memory cap and derive a safe filename."""
        chunks: list[bytes] = []
        total_bytes = 0
        async for chunk in request.stream():
            total_bytes += len(chunk)
            if total_bytes > MAX_VOICE_AUDIO_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Audio upload exceeds maximum allowed size of 10 MB.",
                )
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)

        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio request body cannot be empty.",
            )

        content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        mime_to_ext = {
            "audio/wav": ".wav",
            "audio/wave": ".wav",
            "audio/x-wav": ".wav",
            "audio/webm": ".webm",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/mp4": ".mp4",
            "audio/ogg": ".ogg",
        }
        filename = x_audio_filename or f"recording{mime_to_ext.get(content_type, '.webm')}"
        return audio_bytes, filename

    # ── Voice Transcribe Endpoint (STT Only for Review/Edit Mode) ────────────
    @app.post("/api/v1/voice/transcribe", response_model=STTResult, tags=["Voice"])
    async def voice_transcribe_endpoint(
        request: Request,
        x_audio_filename: str | None = Header(default=None),
    ) -> STTResult:
        voice_inst: VoicePipeline | None = getattr(app.state, "voice_pipeline", None)
        if voice_inst is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Voice pipeline is not initialized.",
            )

        audio_bytes, filename = await _read_voice_upload(request, x_audio_filename)
        try:
            return await run_in_threadpool(
                voice_inst.transcribe_audio,
                audio_bytes=audio_bytes,
                filename=filename,
            )
        except EmptySpeechError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Voice engine dependency not installed: {str(exc)}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Voice transcription failed: {str(exc)}",
            ) from exc

    # ── Voice Query Endpoint (STT -> RAG -> Natural TTS) ─────────────────────
    @app.post("/api/v1/voice/query", response_model=VoiceQueryResponse, tags=["Voice"])
    async def voice_query_endpoint(
        request: Request,
        candidate_pool: int = Query(default=10, ge=1, le=50),
        top_k: int = Query(default=3, ge=1, le=10),
        temperature: float = Query(default=0.0, ge=0.0, le=1.0),
        x_audio_filename: str | None = Header(default=None),
    ) -> VoiceQueryResponse:
        voice_inst: VoicePipeline | None = getattr(app.state, "voice_pipeline", None)
        if voice_inst is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Voice pipeline is not initialized.",
            )

        audio_bytes, filename = await _read_voice_upload(request, x_audio_filename)

        try:
            return await run_in_threadpool(
                voice_inst.process_audio,
                audio_bytes=audio_bytes,
                filename=filename,
                candidate_pool=candidate_pool,
                top_k=top_k,
                temperature=temperature,
            )
        except EmptySpeechError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Voice engine dependency not installed: {str(exc)}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Voice pipeline execution failed: {str(exc)}",
            ) from exc

    # ── Voice Audio Playback Endpoint ────────────────────────────────────────
    @app.get("/api/v1/voice/audio/{audio_id}", tags=["Voice"])
    def voice_audio_endpoint(audio_id: str) -> Response:
        voice_inst: VoicePipeline | None = getattr(app.state, "voice_pipeline", None)
        if voice_inst is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Voice pipeline is not initialized.",
            )
        stored = voice_inst.get_audio(audio_id)
        if stored is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audio clip not found or expired from cache.",
            )
        return Response(
            content=stored.audio_bytes,
            media_type=stored.media_type,
            headers={"Cache-Control": "private, max-age=300"},
        )

    # ── Ingest Text Endpoint ─────────────────────────────────────────────────
    @app.post("/api/v1/ingest", tags=["Ingestion"], dependencies=[Depends(require_ingestion_key)])
    def ingest_text_endpoint(req: IngestTextRequest) -> dict[str, Any]:
        clean_content = req.content.strip()
        if not clean_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content cannot be empty or whitespace.",
            )

        retriever_inst: HybridRetriever = app.state.retriever
        source_type = SourceType.PDF if req.source.endswith(".pdf") else SourceType.MARKDOWN

        # Generate deterministic IDs across restarts
        doc_id = make_stable_id("doc", req.source)
        chunk_id = make_stable_id(
            "chunk",
            req.source,
            clean_content,
            json.dumps(req.heading_context, ensure_ascii=False),
            json.dumps(req.page_numbers),
        )

        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            source=req.source,
            source_type=source_type,
            document_title=req.document_title or req.source,
            kind="text",
            content=clean_content,
            block_ids=[f"block_{chunk_id}"],
            context_block_ids=[],
            heading_context=req.heading_context,
            page_numbers=req.page_numbers,
            indexable=True,
        )

        # Atomic document replacement: validates replacement chunks BEFORE deleting existing ones
        with retriever_inst._lock:
            mark_mutation_pending(retriever_inst)
            if req.replace:
                deleted_count, added = retriever_inst.replace_document(doc_id, [chunk])
            else:
                deleted_count = 0
                added = retriever_inst.add_chunks([chunk])
            persist_mutation(retriever_inst)
        return {
            "status": "success",
            "source": req.source,
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "chunks_deleted": deleted_count,
            "chunks_added": added,
            "total_indexed": retriever_inst.count(),
        }

    # ── Delete Document Endpoint ─────────────────────────────────────────────
    @app.delete("/api/v1/documents/{document_id}", tags=["Ingestion"], dependencies=[Depends(require_ingestion_key)])
    def delete_document_endpoint(document_id: str) -> dict[str, Any]:
        retriever_inst: HybridRetriever = app.state.retriever
        with retriever_inst._lock:
            mark_mutation_pending(retriever_inst)
            deleted = retriever_inst.delete_document(document_id)
            persist_mutation(retriever_inst)
        return {
            "status": "success",
            "document_id": document_id,
            "chunks_deleted": deleted,
            "total_indexed": retriever_inst.count(),
        }

    # Mount decoupled frontend static files at root / if directory exists
    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


# Production entrypoint instance
app = create_app()

