# ResolveOps — Production Engineering Plan

## 1. Goal & Vision

Build an enterprise-grade Retrieval-Augmented Generation (RAG) platform where users upload technical documentation (complex PDFs with tables, HTML docs, Markdown guides) and query them with verifiable, accurate citations.

This is a **production-first learning project**. We treat RAG primarily as a **data engineering and retrieval problem** (80% of RAG success is ingestion, chunking, and search quality; 20% is LLM prompting). We build and evaluate checkpoint by checkpoint, testing on real-world documentation.

---

## 2. Current Status

| Component | Status | Implementation Details |
|---|---|---|
| **Canonical Document & Block Models** | ✅ Completed | `models.py`, `normalization.py`, `identifiers.py` (strict Pydantic, SHA-256 stable IDs). |
| **Multi-Format Parsers** | ✅ Completed | `document_parser.py` (Dispatcher pattern), `html_parser.py`, `markdown_parser.py`. |
| **PDF Extraction Engines** | ✅ Completed | Fast baseline (`pymupdf`) + deep-learning layout engine (`docling_pdf_parser.py`). |
| **Docling Optimizations & Quality Gates** | ✅ Completed | `do_ocr=False`, merged-cell deduplication, automated structural checks (`needs_review`). |
| **Cloud Worker Packaging** | ✅ Ready | Dockerfile, Cloud Run Job worker (`pdf_job.py`), PowerShell deploy script. |
| **Table-Aware Semantic Chunker** | ✅ Completed | `chunking.py` (keeps tables whole, heading breadcrumbs, 1800 char budget, quarantine flag). |
| **Embedding Adapter** | ✅ Completed | `retrieval/embeddings.py` (Adapter interface, `LocalEmbeddingModel` MiniLM/BGE, offline tests). |
| **In-Memory Vector Store** | ✅ Completed | `retrieval/vector_store.py` (NumPy cosine similarity, quarantine filter, deduplication). |

---

## 3. Production Architecture Roadmap

We strictly separate the **Offline Ingestion & Indexing Pipeline** from the **Online Retrieval & Generation Pipeline**.

```
═══════════════════════════════════════════════════════════════════════════════════════════
PART 1: OFFLINE INGESTION & INDEXING PIPELINE (Phases A through H)
═══════════════════════════════════════════════════════════════════════════════════════════
  [ Raw Files ] ──► [ Pipeline Orchestrator ] ──► [ Ingestion Registry (SQLite/JSON) ]
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
         [ Parsers ]            [ Table-Aware Chunker ]
     (Docling / PyMuPDF)         (Breadcrumbs + Intact Tables)
               │                         │
               └────────────┬────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
        [ Dense Embedder ]        [ Sparse Indexer ]
       (MiniLM / BGE Embeddings)    (BM25 Lexical Tokens)
               │                         │
               └────────────┬────────────┘
                            │
                            ▼
            [ Durable Store & Metadata Index ]
                 (ChromaDB / SQLite Registry)

═══════════════════════════════════════════════════════════════════════════════════════════
PART 2: ONLINE RETRIEVAL & GENERATION PIPELINE (Phases I through L)
═══════════════════════════════════════════════════════════════════════════════════════════
  [ User Query ]
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
  [ Dense Search ]      [ Sparse Search ]     [ Metadata Filter ]
  (Vector Cosine)         (Exact BM25)        (Tenant / Doc / Page)
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
                   [ Reciprocal Rank Fusion ]
                               │
                               ▼
                   [ Cross-Encoder Reranker ]
                               │
                               ▼
               [ Context Assembly & Grounding Prompt ]
                               │
                               ▼
                  [ LLM Generation & Citations ]
```

---

## 4. Part 1: Offline Ingestion & Indexing Pipeline

### Phase A — Ingestion Pipeline Orchestration (Immediate Next Step)
* **Goal:** Create the single entrypoint connecting loader $\rightarrow$ parser $\rightarrow$ chunker $\rightarrow$ embedder $\rightarrow$ vector store.
* **Deliverables:**
  * `src/resolveops/ingestion/pipeline.py` with `ingest_document()`.
  * Structured return statuses: `IngestionResult` (`pages_parsed`, `chunks_created`, `tables_quarantined`, `elapsed_seconds`, `status`).
  * End-to-end integration test with real PDF/HTML/Markdown files.

### Phase B — Real-Document Quality Testing & Chunk Tuning
* **Goal:** Test and tune parsing and chunking on real AWS documentation and technical manuals.
* **Deliverables:**
  * Build a golden test set of 20 representative queries and expected source passages.
  * Benchmark chunking parameters: compare `max_chars` thresholds (800, 1200, 1800, 2500).
  * Evaluate chunk boundary overlap: test a small sliding window (100–150 chars) for prose text to prevent cross-boundary context loss.
  * Inspect table extraction fidelity on multi-page and borderless tables.

### Phase C — Metadata, Provenance & Access Controls
* **Goal:** Standardize metadata schemas for enterprise security and auditability.
* **Deliverables:**
  * Extend `Chunk.metadata` to include: `tenant_id`, `access_roles`, `document_version`, `file_hash`, `section_hierarchy`.
  * Ensure exact spatial bounding boxes (`bbox`) and page numbers are preserved across all chunk types.
  * Create permission-aware filtering specifications for retrieval.

### Phase D — Document Lifecycle Management
* **Goal:** Enable incremental updates, upserts, and clean deletions.
* **Deliverables:**
  * Implement SHA-256 file content hashing: if a file hasn't changed, skip re-parsing.
  * Implement atomic document replacement (delete old chunks by `document_id` before inserting updated chunks).
  * Implement soft-delete and purge mechanisms for GDPR / document removal.

### Phase E — Durable Indexing & Persistence (ChromaDB + BM25)
* **Goal:** Move beyond volatile RAM to persistent, production-ready storage.
* **Deliverables:**
  * Implement `ChromaVectorStore` satisfying the `VectorStore` interface with on-disk persistence.
  * Add parallel **BM25 Sparse Indexer** (using `rank_bm25` or SQLite FTS5) to store keyword tokens alongside dense vectors.
  * Lightweight Document Registry (SQLite) tracking document states (`pending`, `parsed`, `indexed`, `failed`).

### Phase F — Reliability & Scalability
* **Goal:** Ensure the pipeline handles crashes, timeouts, and large batches safely.
* **Deliverables:**
  * Worker retries with exponential backoff on transient parser failures.
  * Task timeouts to prevent hanging on corrupted PDF streams.
  * Batching for vector embeddings to optimize throughput and memory usage.
  * Bounded workers: decouple heavy PDF parsing from the main API process.

### Phase G — Observability, Security & Versioning
* **Goal:** Enterprise production readiness and telemetry.
* **Deliverables:**
  * Structured JSON logging across all pipeline steps.
  * Pipeline versioning: tag indexed chunks with `pipeline_version` and `parser_engine_version`.
  * Index rollback procedure: allow switching back to a previous index version without downtime.
  * Sensitive data masking / sanitization checks before indexing.

### Phase H — Retrieval Evaluation Framework
* **Goal:** Mathematically measure retrieval accuracy before connecting any LLM.
* **Deliverables:**
  * Offline query evaluation harness testing:
    * **Recall@K** (Does top-K contain the answer?)
    * **MRR (Mean Reciprocal Rank)** (How high does the right chunk appear?)
    * **NDCG (Normalized Discounted Cumulative Gain)** (Ranking quality).
  * Establish baseline retrieval scores on the Golden Test Set.

---

## 5. Part 2: Online Retrieval & Generation Pipeline

Once the offline pipeline is verified and benchmarked:

### Phase I — Online Retrieval API & Filtering
* Query embedding generation with caching.
* Pre-filtering search space by `tenant_id`, `document_id`, and `source_type`.

### Phase J — Hybrid Search & Reciprocal Rank Fusion (RRF)
* Run Dense Vector search (semantic) + BM25 search (lexical) in parallel.
* Merge results using **Reciprocal Rank Fusion (RRF)** to get the best of both keyword precision and semantic meaning.

### Phase K — Cross-Encoder Reranking
* Pass fused Top-20 results through a cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`).
* Select Top-3 to Top-5 highest-confidence chunks for the final prompt context.

### Phase L — Context Assembly, LLM Client & Grounded Generation
* Construct strict citation-enforced prompts with fallback handling for unanswerable queries.
* Pluggable `LLMClient` adapter (Google Gemini, OpenAI, Ollama).
* Strict citation verification and highlighting in the UI.

---

## 6. Execution Strategy: The Gradual Step-by-Step Rule

To ensure deep learning and zero regressions:
1. **One sub-step at a time:** Discuss the scenario, design the interface, write the code, write the test, and inspect terminal output.
2. **Never break existing tests:** Every new phase must pass the entire test suite (`pytest`).
3. **Inspect real data:** Always test with real AWS PDF, HTML, and Markdown files rather than purely mock data.
