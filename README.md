# ⚡ ResolveOps

> **Enterprise-Grade Hybrid RAG Platform for Technical Documentation with Verifiable Citation Provenance.**

ResolveOps is an end-to-end Retrieval-Augmented Generation (RAG) platform engineered specifically for complex, high-stakes technical documentation (AWS, Kubernetes, infrastructure guides). It combines **multi-engine layout-aware parsing**, **hybrid dense-sparse retrieval with a 3-way consistency handshake**, **two-stage cross-encoder reranking**, and **strict citation-grounded generation** to deliver mathematically verifiable, zero-hallucination answers.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion ["1. Ingestion Pipeline"]
        PDF[PDF / Markdown / HTML Docs] --> Router[Document Parser Dispatcher]
        Router -->|Complex Manuals| Docling[Docling Vision Engine<br/>Borderless Table Recovery]
        Router -->|Fast Path| PyMuPDF[PyMuPDF Digital Text]
        Docling --> Chunker[Table-Aware Semantic Chunker<br/>• Heading Breadcrumbs<br/>• Whole-Table Preservation]
        PyMuPDF --> Chunker
    end

    subgraph Storage ["2. Persistent Dual-Store & Manifest"]
        Chunker --> Embedder[NVIDIA Nemotron-1B Embedder<br/>2048 Dimensions]
        Embedder --> Qdrant[(Qdrant Vector Store<br/>Local / Remote + Sentinel Point)]
        Chunker --> BM25[(Durable BM25 Snapshots<br/>Technical Tokenizer)]
        Manifest{{Index Manifest Coordinator<br/>3-Way Triangular Handshake}} -. Verifies .-> Qdrant
        Manifest -. Verifies .-> BM25
    end

    subgraph Retrieval ["3. Two-Stage Retrieval Funnel"]
        Query[User Question] --> S1_Dense[Qdrant Dense Cosine]
        Query --> S1_Sparse[BM25 Technical Keyword]
        S1_Dense --> RRF[Reciprocal Rank Fusion<br/>k=60 Consensus Scoring]
        S1_Sparse --> RRF
        RRF --> Pool[Candidate Pool<br/>Top-10 Chunks]
        Pool --> Reranker[NVIDIA Nemotron 3.5 8B<br/>Cross-Encoder Cross-Attention]
        Reranker --> TopK[Top-3 High-Precision Passages]
    end

    subgraph Generation ["4. Grounding & Generation"]
        TopK --> Builder[Grounded Context Builder<br/>• Numbered Citations [1], [2]<br/>• Anti-Hallucination Contract]
        Builder --> LLM[DeepSeek-V3 / Llama 3.2 90B]
        LLM --> Sanitizer[Citation Audit & Refusal Sanitizer]
        Sanitizer --> Response[Verified Answer with Provenance]
    end

    subgraph Serving ["5. Delivery & UI"]
        Response --> API[FastAPI REST Engine]
        API --> Studio[ResolveOps RAG Studio UI<br/>Clickable Citations & Telemetry]
    end
```

---

## ✨ Key Capabilities

### 1. Multi-Engine Layout & Table-Aware Ingestion
- **Docling Vision Layout Engine**: Preserves complex borderless and merged-cell tables in technical manuals without cell duplication.
- **Whole-Table Preservation**: Tables are emitted as atomic units—never sliced across arbitrary token boundaries.
- **Heading Context Breadcrumbs**: Automatically tracks heading hierarchies ($H_1 \rightarrow H_2 \rightarrow H_3$) and prepends breadcrumbs to each chunk, eliminating pronoun ambiguity.
- **Safety Quarantine Gates**: Automatically inspects table extraction quality (`empty_table`, `ragged_rows`, `cell_outside_grid`) and flags corrupted tables as `"needs_review"`, preventing bad data from entering the search index.

### 2. Dual-Store Persistence & 3-Way Consistency Handshake
- **Persistent Qdrant Vector Store**: Embedded local on-disk storage (`data/qdrant_db`) or remote server with 2048-dim cloud **NVIDIA Nemotron** embeddings (`nvidia/nemotron-3-embed-1b`).
- **Sentinel System Point**: Uses a dedicated UUID sentinel (`00000000-0000-0000-0000-000000000000`) inside Qdrant to store generation IDs and schema versions, completely isolated from user search queries.
- **Durable BM25 Snapshot Engine**: Preserves technical tokens (e.g. dotted versions `3.3.2`, snake_case, hyphens, and file extensions `pom.xml`) with atomic JSON serialization.
- **Authoritative 3-Way Handshake**: On server startup, `IndexManifest` cryptographically validates:
  $$\text{manifest.generation\_id} \equiv \text{qdrant.get\_generation\_id()} \equiv \text{bm25.get\_generation\_id()}$$
  as well as chunk volume equality and SHA-256 BM25 checksums to prevent cross-store index drift.

### 3. Two-Stage Retrieval (RRF + 8B Cross-Encoder)
- **Stage 1 — Candidate Funnel**: Combines dense vector cosine similarity and sparse BM25 scores using Reciprocal Rank Fusion:
  $$\text{RRF Score} = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \text{rank}_m} \quad (k=60)$$
- **Stage 2 — Cross-Encoder Reranker**: Employs **NVIDIA Nemotron 3.5 8B** (`nvidia/nemotron-3.5-rerank-8b`) to compute deep cross-attention over `(query, passage)` pairs, eliminating the "keyword trap" and promoting passages with direct answer entailment.

### 4. Zero-Hallucination Generation & Citation Audit
- **Strict Grounding Contract**: Every factual claim, code snippet, and configuration parameter is tethered to a numbered citation marker (`[1]`, `[2]`).
- **Refusal Sanitization**: If a question is out-of-scope or unanswerable, the system explicitly refuses without phantom citations (`Citations: 0`), preventing false citation illusions.
- **Pluggable LLM Adapters**: Native support for **DeepSeek-V3** (`openrouter:deepseek/deepseek-chat`) and **NVIDIA Llama-3.2-90B**, with an offline deterministic `MockLLMClient` for fast CI testing.

### 5. ResolveOps Studio UI
- Native **Flexbox layout** with independent, smooth scrolling containers.
- **Interactive Citation & Provenance Audit Panel**: Clicking any citation chip (`[1]`, `[2]`) in the chat highlights the exact source document, section breadcrumb hierarchy, and PDF page numbers in the sidebar.
- **Live Telemetry Waterfall**: Real-time breakdown of `Retrieval Latency (ms)`, `Rerank Latency (ms)`, and `Generation Latency (ms)`.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- NVIDIA NIM API Key (for Nemotron embeddings and reranker)
- OpenRouter API Key (for DeepSeek-V3 generation)

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/uds822/resolveops.git
cd resolveops

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### 3. Environment Configuration
Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=nvapi-your-nvidia-key
OPENROUTER_API_KEY=sk-or-your-openrouter-key

# Storage Configuration
RESOLVEOPS_VECTOR_STORE=qdrant
RESOLVEOPS_QDRANT_PATH=./data/qdrant_db
RESOLVEOPS_BM25_PATH=./data/bm25_index.json
RESOLVEOPS_MANIFEST_PATH=./data/index_manifest.json

# Model Configuration
RESOLVEOPS_EMBED_MODEL=nvidia:nvidia/nemotron-3-embed-1b
RESOLVEOPS_RERANK_MODEL=nvidia:nvidia/nemotron-3.5-rerank-8b
RESOLVEOPS_LLM_MODEL=openrouter:deepseek/deepseek-chat
```

### 4. Ingest the Technical Corpus
Index the AWS technical corpus into Qdrant and BM25 with checkpoint tracking:

```bash
python scripts/ingest_corpus.py
```
*This indexes 770 chunks across AWS DynamoDB, Lambda, IAM, and S3 guides in ~120s with verified manifest checkpoints.*

### 5. Launch the Web Studio & REST API
Start the production Uvicorn server:

```powershell
$env:RESOLVEOPS_VECTOR_STORE="qdrant"
.venv\Scripts\uvicorn resolveops.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open **`http://localhost:8000`** in your browser to access the **ResolveOps RAG Studio**.

---

## 📡 REST API Reference

The FastAPI backend exposes the following endpoints (Interactive Swagger docs available at `http://localhost:8000/docs`):

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check with live indexed chunk telemetry. |
| `POST` | `/api/v1/query` | Citation-grounded RAG query endpoint returning answer, citations, and latency metrics. |
| `POST` | `/api/v1/ingest` | Incremental document text ingestion endpoint with automatic deduplication. |
| `DELETE` | `/api/v1/documents/{document_id}` | Atomically remove all indexed chunks belonging to a document. |

---

## 🧪 Testing

ResolveOps maintains an extensive test suite covering every layer of the architecture:

```bash
# Run all unit tests (offline, fast mocks)
pytest

# Run retrieval & storage tests
pytest tests/test_qdrant.py tests/test_manifest.py tests/test_hybrid.py -v

# Run generation & citation tests
pytest tests/test_context_builder.py tests/test_llm.py tests/test_service.py -v

# Run API lifecycle integration tests
pytest tests/test_api.py -v
```

---

## 📂 Repository Structure

```text
resolveops/
├── data/
│   ├── raw/aws/             # Raw AWS PDF manuals, HTML guides, and Markdown
│   ├── qdrant_db/           # Persistent local Qdrant collection (ignored in git)
│   ├── bm25_index.json      # Durable BM25 snapshot (ignored in git)
│   └── index_manifest.json  # 3-Way consistency manifest (ignored in git)
├── frontend/                # ResolveOps RAG Studio (Decoupled Web Client)
│   ├── index.html           # Modern dark-theme workspace
│   ├── styles.css           # Custom sleek scrollbars, flexbox layout, animations
│   └── app.js               # Reactive state manager, citation highlighter, telemetry
├── scripts/
│   ├── ingest_corpus.py     # Production batch ingestion orchestrator
│   ├── demo_rerank.py       # Empirical 2-stage retrieval demonstration
│   └── benchmark_with_reranker.py # Benchmark suite for reranking precision
├── src/resolveops/
│   ├── ingestion/           # Layout-aware parsers, Docling adapter, chunking
│   ├── retrieval/           # Persistent Qdrant, BM25, Hybrid RRF, Cross-Encoder
│   ├── generation/          # Grounded prompts, LLM adapters, refusal sanitizers
│   ├── service.py           # End-to-end RAG orchestrator facade
│   └── api/                 # Production FastAPI server & static file mount
└── tests/                   # 100% passing automated test suite
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
