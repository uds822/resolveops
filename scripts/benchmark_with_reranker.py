r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        TWO-STAGE RETRIEVAL BENCHMARK (HYBRID + RERANKER) — ResolveOps        ║
║                                                                              ║
║  Stage 1: Hybrid Retrieval (Dense Vector + BM25 Sparse + RRF)                ║
║           Retrieves Top-10 candidate pool in < 50ms                          ║
║                                                                              ║
║  Stage 2: Cross-Encoder Reranking (NVIDIA Nemotron 3.5 8B)                   ║
║           Deep cross-attention query ↔ passage scoring (Top-5 returned)      ║
║                                                                              ║
║  Usage: .venv\Scripts\python scripts\benchmark_with_reranker.py              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from resolveops.ingestion.chunking import chunk_document
from resolveops.ingestion.document_parser import parse_document
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import make_embedding_model
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.reranker import make_reranker
from resolveops.retrieval.vector_store import InMemoryVectorStore

# ─────────────────────────────────────────────────────────────────────────────
# 1. Corpus Sources (Markdown + Multi-Page PDFs)
# ─────────────────────────────────────────────────────────────────────────────
MARKDOWN_FILES = [
    "data/raw/aws/markdown/TTL.md",
    "data/raw/aws/markdown/Programming.Errors.md",
    "data/raw/aws/markdown/Versioning.md",
    "data/raw/aws/markdown/Concepts.MultiAZ.md",
    "data/raw/aws/markdown/id_roles.md",
    "data/raw/aws/markdown/invocation-async.md",
    "data/raw/aws/markdown/object-lock.md",
    "data/raw/aws/markdown/id_credentials_mfa.md",
]

PDF_SOURCES = [
    ("data/raw/aws/pdf/dynamodb-dg.pdf",   57, 59),   # DynamoDB Local release notes + limits
    ("data/raw/aws/pdf/iam-ug.pdf",        30, 32),   # How IAM works & request components
    ("data/raw/aws/pdf/lambda-dg.pdf",     20, 22),   # Lambda invocation & error handling
    ("data/raw/aws/pdf/s3-userguide.pdf",  30, 32),   # S3 data processing, analytics & bucket access
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Benchmark Queries (Exact Technical, Conceptual, Edge Cases)
# ─────────────────────────────────────────────────────────────────────────────
QUERIES = [
    {
        "label": "EXACT TECHNICAL  — Error Code",
        "query": "AccessDeniedException 400 Signature Version 4 signing",
        "candidate_pool": 10,
        "top_k": 5,
        "emoji": "🔑",
    },
    {
        "label": "CONCEPTUAL       — Data Expiry",
        "query": "automatically expire and delete old items without consuming write throughput",
        "candidate_pool": 10,
        "top_k": 5,
        "emoji": "⏰",
    },
    {
        "label": "EXACT TECHNICAL  — DynamoDB Local Version",
        "query": "DynamoDB local version 3.3.2 pom.xml dependencies",
        "candidate_pool": 10,
        "top_k": 5,
        "emoji": "📦",
    },
    {
        "label": "CONCEPTUAL       — Lambda Failure Retry",
        "query": "what happens when a Lambda function fails and how are retries handled?",
        "candidate_pool": 10,
        "top_k": 5,
        "emoji": "🔄",
    },
    {
        "label": "MIXED            — IAM Role Delegation",
        "query": "IAM role assume cross-account delegation temporary credentials",
        "candidate_pool": 10,
        "top_k": 5,
        "emoji": "🔐",
    },
    {
        "label": "CONCEPTUAL       — S3 Bucket Access & Privacy",
        "query": "Are S3 bucket objects private by default and how to grant access permissions?",
        "candidate_pool": 10,
        "top_k": 5,
        "emoji": "🪣",
    },
]

TOP_CONTENT_PREVIEW_LINES = 5


def separator(char: str = "═", width: int = 84) -> str:
    return char * width


def format_rank_shift(initial: int, reranked: int) -> str:
    diff = initial - reranked
    if diff > 0:
        return f"Rank #{initial} ➔ Rank #{reranked} (+{diff} 🚀 PROMOTED)"
    elif diff < 0:
        return f"Rank #{initial} ➔ Rank #{reranked} ({diff} 🔻 DEMOTED)"
    else:
        return f"Rank #{initial} ➔ Rank #{reranked} (CONFIRMED #1 ✅)"


def print_comparison_table(stage1_results, rerank_results) -> None:
    print("\n  ┌───────────────────────────────────────────────────────────────────────────────────────────┐")
    print("  │ 📊 STAGE 1 (HYBRID) vs STAGE 2 (CROSS-ENCODER RERANKER) RANK AUDIT                         │")
    print("  ├──────────────┬──────────────┬──────────────┬──────────┬──────────────┬────────────────────────────┤")
    print("  │ Rerank Rank  │ Initial Rank │ Probability  │ Logit    │ Shift        │ Source File                │")
    print("  ├──────────────┼──────────────┼──────────────┼──────────┼──────────────┼────────────────────────────┤")

    for res in rerank_results:
        prob_pct = f"{res.score * 100:.1f}%"
        shift_num = res.initial_rank - res.reranked_rank
        shift_str = f"+{shift_num} 🚀" if shift_num > 0 else (f"{shift_num} 🔻" if shift_num < 0 else "  0 ━")
        src = Path(res.chunk.source).name[:26]
        print(f"  │  #{res.reranked_rank:<10} │  #{res.initial_rank:<10} │ {prob_pct:<12} │ {res.logit:+.3f}  │ {shift_str:<12} │ {src:<26} │")

    print("  └──────────────┴──────────────┴──────────────┴──────────┴──────────────┴────────────────────────────┘")


def main() -> None:
    print("\n" + separator("═"))
    print("  🚀  RESOLVEOPS TWO-STAGE RETRIEVAL BENCHMARK (HYBRID + RERANKER)")
    print(separator("═"))

    # ── 1. Initialize Models ──────────────────────────────────────────────────
    embed_model_name = "nvidia:nvidia/nemotron-3.5-embed-1b"
    rerank_model_name = "nvidia:nvidia/nemotron-3.5-rerank-8b"

    print(f"\n📡 [Stage 1] Initializing Bi-Encoder Embedding Model ({embed_model_name})...")
    t0 = time.monotonic()
    embedder = make_embedding_model(embed_model_name)
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25, rrf_k=60)
    print(f"   ✅ Hybrid Retriever ready in {time.monotonic() - t0:.2f}s (2048 dims, 8k context)")

    print(f"\n🎯 [Stage 2] Initializing Cross-Encoder Reranker ({rerank_model_name})...")
    t0 = time.monotonic()
    reranker = make_reranker(rerank_model_name)
    print(f"   ✅ NVIDIA Nemotron 8B Reranker ready in {time.monotonic() - t0:.2f}s")

    # ── 2. Ingest Markdown ────────────────────────────────────────────────────
    print(f"\n📁 Ingesting {len(MARKDOWN_FILES)} Markdown files...")
    md_chunks = 0
    for md_path_str in MARKDOWN_FILES:
        md_path = Path(md_path_str)
        if not md_path.exists():
            continue
        doc = parse_document(md_path)
        chunks = chunk_document(doc)
        added = retriever.add_chunks(chunks)
        md_chunks += added
        print(f"   ✅  {md_path.name:<40} {added:3d} chunks")

    # ── 3. Ingest PDFs with Docling Layout Analysis ───────────────────────────
    print(f"\n📄 Ingesting {len(PDF_SOURCES)} PDF ranges with Docling layout awareness...")
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from resolveops.ingestion.docling_pdf_parser import parse_pdf_with_docling

        pipeline_options = PdfPipelineOptions(do_ocr=False, queue_max_size=1)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        use_docling = True
        print("   ✅ Docling layout model ready (OCR disabled)")
    except Exception as exc:
        print(f"   ⚠️ Docling initialization failed ({exc}); falling back to clean PyMuPDF")
        converter = None
        use_docling = False

    pdf_chunks = 0
    for pdf_path_str, page_start, page_end in PDF_SOURCES:
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            continue

        if use_docling:
            doc = parse_pdf_with_docling(
                pdf_path,
                page_range=(page_start, page_end),
                converter=converter,
            )
            chunks = chunk_document(doc)
            added = retriever.add_chunks(chunks)
        else:
            all_added = 0
            for page_num in range(page_start, page_end + 1):
                doc = parse_document(pdf_path, page_number=page_num, pdf_engine="pymupdf")
                chunks = chunk_document(doc)
                all_added += retriever.add_chunks(chunks)
            added = all_added

        pdf_chunks += added
        print(f"   ✅  {pdf_path.name:<40} pages {page_start}-{page_end}  →  {added:3d} chunks")

    total_chunks = retriever.count()
    print(f"\n{separator('─')}")
    print(f"  📊 CORPUS INDEXED: {total_chunks} chunks ({md_chunks} markdown + {pdf_chunks} PDF)")
    print(separator('─'))

    # ── 4. Run Two-Stage Retrieval on Benchmark Queries ───────────────────────
    print("\n\n" + separator("═"))
    print("  🔍  RUNNING TWO-STAGE RETRIEVAL ON BENCHMARK QUERIES")
    print(separator("═"))

    for q in QUERIES:
        print("\n" + separator())
        print(f" {q['emoji']}  QUERY [{q['label']}]")
        print(f"    Text: \"{q['query']}\"")
        print(f"    Pipeline Funnel: Stage 1 retrieves top {q['candidate_pool']} candidates ➔ Stage 2 reranks to top {q['top_k']}")
        print(separator())

        # Stage 1: Hybrid Retrieval
        t_stage1 = time.monotonic()
        stage1_candidates = retriever.search(q["query"], top_k=q["candidate_pool"])
        ms_stage1 = (time.monotonic() - t_stage1) * 1000

        print(f"\n  ⚡ [Stage 1] Hybrid Search retrieved {len(stage1_candidates)} candidates in {ms_stage1:.1f}ms")

        # Stage 2: Cross-Encoder Reranking
        candidate_chunks = [res.chunk for res in stage1_candidates]
        t_stage2 = time.monotonic()
        rerank_results = reranker.rerank(q["query"], candidate_chunks, top_k=q["top_k"])
        ms_stage2 = (time.monotonic() - t_stage2) * 1000

        print(f"  🎯 [Stage 2] Cross-Encoder Reranked top {len(rerank_results)} chunks in {ms_stage2:.1f}ms")
        print(f"  ⏱️  Total Funnel Latency: {ms_stage1 + ms_stage2:.1f}ms")

        # Detailed cards for reranked results
        for res in rerank_results:
            medal = "🥇" if res.reranked_rank == 1 else "🥈" if res.reranked_rank == 2 else "🥉" if res.reranked_rank == 3 else f" #{res.reranked_rank}"
            shift_display = format_rank_shift(res.initial_rank, res.reranked_rank)
            prob_pct = res.score * 100.0

            # Match initial stage 1 candidate to show dense & bm25 scores
            init_cand = stage1_candidates[res.initial_rank - 1] if res.initial_rank <= len(stage1_candidates) else None

            print(f"\n  {medal}  Final Rank #{res.reranked_rank}  │  Confidence: {prob_pct:.2f}%  (Logit = {res.logit:+.3f})")
            print(f"      ├─ Movement:    {shift_display}")
            if init_cand:
                print(f"      ├─ Stage 1 RRF: RRF Score = {init_cand.score:.6f} (Dense #{init_cand.dense_rank}, BM25 #{init_cand.bm25_rank or '—'})")
            print(f"      ├─ Source:      {Path(res.chunk.source).name} (Pages: {res.chunk.page_numbers})")
            print(f"      ├─ Context:     {' ▸ '.join(res.chunk.heading_context) or '(no heading)'}")
            print(f"      └─ Content:")
            for line in res.chunk.content.strip().splitlines()[:TOP_CONTENT_PREVIEW_LINES]:
                if line.strip():
                    print(f"            {line.strip()[:88]}")
            if len(res.chunk.content.strip().splitlines()) > TOP_CONTENT_PREVIEW_LINES:
                print(f"            ...")

        # Comparison audit table
        print_comparison_table(stage1_candidates, rerank_results)

    # ── 5. How to Analyze Section ─────────────────────────────────────────────
    print(f"""
{separator("═")}
  🎓  HOW TO ANALYZE THESE BENCHMARK RESULTS (ENGINEER'S GUIDE)
{separator("─")}
  1. THE RANK SHIFT COLUMN (Initial Rank ➔ Final Rank):
     • Look for (+N 🚀 PROMOTED): A chunk that was buried at Rank #3, #4, or #6
       in Hybrid search, but the Cross-Encoder recognized as the DIRECT answer to
       the user's question and promoted to #1.
     • Look for (-N 🔻 DEMOTED): A chunk that scored high in BM25 because of
       repeated keywords, but was actually irrelevant or off-topic. The 8B Cross-Encoder
       read the passage in full context and demoted it.
     • Look for (CONFIRMED #1 ✅): Hybrid search and Cross-Encoder both strongly agreed.

  2. CONFIDENCE PERCENTAGE & LOGIT:
     • Probability > 70% (Logit > +0.85): High confidence direct answer.
     • Probability 20% - 60% (Logit -1.38 to +0.40): Contextually relevant or supporting text.
     • Probability < 15% (Logit < -1.73): Low relevance distractor.

  3. TWO-STAGE LATENCY BUDGET:
     • Stage 1 (Hybrid) runs in ~10-40ms on thousands of chunks to prune down to 10 candidates.
     • Stage 2 (Cross-Encoder) runs in ~200-500ms on ONLY the 10 candidates.
     • You get the deep intelligence of an 8-Billion parameter transformer without having
       to pay the latency penalty of scoring your entire 10,000-chunk database with it!
{separator("═")}
""")


if __name__ == "__main__":
    main()

