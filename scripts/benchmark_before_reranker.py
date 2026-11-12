r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PRE-RERANKER HYBRID SEARCH BENCHMARK — ResolveOps                 ║
║                                                                              ║
║  Purpose: Visually observe what the Hybrid Retriever (Dense + BM25 + RRF)  ║
║           returns BEFORE the Cross-Encoder Reranker touches the results.   ║
║                                                                              ║
║  This benchmark proves WHY a reranker is needed in production:              ║
║    1. RRF score margins between Rank #1 and Rank #3 are tiny               ║
║    2. Topically related but non-answering chunks sneak into top results     ║
║    3. The right chunk sometimes lands at Rank #2 or #3, not Rank #1        ║
║                                                                              ║
║  Usage: .venv\Scripts\python scripts\benchmark_before_reranker.py          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add src to Python path so we can import resolveops without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolveops.ingestion.chunking import chunk_document
from resolveops.ingestion.document_parser import parse_document
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import make_embedding_model
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.vector_store import InMemoryVectorStore

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Sources to Index
# We mix Markdown files + PDF pages to simulate a real multi-format corpus
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

# PDF page ranges to ingest from each guide (using deep-learning Docling layout)
PDF_SOURCES = [
    ("data/raw/aws/pdf/dynamodb-dg.pdf",   57, 59),   # DynamoDB Local release notes + limits
    ("data/raw/aws/pdf/iam-ug.pdf",        30, 32),   # How IAM works & request components
    ("data/raw/aws/pdf/lambda-dg.pdf",     20, 22),   # Lambda invocation & error handling
    ("data/raw/aws/pdf/s3-userguide.pdf",  30, 32),   # S3 data processing, analytics & bucket access
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Benchmark Queries
# Mix of exact technical, conceptual, and mixed queries
# ─────────────────────────────────────────────────────────────────────────────
QUERIES = [
    {
        "label": "EXACT TECHNICAL  — Error Code",
        "query": "AccessDeniedException 400 Signature Version 4 signing",
        "top_k": 5,
        "emoji": "🔑",
    },
    {
        "label": "CONCEPTUAL       — Data Expiry",
        "query": "automatically expire and delete old items without consuming write throughput",
        "top_k": 5,
        "emoji": "⏰",
    },
    {
        "label": "EXACT TECHNICAL  — DynamoDB Local Version",
        "query": "DynamoDB local version 3.3.2 pom.xml dependencies",
        "top_k": 5,
        "emoji": "📦",
    },
    {
        "label": "CONCEPTUAL       — Lambda Failure Retry",
        "query": "what happens when a Lambda function fails and how are retries handled?",
        "top_k": 5,
        "emoji": "🔄",
    },
    {
        "label": "MIXED            — IAM Role Delegation",
        "query": "IAM role assume cross-account delegation temporary credentials",
        "top_k": 5,
        "emoji": "🔐",
    },
    {
        "label": "CONCEPTUAL       — S3 Bucket Access & Privacy",
        "query": "Are S3 bucket objects private by default and how to grant access permissions?",
        "top_k": 5,
        "emoji": "🪣",
    },
]

TOP_CONTENT_PREVIEW_LINES = 6   # How many lines of chunk content to show


def separator(char: str = "═", width: int = 80) -> str:
    return char * width


def print_query_banner(label: str, query: str, top_k: int, emoji: str) -> None:
    print("\n" + separator())
    print(f" {emoji}  QUERY [{label}]")
    print(f"    Text  : \"{query}\"")
    print(f"    Top-K : {top_k}")
    print(separator())


def print_result(rank: int, res) -> None:
    chunk = res.chunk
    content_lines = chunk.content.strip().splitlines()
    preview_lines = content_lines[:TOP_CONTENT_PREVIEW_LINES]
    ellipsis = "..." if len(content_lines) > TOP_CONTENT_PREVIEW_LINES else ""

    # Score bar: visual representation of RRF score magnitude
    bar_len = int(res.score * 10000)
    score_bar = "█" * min(bar_len, 30)

    print(f"\n  {'🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '  #' + str(rank)} "
          f" Rank #{rank}  │  RRF Score: {res.score:.6f}  {score_bar}")
    print(f"      ├─ Dense Rank:  #{res.dense_rank:2}  Cosine Sim = {res.dense_score or 0.0:+.4f}")
    print(f"      ├─ BM25  Rank:  #{res.bm25_rank:2}  Okapi BM25 = {res.bm25_score or 0.0:+.4f}" if res.bm25_rank else
          f"      ├─ BM25  Rank:  — (not in BM25 candidate pool)")
    print(f"      ├─ Kind:        {chunk.kind.upper():<10}  Pages: {chunk.page_numbers}")
    print(f"      ├─ Source:      {Path(chunk.source).name}")
    print(f"      ├─ Context:     {' ▸ '.join(chunk.heading_context) or '(no heading)'}")
    print(f"      └─ Content:")
    for line in preview_lines:
        stripped = line.strip()
        if stripped:
            print(f"            {stripped[:90]}")
    if ellipsis:
        print(f"            ...")
    print()


def print_score_gap_analysis(results) -> None:
    """Prints the score gap table - this is the key observation for why reranking is needed."""
    print("  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │  📊 SCORE GAP ANALYSIS (Margin between consecutive ranks)          │")
    print("  ├───────┬────────────┬────────────┬──────────────────────────────────┤")
    print("  │ Rank  │ RRF Score  │  Gap ↓     │ Observation                      │")
    print("  ├───────┼────────────┼────────────┼──────────────────────────────────┤")

    for i, res in enumerate(results):
        gap = results[i - 1].score - res.score if i > 0 else 0.0
        is_tiny = gap > 0 and gap < 0.001
        flag = "⚠️  TINY GAP — reranker can flip order!" if is_tiny else ("" if i == 0 else "OK")
        print(f"  │  #{i+1:<4} │ {res.score:.6f} │ -{gap:.6f} │ {flag:<32} │")

    print("  └───────┴────────────┴────────────┴──────────────────────────────────┘")


def print_dual_coverage_analysis(results) -> None:
    """Shows which chunks were corroborated by both engines vs only one."""
    print("\n  📡 ENGINE COVERAGE PER RESULT:")
    for i, res in enumerate(results):
        both = res.dense_rank is not None and res.bm25_rank is not None
        only_dense = res.dense_rank is not None and res.bm25_rank is None
        only_bm25 = res.bm25_rank is not None and res.dense_rank is None
        tag = "✅ BOTH Dense + BM25" if both else ("🔵 Dense only" if only_dense else "🟡 BM25 only")
        print(f"      #{i+1}  {tag}  →  {Path(res.chunk.source).name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Index Corpus, Then Run Benchmark Queries
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + separator("═"))
    print("  🚀  PRE-RERANKER HYBRID SEARCH BENCHMARK — ResolveOps")
    print(separator("═"))

    # ── 1. Build Embedding Model & Retriever ─────────────────────────────────
    model_name = "nvidia:nvidia/nemotron-3.5-embed-1b"
    print(f"\n📡 Initializing cloud embedding model ({model_name})...")
    t0 = time.monotonic()
    embedder = make_embedding_model(model_name)
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25, rrf_k=60)
    print(f"   ✅ Ready in {time.monotonic() - t0:.2f}s (2048 dims, 8k context window)")

    # ── 2. Ingest Markdown Files ──────────────────────────────────────────────
    print(f"\n📁 Ingesting {len(MARKDOWN_FILES)} Markdown files...")
    md_chunks_total = 0
    for md_path_str in MARKDOWN_FILES:
        md_path = Path(md_path_str)
        if not md_path.exists():
            print(f"   ⚠️  Skipping (not found): {md_path_str}")
            continue
        doc = parse_document(md_path)
        chunks = chunk_document(doc)
        added = retriever.add_chunks(chunks)
        md_chunks_total += added
        print(f"   ✅  {md_path.name:<42}  {added:3d} chunks indexed")

    # ── 3. Ingest PDF Pages with Docling Layout Analysis ──────────────────────
    print(f"\n📄 Ingesting PDF page ranges with Docling (Layout-Aware)...")
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
        print("   ✅ Docling converter ready (OCR disabled, layout model active)")
    except Exception as exc:
        print(f"   ⚠️ Docling initialization failed ({exc}); falling back to clean PyMuPDF")
        converter = None
        use_docling = False

    pdf_chunks_total = 0
    for pdf_path_str, page_start, page_end in PDF_SOURCES:
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            print(f"   ⚠️  Skipping (not found): {pdf_path_str}")
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

        pdf_chunks_total += added
        print(f"   ✅  {pdf_path.name:<42}  pages {page_start}-{page_end}  →  {added:3d} chunks indexed")

    # ── 4. Corpus Summary ─────────────────────────────────────────────────────
    total_indexed = retriever.count()
    print(f"\n{separator('─')}")
    print(f"  📊 CORPUS SUMMARY")
    print(f"      Markdown chunks:  {md_chunks_total}")
    print(f"      PDF chunks:       {pdf_chunks_total}")
    print(f"      Total indexed:    {total_indexed} chunks in Hybrid Retriever")
    print(f"      Vector store:     {vector_store.count()} dense vectors (NumPy float32)")
    print(f"      BM25 index:       {bm25.count()} docs with technical tokenization")
    print(separator("─"))

    # ── 5. Run Benchmark Queries ──────────────────────────────────────────────
    print("\n\n" + separator("═"))
    print("  🔍  BENCHMARK QUERIES — OBSERVE YOUR HYBRID RETRIEVER")
    print(separator("═"))

    for q in QUERIES:
        print_query_banner(q["label"], q["query"], q["top_k"], q["emoji"])

        t_search = time.monotonic()
        results = retriever.search(q["query"], top_k=q["top_k"])
        elapsed_ms = (time.monotonic() - t_search) * 1000

        print(f"\n  ⚡ Search completed in {elapsed_ms:.1f} ms  │  {len(results)} results returned\n")

        for rank, res in enumerate(results, 1):
            print_result(rank, res)

        if results:
            print_score_gap_analysis(results)
            print()
            print_dual_coverage_analysis(results)

        print("\n" + separator("─"))

    # ── 6. Final Observation Note ─────────────────────────────────────────────
    print(f"""
{separator("═")}
  🧠  WHAT TO OBSERVE AS AN ENGINEER:
{separator("─")}
  1. SCORE GAPS:  Look at the ⚠️ TINY GAP warnings above.
                  When Rank #1 and Rank #2 differ by < 0.001, a reranker
                  can dramatically flip their order based on deep reasoning.

  2. ENGINE COVERAGE:  Chunks with ✅ BOTH Dense + BM25 always win.
                       Chunks with only one engine are more fragile candidates.

  3. SOURCE DIVERSITY:  With a real corpus of {total_indexed} chunks from 4 PDFs + 8 Markdown
                        files, observe whether irrelevant sources sneak into top-5.
                        Any off-topic chunk in the top-3 is a reranker use case!

  4. COSINE SCORES:    Notice how some cosine similarities are surprisingly LOW
                       (0.25–0.40) even for Rank #1 results. This is the
                       "embedding blind spot" for exact technical terms.
{separator("═")}
""")


if __name__ == "__main__":
    main()

