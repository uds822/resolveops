r"""
ResolveOps Batch Corpus Ingestion Script.

Executes staged, resilient batch indexing into Qdrant & BM25 with
manifest-backed cross-store consistency checkpoints.

Usage:
    .venv\Scripts\python scripts/ingest_corpus.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

load_dotenv()

from resolveops.ingestion.chunking import chunk_document
from resolveops.ingestion.document_parser import parse_document
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import make_embedding_model
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.manifest import (
    IndexManifest,
    InconsistentIndexError,
    load_manifest,
    verify_index_consistency,
)
from resolveops.retrieval.vector_store import QdrantVectorStore

# ─────────────────────────────────────────────────────────────────────────────
# Corpus Definitions
# ─────────────────────────────────────────────────────────────────────────────

MARKDOWN_DOCS = [
    "data/raw/aws/markdown/TTL.md",
    "data/raw/aws/markdown/Programming.Errors.md",
    "data/raw/aws/markdown/Versioning.md",
    "data/raw/aws/markdown/Concepts.MultiAZ.md",
    "data/raw/aws/markdown/id_roles.md",
    "data/raw/aws/markdown/invocation-async.md",
    "data/raw/aws/markdown/object-lock.md",
    "data/raw/aws/markdown/id_credentials_mfa.md",
    "data/raw/aws/markdown/ec2-security-groups.md",
    "data/raw/aws/markdown/invocation-eventsourcemapping.md",
]

HTML_DOCS = [
    "data/raw/aws/html/Policy evaluation logic - AWS Identity and Access Management.html",
    "data/raw/aws/html/Security best practices in IAM - AWS Identity and Access Management.html",
    "data/raw/aws/html/Troubleshoot access denied (403 Forbidden) errors in Amazon S3 - Amazon Simple Storage Service.html",
    "data/raw/aws/html/Troubleshoot configuration issues in Lambda - AWS Lambda.html",
    "data/raw/aws/html/Troubleshoot invocation issues in Lambda - AWS Lambda.html",
]

# High-yield core chapters of massive AWS manuals (limits, releases, permissions, retries)
PDF_BATCHES = [
    ("data/raw/aws/pdf/dynamodb-dg.pdf", (50, 75), "DynamoDB Local & Partition Limits"),
    ("data/raw/aws/pdf/dynamodb-dg.pdf", (100, 125), "DynamoDB Indexes & Queries"),
    ("data/raw/aws/pdf/iam-ug.pdf",        (25, 50), "IAM Roles & Credentials"),
    ("data/raw/aws/pdf/iam-ug.pdf",        (100, 125), "IAM Policy Evaluation Logic"),
    ("data/raw/aws/pdf/lambda-dg.pdf",     (15, 40), "Lambda Async Invocations & Retries"),
    ("data/raw/aws/pdf/lambda-dg.pdf",     (80, 105), "Lambda Concurrency & Environment"),
    ("data/raw/aws/pdf/s3-userguide.pdf",  (25, 50), "S3 Permissions & Bucket Policies"),
    ("data/raw/aws/pdf/s3-userguide.pdf",  (100, 125), "S3 Object Lock & Versioning"),
]


def setup_persistent_retriever(
    db_path: Path,
    manifest_path: Path,
    bm25_path: Path,
) -> tuple[HybridRetriever, bool]:
    """Initialize or resume persistent Qdrant + BM25 stores with 3-way handshake verification."""
    embed_spec = os.environ.get("RESOLVEOPS_EMBED_MODEL", "nvidia:nvidia/nemotron-3-embed-1b")
    print(f"📦 [1/4] Initializing Cloud Embedding Model ({embed_spec})...")
    embedder = make_embedding_model(embed_spec)

    print(f"🗄️  [2/4] Connecting to Persistent Qdrant Store at: {db_path}...")
    vector_store = QdrantVectorStore(
        embedding_model=embedder,
        path=db_path,
        collection_name="resolveops_chunks",
    )

    resumed = False
    if manifest_path.is_file() and bm25_path.is_file():
        print(f"🔄 Resuming from existing manifest: {manifest_path}...")
        bm25 = BM25Index.load_snapshot(bm25_path)
        try:
            manifest = verify_index_consistency(manifest_path, vector_store, bm25)
            print(f"✅ Verified 3-way handshake! Existing chunks: {manifest.chunk_count} (gen: {manifest.generation_id})")
            resumed = True
        except InconsistentIndexError as err:
            print(f"⚠️  Manifest desync detected ({err}). Initializing fresh BM25 index.")
            bm25 = BM25Index()
    else:
        bm25 = BM25Index()

    retriever = HybridRetriever(vector_store, bm25, rrf_k=60)
    return retriever, resumed


def ingest_file_batch(
    retriever: HybridRetriever,
    filepath: Path,
    *,
    page_range: tuple[int, int] | None = None,
    label: str = "",
) -> int:
    """Parse, chunk, and index a single document batch into the hybrid retriever."""
    if not filepath.exists():
        print(f"  ❌ Missing file: {filepath}")
        return 0

    t0 = time.monotonic()
    doc = parse_document(filepath, page_range=page_range)
    chunks = chunk_document(doc)
    added = retriever.add_chunks(chunks)
    dt = time.monotonic() - t0

    desc = f" [{label}]" if label else ""
    pages_str = f" (pages {page_range[0]}-{page_range[1]})" if page_range else ""
    print(f"  ✓ {filepath.name}{desc}{pages_str}: +{added} chunks (took {dt:.2f}s)")
    return added


def main():
    print("\n" + "═" * 78)
    print("   RESOLVEOPS RESILIENT BATCH INGESTION PIPELINE")
    print("═" * 78 + "\n")

    db_path = project_root / "data" / "qdrant_db"
    manifest_path = project_root / "data" / "index_manifest.json"
    bm25_path = project_root / "data" / "bm25_index.json"

    retriever, resumed = setup_persistent_retriever(db_path, manifest_path, bm25_path)
    initial_count = retriever.count()

    total_added = 0
    t_start = time.monotonic()

    # ── Phase 1: Markdown Documents ───────────────────────────────────────────
    print(f"\n📂 PHASE 1: Ingesting {len(MARKDOWN_DOCS)} AWS Markdown Documents...")
    md_added = 0
    for idx, path_str in enumerate(MARKDOWN_DOCS, start=1):
        p = project_root / path_str
        added = ingest_file_batch(retriever, p, label=f"MD {idx}/{len(MARKDOWN_DOCS)}")
        md_added += added

    # Checkpoint after Markdown
    if md_added > 0:
        manifest = retriever.save_state(manifest_path, bm25_path)
        print(f"  💾 Checkpoint saved: {manifest.chunk_count} total chunks (gen: {manifest.generation_id})")
    total_added += md_added

    # ── Phase 2: HTML Troubleshooting Guides ─────────────────────────────────
    print(f"\n🌐 PHASE 2: Ingesting {len(HTML_DOCS)} AWS HTML Troubleshooting Guides...")
    html_added = 0
    for idx, path_str in enumerate(HTML_DOCS, start=1):
        p = project_root / path_str
        added = ingest_file_batch(retriever, p, label=f"HTML {idx}/{len(HTML_DOCS)}")
        html_added += added

    # Checkpoint after HTML
    if html_added > 0:
        manifest = retriever.save_state(manifest_path, bm25_path)
        print(f"  💾 Checkpoint saved: {manifest.chunk_count} total chunks (gen: {manifest.generation_id})")
    total_added += html_added

    # ── Phase 3: High-Yield AWS PDF Manual Chapters ──────────────────────────
    print(f"\n📑 PHASE 3: Ingesting {len(PDF_BATCHES)} Core AWS PDF Chapters...")
    pdf_added = 0
    for idx, (path_str, p_range, desc) in enumerate(PDF_BATCHES, start=1):
        p = project_root / path_str
        added = ingest_file_batch(
            retriever,
            p,
            page_range=p_range,
            label=f"PDF {idx}/{len(PDF_BATCHES)}: {desc}",
        )
        pdf_added += added
        # Checkpoint after each PDF batch
        if added > 0:
            manifest = retriever.save_state(manifest_path, bm25_path)
            print(f"  💾 Checkpoint saved: {manifest.chunk_count} total chunks (gen: {manifest.generation_id})")
    total_added += pdf_added

    total_time = time.monotonic() - t_start
    final_count = retriever.count()

    # Final verification handshake
    print("\n" + "─" * 78)
    print("🔒 Executing Final 3-Way Consistency Handshake...")
    verified_manifest = retriever.verify_consistency(manifest_path)

    print("═" * 78)
    print(f"🎉 INGESTION COMPLETE IN {total_time:.1f}s!")
    print(f"  • Initial Chunks:        {initial_count}")
    print(f"  • Newly Added Chunks:    {total_added}")
    print(f"  • Total Active Chunks:   {final_count}")
    print(f"  • Qdrant Database:       {db_path}")
    print(f"  • BM25 Snapshot:         {bm25_path}")
    print(f"  • Index Manifest:        {manifest_path}")
    print(f"  • Authoritative Gen ID:  {verified_manifest.generation_id}")
    print("═" * 78 + "\n")


if __name__ == "__main__":
    main()

