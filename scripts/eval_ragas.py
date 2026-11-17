#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               RESOLVEOPS RAGAS OFFLINE BENCHMARK EVALUATOR                   ║
║                                                                              ║
║  Runs the 20-item Golden Evaluation Benchmark across 5 Core Metrics:         ║
║    1. Context Precision (MAP@K on retrieved evidence chunks)                 ║
║    2. Context Recall (coverage of essential reference facts in context)      ║
║    3. Faithfulness (ratio of answer claims entailed by context)              ║
║    4. Answer Relevance (semantic relevance of answer to user query)          ║
║    5. Abstention Correctness (appropriate refusal on adversarial queries)    ║
║                                                                              ║
║  Usage:                                                                      ║
║    .venv\Scripts\python scripts/eval_ragas.py --mock --smoke                 ║
║    .venv\Scripts\python scripts/eval_ragas.py --mock                         ║
║    .venv\Scripts\python scripts/eval_ragas.py                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from resolveops.evals.models import EvalExample
from resolveops.evals.ragas_evaluator import RagasEvaluator, format_markdown_report
from resolveops.generation.context_builder import build_grounded_prompt
from resolveops.generation.grounding_verifier import MockGroundingVerifier
from resolveops.generation.llm import MockLLMClient, make_llm_client
from resolveops.ingestion.chunking import chunk_document
from resolveops.ingestion.document_parser import parse_document
from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import MockEmbeddingModel, make_embedding_model
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.reranker import MockReranker, make_reranker
from resolveops.retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore
from resolveops.service import RAGResponse, RAGService


def build_offline_benchmark_service() -> RAGService:
    """Construct an isolated in-memory RAG pipeline loaded with AWS Markdown corpus."""
    print("📦 Building offline in-memory RAG test pipeline...")
    embedder = MockEmbeddingModel(dimension=384)
    vector_store = InMemoryVectorStore(embedder)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store, bm25, rrf_k=60)

    # Ingest markdown files from data/raw/aws/markdown
    md_dir = Path("data/raw/aws/markdown")
    if md_dir.exists():
        for md_file in md_dir.glob("*.md"):
            doc = parse_document(md_file)
            chunks = chunk_document(doc)
            retriever.add_chunks(chunks)

    reranker = MockReranker()
    llm_client = MockLLMClient()
    verifier = MockGroundingVerifier()

    return RAGService(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
        grounding_verifier=verifier,
    )


def build_production_service() -> RAGService:
    """Load default configured service (Qdrant + BM25 + Nemotron/DeepSeek)."""
    from resolveops.api.main import build_default_rag_service

    service, _ = build_default_rag_service()
    return service


def main() -> None:
    parser = argparse.ArgumentParser(description="ResolveOps RAGAS Offline Evaluator")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_golden_set.json",
        help="Path to evaluation golden dataset JSON",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test on the first 5 examples only",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/offline mode without live LLM API calls",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/ragas_benchmark_report.md",
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    examples = [EvalExample(**item) for item in raw_items]
    if args.smoke:
        examples = examples[:5]
        print(f"💨 Running in SMOKE mode: evaluating {len(examples)} examples.")
    else:
        print(f"📊 Loaded {len(examples)} evaluation examples from {dataset_path.name}.")

    # Build Service
    if args.mock:
        service = build_offline_benchmark_service()
        evaluator = RagasEvaluator(
            embedding_model=MockEmbeddingModel(384),
            judge_client=MockLLMClient(),
            grounding_verifier=MockGroundingVerifier(),
        )
    else:
        try:
            service = build_production_service()
            evaluator = RagasEvaluator(
                embedding_model=service.retriever.vector_store.embedding_model,
                judge_client=service.llm_client,
                grounding_verifier=service.grounding_verifier,
            )
        except Exception as exc:
            print(f"⚠️ Failed to load production service ({exc}); falling back to offline pipeline.")
            service = build_offline_benchmark_service()
            evaluator = RagasEvaluator(
                embedding_model=MockEmbeddingModel(384),
                judge_client=MockLLMClient(),
                grounding_verifier=MockGroundingVerifier(),
            )

    print("\n" + "═" * 86)
    print("  🚀 EXECUTING RESOLVEOPS RAGAS OFFLINE BENCHMARK")
    print("═" * 86 + "\n")

    t_start = time.monotonic()

    def on_progress(idx: int, total: int, res):
        q_trunc = res.question[:50] + "..." if len(res.question) > 50 else res.question
        m = res.metrics
        cp = f"{m.context_precision*100:.0f}%" if m.context_precision is not None else "-"
        cr = f"{m.context_recall*100:.0f}%" if m.context_recall is not None else "-"
        faith = f"{m.faithfulness*100:.0f}%" if m.faithfulness is not None else "-"
        rel = f"{m.answer_relevance*100:.0f}%" if m.answer_relevance is not None else "-"
        abst = f"{m.abstention_correctness*100:.0f}%" if m.abstention_correctness is not None else "-"
        print(f"  [{idx:02d}/{total:02d}] {res.example_id:<18} | CP:{cp:>4} | CR:{cr:>4} | F:{faith:>4} | Rel:{rel:>4} | Abst:{abst:>4} | {q_trunc}")

    report = evaluator.evaluate_dataset(examples, service, on_progress=on_progress)
    elapsed = time.monotonic() - t_start

    print("\n" + "═" * 86)
    print("  📈 BENCHMARK SUMMARY & RELEASE GATE VERIFICATION")
    print("═" * 86)
    print(f"  Total Queries Evaluated:    {report.total_examples} ({report.answerable_count} answerable, {report.unanswerable_count} adversarial)")
    print(f"  Mean Context Precision:    {report.mean_context_precision * 100:.1f}% (Gate: >= 70%)")
    print(f"  Mean Context Recall:       {report.mean_context_recall * 100:.1f}% (Gate: >= 75%)")
    print(f"  Mean Faithfulness:         {report.mean_faithfulness * 100:.1f}% (Gate: >= 90%)")
    print(f"  Mean Answer Relevance:     {report.mean_answer_relevance * 100:.1f}% (Gate: >= 80%)")
    print(f"  Mean Abstention Accuracy:  {report.abstention_accuracy * 100:.1f}% (Gate: >= 80%)")
    print(f"  Mean End-to-End Latency:   {report.mean_latency_ms:.1f}ms")
    print(f"  Total Duration:            {elapsed:.2f}s")
    print("─" * 86)

    gate_status = "🟢 PASS (Approved for Release)" if report.release_gate_passed else "🔴 FAIL (Regressions Found)"
    print(f"  Release Gate Status:       {gate_status}")
    for gn in report.gate_notes:
        print(f"    • {gn}")
    print("═" * 86 + "\n")

    # Generate Markdown Report
    md_content = format_markdown_report(report)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    print(f"📝 Full benchmark report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

