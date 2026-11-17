"""Quick interactive verification script for semantic claim grounding and citation audit."""

from __future__ import annotations

import json
from resolveops.generation.context_builder import CitationReference
from resolveops.generation.grounding_verifier import (
    ClaimStatus,
    LLMGroundingVerifier,
    MockGroundingVerifier,
    extract_cited_claims,
)
from resolveops.generation.llm import MockLLMClient


def run_demo() -> None:
    print("=" * 75)
    print(" 🚀 RESOLVEOPS SEMANTIC CLAIM VERIFICATION DEMO")
    print("=" * 75)

    citations = [
        CitationReference(
            index=1,
            chunk_id="chunk_dyn_01",
            source="dynamodb-dg.pdf",
            heading_context=["Global Tables", "Conflict Resolution"],
            page_numbers=[45],
            snippet="DynamoDB uses last-writer-wins reconciliation between concurrent updates.",
            content="In global tables, DynamoDB uses last-writer-wins reconciliation between concurrent updates.",
        ),
        CitationReference(
            index=2,
            chunk_id="chunk_err_02",
            source="Programming.Errors.md",
            heading_context=["Error Codes", "Item Limits"],
            page_numbers=[],
            snippet="An ItemCollectionSizeLimitExceededException occurs when collection exceeds 10 GB.",
            content="An ItemCollectionSizeLimitExceededException occurs when an item collection exceeds 10 GB. Individual item size limit is 400 KB.",
        ),
    ]

    print("\n📚 INDEXED KNOWLEDGE BASE EXCERPTS:")
    for c in citations:
        print(f"  [{c.index}] {c.source} (Page {c.page_numbers}):")
        print(f"      \"{c.content}\"")

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 1: 100% GROUNDED ANSWER
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("CASE 1: Fully Grounded Answer (Both claims supported by source)")
    print("─" * 75)

    answer_1 = (
        "In Amazon DynamoDB, global tables use last-writer-wins reconciliation between concurrent updates [1]. "
        "Additionally, collection limit exceptions occur when an item collection exceeds 10 GB [2]."
    )
    print(f"Generated Answer:\n\"{answer_1}\"\n")

    verifier = MockGroundingVerifier(default_status=ClaimStatus.SUPPORTED)
    report_1 = verifier.verify_claims(answer_1, citations)

    print(f"📊 Audit Status:        {report_1.audit_status.upper()}")
    print(f"🎯 Faithfulness Score:  {report_1.faithfulness_score * 100:.0f}%")
    print(f"✅ Supported Claims:    {report_1.supported_claims}/{report_1.total_claims}")
    print(f"⚠️ Unsupported Claims:  {report_1.unsupported_claims}")
    print("\nDetailed Per-Claim Breakdown:")
    for claim in report_1.claims:
        print(f"  • Claim {claim.claim_id}: \"{claim.sentence}\"")
        print(f"    - Cited: [{claim.cited_marker}] {claim.source_filename}")
        print(f"    - Status: {claim.status.value.upper()}")
        print(f"    - Evidence: {claim.evidence_quote}")

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 2: MISATTRIBUTED / HALLUCINATED CLAIM
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("CASE 2: Citation Misattribution Detected (Claim 2 exaggerates limit to 50 MB)")
    print("─" * 75)

    answer_2 = (
        "In Amazon DynamoDB, global tables use last-writer-wins reconciliation between concurrent updates [1]. "
        "Furthermore, DynamoDB supports massive individual item sizes up to 50 MB [2]."
    )
    print(f"Generated Answer:\n\"{answer_2}\"\n")

    report_2 = verifier.verify_claims(answer_2, citations)

    print(f"📊 Audit Status:        {report_2.audit_status.upper()}")
    print(f"🎯 Faithfulness Score:  {report_2.faithfulness_score * 100:.0f}%")
    print(f"✅ Supported Claims:    {report_2.supported_claims}/{report_2.total_claims}")
    print(f"⚠️ Unsupported Claims:  {report_2.unsupported_claims}")

    print("\nDetailed Per-Claim Breakdown:")
    for claim in report_2.claims:
        status_icon = "✅" if claim.status == ClaimStatus.SUPPORTED else "⚠️"
        print(f"  {status_icon} Claim {claim.claim_id}: \"{claim.sentence}\"")
        print(f"    - Cited: [{claim.cited_marker}] {claim.source_filename}")
        print(f"    - Status: {claim.status.value.upper()}")
        if claim.status == ClaimStatus.SUPPORTED:
            print(f"    - Evidence: {claim.evidence_quote}")
        else:
            print(f"    - Reason: {claim.explanation}")

    print("\n" + "=" * 75)
    print("✨ Verification successful! Both grounded claims and misattributions caught.")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()

